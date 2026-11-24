// The arithmetic behind Part 2's three kernels: how work is cut into programs, what each program
// moves, what fusing steps saves, and how attention's split softmax is put back together.
import { shapeFloor } from "./challenge";
import {
	CONTEXT_TOKENS,
	D_EXPERT,
	floorUs,
	getProjection,
	H200,
	HEAD_DIM,
	HIDDEN,
	KV_HEADS,
	KV_SIZE,
	kvBytes,
	memoryUs,
	N_EXPERTS,
	projectionCost,
	Q_HEADS,
	TOP_K,
} from "./model";
import { output, products, W, X } from "./toy";

const cdiv = (a: number, b: number) => Math.ceil(a / b);

export const ON_CHIP_BYTES = H200.l2Bytes + H200.sms * (H200.sharedBytesPerSm + H200.registerBytesPerSm);

/** Triton runs each program as this many warps unless told otherwise: triton.Config's default
 *  (triton-lang.org/main/python-api/generated/triton.Config.html). */
export const TRITON_WARPS = 4;

/** NVIDIA's CUDA graphs post (developer.nvidia.com/blog/cuda-graphs) measures a 2.9 µs kernel on a
 *  V100. Waiting after each launch exposes 6.7 µs of CPU and GPU overhead per kernel. Queuing 20
 *  launches asynchronously into one stream brings the total to 3.8 µs per kernel, and replaying
 *  those 20 launches as a graph brings it to 3.4 µs per kernel. */
export const GRAPH_BLOG_US = { kernel: 2.9, launched: 9.6, async: 3.8, graph: 3.4 } as const;

/** Kernel 1 on the toy: one program per output row, each reading its row and the whole input. */
export const toyProgram = (row: number) => ({
	row,
	w: W[row],
	x: X,
	products: products(row),
	y: output(row),
});

export const toyTrips = {
	weightLoads: W.length * X.length,
	inputLoads: W.length * X.length,
	stores: W.length,
};

/** The toy's rows cut into tiles of `block`: which rows each program owns, and which are past the edge. */
export function toyTiles(block: number) {
	return Array.from({ length: cdiv(W.length, block) }, (_, program) =>
		Array.from({ length: block }, (_, i) => {
			const row = program * block + i;
			return { row, live: row < W.length };
		}),
	);
}

/** One choice of tile for kernel 1 at the real size: rows per program and columns per step. */
export const PROJ_BLOCK_N = 32;
export const PROJ_BLOCK_K = 128;
export const TILE_CHOICES = [PROJ_BLOCK_N, 16, 8, 1] as const;

/** Kernel 1 launched on a projection: programs, the loop's steps, and the weights each program reads. */
export function projectionGrid(
	id: Parameters<typeof getProjection>[0],
	blockN = PROJ_BLOCK_N,
	blockK = PROJ_BLOCK_K,
) {
	const p = getProjection(id);
	const programs = cdiv(p.rows, blockN);
	return {
		rows: p.rows,
		cols: p.cols,
		programs,
		idle: Math.max(0, H200.sms - programs),
		steps: cdiv(p.cols, blockK),
		bytesPerProgram: blockN * p.cols * p.bytesPerWeight,
		bytesPerStep: blockN * blockK * p.bytesPerWeight,
	};
}

/** Kernel 1 with an evenly distributed grid. The warp count is an upper bound capped by hardware. */
export function tileTradeoff(blockN: number, id = "pooler") {
	const { programs, idle: idleSms, bytesPerProgram, steps } = projectionGrid(id, blockN);
	const maxProgramsPerSm = cdiv(programs, H200.sms);
	return {
		blockN,
		programs,
		idleSms,
		maxProgramsPerSm,
		minProgramsPerSm: Math.floor(programs / H200.sms),
		warpsPerSm: Math.min(maxProgramsPerSm * TRITON_WARPS, H200.maxThreadsPerSm / H200.warpSize),
		tokenLoads: programs,
		bytesPerProgram,
		steps,
	};
}

/** Kernel 2, the gated output norm, as separate kernels or one: each trip moves one list of a token's
 *  attention output (16 heads of 64 numbers, 16-bit). */
export type FuseKernel = { name: string; reads: string[]; writes: string[] };
export const NORM_GATE_UNFUSED: FuseKernel[] = [
	{ name: "normalize", reads: ["result"], writes: ["normalized"] },
	{ name: "squash the gate", reads: ["gate"], writes: ["squashed"] },
	{ name: "multiply", reads: ["normalized", "squashed"], writes: ["output"] },
];
export const NORM_GATE_FUSED: FuseKernel[] = [
	{ name: "normalize and gate", reads: ["result", "gate"], writes: ["output"] },
];
export const trips = (ks: FuseKernel[]) => ks.reduce((n, k) => n + k.reads.length + k.writes.length, 0);
/** Bytes in one trip for one token: its attention output on this GPU. */
export const NORM_GATE_LIST_BYTES = Q_HEADS * HEAD_DIM * 2;

/** Kernel 3's grid: one program per query head and slice of the context. */
export const ATTN_SPLITS = 8;
export const KV_GROUP = Q_HEADS / KV_HEADS;
export function attentionGrid(splits: number, context = CONTEXT_TOKENS) {
	const sliceTokens = cdiv(context, splits);
	return {
		programs: Q_HEADS * splits,
		idle: Math.max(0, H200.sms - Q_HEADS * splits),
		sliceTokens,
		// A key and a value per token for this head's key-value head, one byte a number.
		bytesPerProgram: sliceTokens * 2 * HEAD_DIM,
		bytesPerHead: context * 2 * HEAD_DIM,
	};
}

export const GQA_SPLITS = 64;

/** One program per KV head and slice, reading its keys and values once for all KV_GROUP query heads. */
export function attentionGroupGrid(splits: number, context = CONTEXT_TOKENS) {
	const programs = KV_HEADS * splits;
	const sliceTokens = cdiv(context, splits);
	const bytesPerToken = 2 * HEAD_DIM;
	const bytesPerGroup = context * bytesPerToken;
	return {
		groups: KV_HEADS,
		programs,
		idle: Math.max(0, H200.sms - programs),
		sliceTokens,
		bytesPerProgram: sliceTokens * bytesPerToken,
		bytesPerGroup,
		totalBytes: KV_HEADS * bytesPerGroup,
	};
}

type Partial = { max: number; sum: number; out: number[] };

/** One slice of attention: the running maximum, the sum of shifted exponentials, and the slice's own blend. */
function partialSoftmax(scores: number[], values: number[][]): Partial {
	const max = Math.max(...scores);
	const p = scores.map((s) => Math.exp(s - max));
	const sum = p.reduce((a, b) => a + b, 0);
	const out = values[0].map((_, d) => values.reduce((acc, v, j) => acc + p[j] * v[d], 0) / sum);
	return { max, sum, out };
}

/** What kernel 3 stores for a slice: its blend (0 ÷ 0 when the slice is empty) and its largest
 *  score plus the log of its total, which is −∞ when the slice is empty. */
function storedSlice(scores: number[], values: number[][], width: number) {
	if (scores.length === 0)
		return { out: Array.from({ length: width }, () => Number.NaN), lse: Number.NEGATIVE_INFINITY };
	const { max, sum, out } = partialSoftmax(scores, values);
	return { out, lse: max + Math.log(sum) };
}

/** The combine kernel: skip empty slices, weigh each blend by e^(lse − the largest lse). */
function combine(parts: { out: number[]; lse: number }[]): number[] {
	const live = parts.filter((p) => p.lse > Number.NEGATIVE_INFINITY);
	const top = Math.max(...live.map((p) => p.lse));
	const w = live.map((p) => Math.exp(p.lse - top));
	const total = w.reduce((a, b) => a + b, 0);
	return live[0].out.map((_, d) => live.reduce((acc, p, i) => acc + w[i] * p.out[d], 0) / total);
}

/** Attention split into `splits` slices of equal length, rounded up, as kernel 3 cuts it: the last
 *  slices of a short conversation can be empty. */
export function splitAttention(scores: number[], values: number[][], splits: number): number[] {
	const size = cdiv(scores.length, splits);
	const parts = Array.from({ length: splits }, (_, i) =>
		storedSlice(
			scores.slice(i * size, (i + 1) * size),
			values.slice(i * size, (i + 1) * size),
			values[0].length,
		),
	);
	return combine(parts);
}

/** The worked example: four scores and one-number values, two slices of two. */
export const SPLIT_SCORES = [1, 3, 2, 0];
export const SPLIT_VALUES = [0.5, 1, -0.5, 2];

const r2 = (n: number) => Math.round(n * 100) / 100;

/** The worked example as the figure shows it: every number rounded to hundredths, and each later
 *  number computed from the rounded ones, so the displayed sums add up. */
export function splitWorked() {
	const top = Math.max(...SPLIT_SCORES);
	const whole = SPLIT_SCORES.map((s, j) => {
		const p = r2(Math.exp(s - top));
		return { s, v: SPLIT_VALUES[j], p, pv: r2(p * SPLIT_VALUES[j]) };
	});
	const wholeSum = r2(whole.reduce((a, t) => a + t.p, 0));
	const wholeAcc = r2(whole.reduce((a, t) => a + t.pv, 0));
	const slices = [0, 2].map((start) => {
		const scores = SPLIT_SCORES.slice(start, start + 2);
		const max = Math.max(...scores);
		const tokens = scores.map((s, i) => {
			const p = r2(Math.exp(s - max));
			const v = SPLIT_VALUES[start + i];
			return { s, v, p, pv: r2(p * v) };
		});
		const sum = r2(tokens.reduce((a, t) => a + t.p, 0));
		const acc = r2(tokens.reduce((a, t) => a + t.pv, 0));
		const out = r2(acc / sum);
		const shrink = r2(Math.exp(max - top));
		const weight = r2(sum * shrink);
		return { start, max, tokens, sum, acc, out, shrink, weight, weighted: r2(weight * out) };
	});
	const total = r2(slices.reduce((a, s) => a + s.weight, 0));
	const blend = r2(slices.reduce((a, s) => a + s.weighted, 0));
	return {
		top,
		whole,
		wholeSum,
		wholeAcc,
		wholeOut: r2(wholeAcc / wholeSum),
		slices,
		total,
		blend,
		splitOut: r2(blend / total),
	};
}

/** One layer for one token as a chain of kernels, each at its ideal time. A simplified version of
 *  SGLang's chain (the contest's muse_decode_layer baseline.py): the expert work is the contest's
 *  own ideal time for one token on the busiest GPU, divided between its two multiplications by
 *  their weights (two tables, then one). */
type ChainStep = { name: string; kind: "weights" | "cache" | "small"; us: number };

const smallUs = (bytes: number) => memoryUs(bytes);
const proj = (name: string, id: Parameters<typeof getProjection>[0]): ChainStep => ({
	name,
	kind: "weights",
	us: floorUs(projectionCost(getProjection(id), 1)),
});

export function layerChain(layer: "sliding" | "full"): ChainStep[] {
	const q = Q_HEADS * HEAD_DIM;
	const experts = shapeFloor("muse_moe_w4a16", "decode 1", "bs 1");
	return [
		proj("into attention", "qkv_gate"),
		{ name: "normalize query and key", kind: "small", us: smallUs(2 * (q + KV_SIZE) * 2) },
		{ name: "write the cache", kind: "small", us: smallUs(2 * KV_SIZE * 2 + 2 * KV_SIZE) },
		{ name: "attention", kind: "cache", us: memoryUs(kvBytes(CONTEXT_TOKENS, layer) + q * 2 * 2) },
		{ name: "normalize and gate", kind: "small", us: smallUs(3 * q * 2) },
		proj("out of attention", "o"),
		{ name: "mix into the total", kind: "small", us: smallUs(HIDDEN * (4 + 2 + 4 + 2)) },
		proj("choose experts", "router"),
		{ name: `pick the top ${TOP_K}`, kind: "small", us: smallUs(N_EXPERTS * 4 + TOP_K * 8) },
		proj("into the experts", "pooler"),
		{ name: "normalize", kind: "small", us: smallUs(D_EXPERT * 2 * 2) },
		{ name: "sort tokens by expert", kind: "small", us: smallUs(TOP_K * 8) },
		{ name: "experts, first tables", kind: "weights", us: (experts * 2) / 3 },
		{ name: "squash and multiply", kind: "small", us: smallUs(TOP_K * D_EXPERT * 2 * 3) },
		{ name: "experts, last table", kind: "weights", us: experts / 3 },
		{ name: "weigh the experts", kind: "small", us: smallUs(TOP_K * D_EXPERT * 2 + D_EXPERT * 2) },
		proj("out of the experts", "unpooler"),
		{ name: "mix into the total", kind: "small", us: smallUs(HIDDEN * (4 + 2 + 4 + 2)) },
	];
}

/** A fixed cost per kernel, assumed for the timeline: launching, waiting for the first data and
 *  finishing. The cost model has none; this is a round number for illustration. */
export const FIXED_US = 2;

/** One invented choice of a token's 8 experts, as they fall on the 8 GPUs: how many each holds. */
export const EXAMPLE_PLACEMENT = [2, 1, 0, 1, 3, 0, 1, 0];

/** Kernel 2 on a toy head of four numbers: the token's numbers from kernel 1 as one head's attention
 *  result, and four gate numbers. Each displayed value is computed from the rounded ones before it. */
const TOY_GATE = [0, 2, -2, 0];
export function toyNormGate() {
	const a = X;
	const meanSquare = a.reduce((s, v) => s + v * v, 0) / a.length;
	const rms = r2(Math.sqrt(meanSquare));
	const scaled = a.map((v) => r2(v / rms));
	const squashed = TOY_GATE.map((g) => r2(1 / (1 + Math.exp(-g))));
	const out = scaled.map((v, i) => r2(v * squashed[i]));
	return { a, gate: TOY_GATE, meanSquare, rms, scaled, squashed, out };
}

/** Tokens per tile in kernel 3's loop. */
export const ATTN_BLOCK = 64;

/** A second token for the toy, to show one read of a weight serving two tokens. */
export const TOY_X2 = [0.5, -1, 1, 0];
export const toyOutput2 = (row: number) => W[row].reduce((a, w, i) => a + w * TOY_X2[i], 0);

/** Kernel 1's address arithmetic for one program of the toy in tiles of `block` rows: the
 *  column of row numbers, the row of column numbers, the grid they broadcast to, the mask, the
 *  weights loaded (0 where masked), the products with the token's numbers, and each row's sum. */
export function toyTileGrid(program: number, block = 2) {
	const K = X.length;
	const rows = Array.from({ length: block }, (_, i) => program * block + i);
	const cols = Array.from({ length: K }, (_, j) => j);
	const live = rows.map((r) => r < W.length);
	const addresses = rows.map((r) => cols.map((c) => r * K + c));
	const w = rows.map((r, i) => cols.map((c) => (live[i] ? W[r][c] : 0)));
	const products = w.map((row) => row.map((v, j) => v * X[j]));
	const sums = products.map((row) => row.reduce((a, b) => a + b, 0));
	return { K, rows, cols, live, addresses, w, products, sums };
}

/** One program walking two tiles of the worked example's tokens, the second tile raising the
 *  largest score: the state after each tile, rounded as displayed. */
export function runningWorked() {
	const tiles = [
		[2, 3],
		[0, 1],
	].map((idx) => idx.map((j) => ({ s: SPLIT_SCORES[j], v: SPLIT_VALUES[j] })));
	let max = Number.NEGATIVE_INFINITY;
	let sum = 0;
	let acc = 0;
	return tiles.map((tokens) => {
		const before = { max, sum, acc };
		const next = Math.max(max, ...tokens.map((t) => t.s));
		const shrink = Number.isFinite(max) ? r2(Math.exp(max - next)) : 0;
		const kept = { sum: r2(sum * shrink), acc: r2(acc * shrink) };
		const own = tokens.map((t) => {
			const p = r2(Math.exp(t.s - next));
			return { ...t, p, pv: r2(p * t.v) };
		});
		max = next;
		sum = r2(kept.sum + own.reduce((a, t) => a + t.p, 0));
		acc = r2(kept.acc + own.reduce((a, t) => a + t.pv, 0));
		return { tokens: own, before, shrink, kept, max, sum, acc, blend: r2(acc / sum) };
	});
}
