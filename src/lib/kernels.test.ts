import { expect, test } from "vitest";
import { shapeFloor } from "./challenge";
import {
	attentionGrid,
	attentionGroupGrid,
	EXAMPLE_PLACEMENT,
	GQA_SPLITS,
	KV_GROUP,
	layerChain,
	NORM_GATE_FUSED,
	NORM_GATE_UNFUSED,
	projectionGrid,
	runningWorked,
	SPLIT_SCORES,
	SPLIT_VALUES,
	splitAttention,
	splitWorked,
	TILE_CHOICES,
	tileTradeoff,
	toyNormGate,
	toyProgram,
	toyTiles,
	trips,
} from "./kernels";
import { CONTEXT_TOKENS, GPUS, H200, HEAD_DIM, KV_HEADS, kvBytes, Q_HEADS, TOP_K } from "./model";
import { W, X } from "./toy";

const softmaxBlend = (scores: number[], values: number[][]) => {
	const e = scores.map(Math.exp);
	const total = e.reduce((a, b) => a + b, 0);
	return values[0].map((_, d) => values.reduce((acc, v, j) => acc + (e[j] / total) * v[d], 0));
};

test("program 2 of the toy computes output 2 from row 2", () => {
	const p = toyProgram(2);
	expect(p.w).toEqual(W[2]);
	expect(p.y).toBeCloseTo(
		p.products.reduce((a, b) => a + b, 0),
		12,
	);
});

test("tiles of two rows cover the toy's three rows with one masked row", () => {
	const tiles = toyTiles(2);
	expect(tiles).toHaveLength(2);
	expect(
		tiles
			.flat()
			.filter((r) => r.live)
			.map((r) => r.row),
	).toEqual([0, 1, 2]);
	expect(tiles[1][1]).toEqual({ row: 3, live: false });
});

test("the pooler in tiles of 32 rows is 128 programs of 64 steps each", () => {
	const g = projectionGrid("pooler");
	expect(g.programs).toBe(128);
	expect(g.steps).toBe(64);
	expect(g.programs * g.bytesPerProgram).toBe(g.rows * g.cols * 2);
	expect(g.idle).toBe(H200.sms - 128);
});

test("the pooler offers four tile heights", () => {
	expect(TILE_CHOICES).toEqual([32, 16, 8, 1]);
});

test.each([
	{ blockN: 32, programs: 128, idleSms: 4, minProgramsPerSm: 0, maxProgramsPerSm: 1, warpsPerSm: 4 },
	{ blockN: 16, programs: 256, idleSms: 0, minProgramsPerSm: 1, maxProgramsPerSm: 2, warpsPerSm: 8 },
	{ blockN: 8, programs: 512, idleSms: 0, minProgramsPerSm: 3, maxProgramsPerSm: 4, warpsPerSm: 16 },
	{ blockN: 1, programs: 4096, idleSms: 0, minProgramsPerSm: 31, maxProgramsPerSm: 32, warpsPerSm: 64 },
])("pooler in rows of $blockN: distribution, warp limit and reads", (expected) => {
	const { blockN, programs } = expected;
	expect(tileTradeoff(blockN)).toEqual({
		...expected,
		tokenLoads: programs,
		bytesPerProgram: 16_384 * blockN,
		steps: 64,
	});
});

test("the tradeoff can use a different projection", () => {
	expect(tileTradeoff(16, "router")).toMatchObject({ programs: 16, bytesPerProgram: 524_288, steps: 64 });
});

test("fusing the gated norm cuts seven trips to memory to three", () => {
	expect(trips(NORM_GATE_UNFUSED)).toBe(7);
	expect(trips(NORM_GATE_FUSED)).toBe(3);
});

test("split attention, combined from stored log-sum-exps, matches unsplit softmax on a sample", () => {
	const n = 37;
	const scores = Array.from({ length: n }, (_, i) => Math.sin(i * 1.7) * 4);
	const values = Array.from({ length: n }, (_, i) => [Math.cos(i), i / n, -1 + (i % 5)]);
	const whole = softmaxBlend(scores, values);
	// 16 slices of 3 leave the last three empty; 64 slices of 1 leave 27 empty.
	for (const splits of [1, 2, 3, 8, 16, 37, 64]) {
		const split = splitAttention(scores, values, splits);
		for (const [d, x] of split.entries()) expect(x, `${splits} slices`).toBeCloseTo(whole[d], 12);
	}
});

test("the worked split example adds up as displayed and agrees with the unsplit blend", () => {
	const w = splitWorked();
	const sum = (xs: number[]) => Math.round(xs.reduce((a, b) => a + b, 0) * 100) / 100;
	expect(sum(w.whole.map((t) => t.p))).toBe(w.wholeSum);
	expect(sum(w.slices.map((s) => s.weight))).toBe(w.total);
	for (const s of w.slices) expect(sum(s.tokens.map((t) => t.p))).toBe(s.sum);
	expect(w.total).toBe(w.wholeSum);
	expect(w.splitOut).toBe(w.wholeOut);
	expect(w.wholeOut).toBeCloseTo(
		softmaxBlend(
			SPLIT_SCORES,
			SPLIT_VALUES.map((v) => [v]),
		)[0],
		2,
	);
});

test("eight slices per head give about one program per compute block", () => {
	const one = attentionGrid(1);
	const eight = attentionGrid(8);
	expect(one.programs).toBe(Q_HEADS);
	expect(eight.programs).toBeLessThanOrEqual(H200.sms);
	expect(eight.programs).toBeGreaterThan(H200.sms * 0.9);
	expect(eight.bytesPerProgram * 8).toBe(one.bytesPerHead);
	expect(one.bytesPerHead).toBe(80_000 * 2 * HEAD_DIM);
});

test("64 slices per group fill 128 compute blocks and read each KV head once", () => {
	expect(GQA_SPLITS).toBe(64);
	expect(KV_GROUP).toBe(8);
	const unsplit = attentionGroupGrid(1);
	expect(unsplit.programs).toBe(2);
	expect(unsplit.idle).toBe(130);
	expect(unsplit.bytesPerProgram).toBe(10_240_000);

	const split = attentionGroupGrid(GQA_SPLITS);
	expect(split).toEqual({
		groups: KV_HEADS,
		programs: 128,
		idle: 4,
		sliceTokens: 1250,
		bytesPerProgram: 160_000,
		bytesPerGroup: 10_240_000,
		totalBytes: 20_480_000,
	});
	expect(split.totalBytes).toBe(kvBytes(CONTEXT_TOKENS, "full"));
	expect(split.totalBytes).toBe(unsplit.totalBytes);
});

test("grouped slices round up in capacity and count only live cache entries in the total", () => {
	const context = 4097;
	const grid = attentionGroupGrid(128, context);
	expect(grid.idle).toBe(0);
	expect(grid.sliceTokens).toBe(33);
	expect(grid.bytesPerProgram).toBe(4224);
	expect(grid.totalBytes).toBe(kvBytes(context, "full"));
	expect(grid.totalBytes).toBeLessThan(grid.programs * grid.bytesPerProgram);
});

test("the layer chain's ideal times add up to the contest's floor for one token", () => {
	for (const layer of ["sliding", "full"] as const) {
		const total = layerChain(layer).reduce((a, k) => a + k.us, 0);
		const floor = shapeFloor("muse_decode_layer", "decode 1", layer);
		expect(Math.abs(total - floor) / floor, layer).toBeLessThan(0.02);
	}
});

test("the example placement spreads one token's experts over the GPUs", () => {
	expect(EXAMPLE_PLACEMENT).toHaveLength(GPUS);
	expect(EXAMPLE_PLACEMENT.reduce((a, b) => a + b, 0)).toBe(TOP_K);
});

test("the toy gated norm matches the formula and adds up as displayed", () => {
	const t = toyNormGate();
	const exact = X.map(
		(v, i) => v / Math.sqrt(X.reduce((s, x) => s + x * x, 0) / X.length) / (1 + Math.exp(-t.gate[i])),
	);
	for (const [i, v] of t.out.entries()) expect(v).toBeCloseTo(exact[i], 1);
	expect(t.rms * t.rms).toBeCloseTo(t.meanSquare, 6);
});

test("walking two tiles, the second raising the largest score, gives the worked blend", () => {
	const [first, second] = runningWorked();
	expect(first.shrink).toBe(0);
	expect(second.max).toBeGreaterThan(first.max);
	expect(second.blend).toBe(splitWorked().wholeOut);
	expect(second.sum).toBe(splitWorked().wholeSum);
});
