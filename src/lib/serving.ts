// The server's side of the cost model: what one pass reads at each number of people, the speeds
// that follow, and a few comparisons the server chapters and the appendix draw from the judge's
// published floors. The counting rules follow speed_of_light_ns in the official muse_decode_layer
// reference.py: https://openkernelchallenge.dev/api/v1/problems/muse_decode_layer (files.reference.py).
import { loadFloors, type ProblemName, passBytesPerGpu, problem } from "./challenge";
import {
	CONTEXT_TOKENS,
	EXPERT_BYTES,
	EXPERT_WEIGHTS,
	H200,
	HEAD_DIM,
	HIDDEN,
	KV_SIZE,
	kvBytes,
	LAYERS,
	mathUs,
	memoryUs,
	NVFP4_BYTES,
	PROJECTIONS,
	projectionCost,
	Q_HEADS,
	TOP_K,
	TURN_TOKENS,
	weightBytes,
	weightBytesPerGpu,
} from "./model";

/** In the batched loads, each conversation's length is drawn up to this share either side of
 *  CONTEXT_TOKENS: AGENTIC_SPREAD in the official muse_decode_layer reference.py. */
export const LENGTH_SPREAD = 0.075;
/** The judge counts this share of the experts its routing touches, so its floor stays a lower bound:
 *  ROUTING_SHARE in the same reference.py. */
export const ROUTING_SHARE = 0.75;

type Layer = "sliding" | "full";
const layerOf = (shape: string): Layer => (/\blayer sliding\b/.test(shape) ? "sliding" : "full");

/** Bytes one layer moves for `people` people writing one token each, apart from the experts. */
function layerBytes(people: number, layer: Layer) {
	// The judge counts each cache at the shortest average length the spread allows.
	const share = people === 1 ? 1 : 1 - LENGTH_SPREAD;
	return {
		weights: PROJECTIONS.reduce((sum, p) => sum + weightBytes(p, people), 0),
		cache: people * kvBytes(CONTEXT_TOKENS, layer) * share,
		// The running total (FP32) and the attention input (BF16), read in and written out.
		working: people * HIDDEN * (4 + 2) * 2,
	};
}

const decodeShapes = (people: number) => {
	const shapes = problem("muse_decode_layer").shapes.filter((s) => s.load === `decode ${people}`);
	if (!shapes.length) throw new Error(`muse_decode_layer has no decode ${people} load`);
	return shapes;
};

/** One pass's floor at `people` people, in microseconds, split by what it moves. Every part but
 *  the experts is computed from model.ts. The judge's expert counts come from routing targets it
 *  doesn't publish, so the expert weights are what its floor leaves after the rest. Both the
 *  sliding and full decoder shapes include expert work, with 62 layers between them. */
export function passBill(people: number) {
	const shapes = decodeShapes(people);
	const totalUs = shapes.reduce((sum, s) => sum + s.weight * s.floorUs, 0);
	let weightsUs = 0;
	let cacheUs = 0;
	let workingUs = 0;
	let layers = 0;
	for (const s of shapes) {
		const b = layerBytes(people, layerOf(s.shape));
		weightsUs += s.weight * memoryUs(b.weights);
		cacheUs += s.weight * memoryUs(b.cache);
		workingUs += s.weight * memoryUs(b.working);
		layers += s.weight;
	}
	const expertsUs = totalUs - weightsUs - cacheUs - workingUs;
	return {
		people,
		totalUs,
		weightsUs,
		expertsUs,
		cacheUs,
		workingUs,
		/** The experts per layer the judge counts on the busiest GPU. */
		expertsPerLayer: (expertsUs * 1e-6 * H200.bytesPerSecond) / EXPERT_BYTES / layers,
	};
}

/** The most arithmetic a pass could need: as if every token's picks all landed on this GPU. If
 *  this is shorter than the bill, moving data sets the time and the bill is the whole floor. */
export function passArithmeticCeilingUs(people: number) {
	return decodeShapes(people).reduce((sum, s) => {
		const layer = layerOf(s.shape);
		const share = people === 1 ? 1 : 1 - LENGTH_SPREAD;
		const projections = PROJECTIONS.reduce((f, p) => f + projectionCost(p, people).flops, 0);
		const experts = 2 * people * TOP_K * EXPERT_WEIGHTS;
		const attended = (kvBytes(CONTEXT_TOKENS, layer) / (KV_SIZE * 2)) * share;
		const attention = 4 * people * attended * Q_HEADS * HEAD_DIM;
		return sum + s.weight * mathUs(projections + experts + attention);
	}, 0);
}

export type BillLine = { key: "weights" | "experts" | "cache" | "working"; us: number };

/** The bill as displayed, in whole microseconds that add up to the displayed total. The expert line
 *  takes the rounding, since it is the one the judge's floor sets. */
export function billLines(people: number) {
	const b = passBill(people);
	const total = Math.round(b.totalUs);
	const weights = Math.round(b.weightsUs);
	const cache = Math.round(b.cacheUs);
	const working = Math.round(b.workingUs);
	const lines: BillLine[] = [
		{ key: "weights", us: weights },
		{ key: "experts", us: total - weights - cache - working },
		{ key: "cache", us: cache },
		{ key: "working", us: working },
	];
	return { people, totalUs: total, lines, each: 1e6 / total, all: (people * 1e6) / total };
}

/** One pass's floor for any load, summed over its shapes. */
export function passUs(name: ProblemName, load: string) {
	const found = loadFloors(name).find((l) => l.load === load);
	if (!found) throw new Error(`${name} has no ${load} load`);
	return found.us;
}

/** Reading a prompt against writing a reply, in the server's time per token at the floor. */
export const promptTokenUs = passUs("muse_prefill_layer", "turn 1") / TURN_TOKENS;
export const replyTokenUs = (people: number) => passUs("muse_decode_layer", `decode ${people}`) / people;

/** Checking an 8-token draft against writing one token, for one person: how many passes it costs. */
export const draftCheckPasses =
	passUs("muse_decode_layer", "verify 1") / passUs("muse_decode_layer", "decode 1");

/** One conversation's cache in the full-attention layers on one GPU. */
export const fullLayerCacheBytes = (contextTokens: number) => LAYERS.full * kvBytes(contextTokens, "full");

/** The most conversations of this length whose full-layer caches fit beside the weights on one GPU:
 *  a ceiling, since the sliding layers' caches and the server's working space also need room. */
export const conversationsThatFit = (contextTokens: number) =>
	Math.floor((H200.memoryGB * 1e9 - weightBytesPerGpu) / fullLayerCacheBytes(contextTokens));

/** The ideal speed of a machine that reads `gbPerToken` gigabytes for every token. */
export const tokensPerSecond = (gbPerSecond: number, gbPerToken: number) => gbPerSecond / gbPerToken;

/** Memory bandwidth as each maker publishes it. */
const MACHINES = {
	// model.ts: NVIDIA's H200 datasheet.
	h200: H200.bytesPerSecond / 1e9,
	// https://www.nvidia.com/en-us/geforce/graphics-cards/compare (GeForce RTX 5090: 1792 GB/sec).
	rtx5090: 1792,
	// https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/ (M4 Max, 40-core GPU: 546GB/s).
	m4max: 546,
	// https://www.nvidia.com/en-us/products/workstations/dgx-spark/ (273 GB/s).
	spark: 273,
};

/** A whole model's weights read once per token, with the cache left out. 4-bit weights are in the
 *  guide's NVFP4 format, scales included. */
const SMALL_MODEL_WEIGHTS = 8e9;
const smallModelGb = (bytesPerWeight: number) => (SMALL_MODEL_WEIGHTS * bytesPerWeight) / 1e9;

/** Each example pairs a machine with what one token reads on it. Muse Spark Open's is the busiest
 *  H200's bytes at the judge's floor for one person, from the bill, so it pairs only with the H200. */
export const EXAMPLES = [
	{
		id: "muse",
		label: "Muse Spark Open, busiest H200 of the server",
		gbPerSecond: MACHINES.h200,
		gb: Math.round(passBytesPerGpu() / 1e8) / 10,
	},
	{
		id: "rtx5090",
		label: "8B model, 16-bit, on an RTX 5090",
		gbPerSecond: MACHINES.rtx5090,
		gb: smallModelGb(2),
	},
	{
		id: "m4max",
		label: "8B model, 4-bit, on an M4 Max laptop",
		gbPerSecond: MACHINES.m4max,
		gb: smallModelGb(NVFP4_BYTES),
	},
	{
		id: "spark",
		label: "8B model, 4-bit, on a DGX Spark",
		gbPerSecond: MACHINES.spark,
		gb: smallModelGb(NVFP4_BYTES),
	},
] as const;
