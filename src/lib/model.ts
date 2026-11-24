// Muse Spark Open's per-GPU shapes and the H200 limits, as the challenge defines them.
// The cost model follows `speed_of_light_ns` in the official muse_bf16_proj reference.py:
// the time to move the bytes or do the math at full speed, whichever is longer.
// Source: https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj (files.reference.py).

// On-chip sizes are GH100's, from NVIDIA's Hopper Tuning Guide and Hopper Architecture In-Depth.
// H200 SXM5 enables all six HBM3e stacks (6144-bit bus, 141 GB usable, 4.8 TB/s). H100 SXM5
// enabled five of the six stacks, as described in NVIDIA Hopper Architecture In-Depth.
export const H200 = {
	bytesPerSecond: 4.8e12,
	// Dense BF16 Tensor Core peak: NVIDIA's 1,979 TFLOPS with sparsity, halved.
	flopsPerSecond: 989e12,
	// Hopper Architecture In-Depth describes the FP32 CUDA cores; NVIDIA's H100 specs give 67 TFLOPS for SXM.
	// https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
	// https://www.nvidia.com/en-us/data-center/h100/
	fp32FlopsPerSecond: 67e12,
	sms: 132,
	memoryGB: 141,
	hbmStacks: 6,
	l2Bytes: 50 * 2 ** 20,
	sharedBytesPerSm: 228 * 2 ** 10,
	registerBytesPerSm: 256 * 2 ** 10,
	tensorCoresPerSm: 4,
	warpSize: 32,
	maxThreadsPerSm: 2048,
} as const;

export const HIDDEN = 8192;
export const HEAD_DIM = 64;
export const Q_HEADS = 16;
export const KV_HEADS = 2;
const Q_SIZE = Q_HEADS * HEAD_DIM;
export const KV_SIZE = KV_HEADS * HEAD_DIM;
const QKV_SIZE = 2 * Q_SIZE + 2 * KV_SIZE;
export const D_EXPERT = 4096;
export const N_EXPERTS = 256;
export const LOCAL_EXPERTS = 32;
export const TOP_K = 8;
/** The router multiplies its outputs by this before the sigmoid: 1/√32, from the contest's template. */
export const ROUTER_SCALE = 1 / Math.sqrt(32);
export const SLIDING_WINDOW = 2048;
export const NVFP4_BLOCK = 16;
export const GPUS = 8;
// Up to 8 tokens, each GPU runs only its 1/8 slice of the unpooler.
const UNPOOL_SLICE = HIDDEN / GPUS;
export const SLICED_MAX_TOKENS = 8;

/** Meta's published sizes: all weights, and the weights one token uses. */
export const TOTAL_WEIGHTS = 816e9;
export const ACTIVE_WEIGHTS = 42e9;

// The serving setup the contest times.
export const CONTEXT_TOKENS = 80_000;
export const TURN_TOKENS = 753;
export const DRAFT_TOKENS = 8;
/** SGLang reads long prompts in chunks of this many tokens. */
export const PREFILL_CHUNK = 16_384;

/** Bytes per weight in NVFP4: a 4-bit value plus a shared 8-bit scale per block of 16. */
export const NVFP4_BYTES = 0.5 + 1 / NVFP4_BLOCK;
/** Weights in one expert: gate, up and down, each 4096 x 4096. */
export const EXPERT_WEIGHTS = 3 * D_EXPERT * D_EXPERT;
/** One expert's weights in NVFP4, scales included, read whole whenever a token picks it. */
export const EXPERT_BYTES = EXPERT_WEIGHTS * NVFP4_BYTES;

export type Projection = {
	id: "qkv_gate" | "o" | "pooler" | "unpooler" | "router";
	label: string;
	rows: number;
	cols: number;
	bytesPerWeight: number;
};

export const PROJECTIONS: Projection[] = [
	{ id: "qkv_gate", label: "qkv_gate", rows: QKV_SIZE, cols: HIDDEN, bytesPerWeight: 2 },
	{ id: "o", label: "o", rows: HIDDEN, cols: Q_SIZE, bytesPerWeight: 2 },
	{ id: "router", label: "router", rows: N_EXPERTS, cols: HIDDEN, bytesPerWeight: 4 },
	{ id: "pooler", label: "pooler", rows: D_EXPERT, cols: HIDDEN, bytesPerWeight: 2 },
	{ id: "unpooler", label: "unpooler", rows: HIDDEN, cols: D_EXPERT, bytesPerWeight: 2 },
];

export function getProjection(id: string): Projection {
	const projection = PROJECTIONS.find((p) => p.id === id);
	if (!projection) throw new Error(`model.ts has no projection ${id}`);
	return projection;
}

type Cost = { bytes: number; flops: number };

export const memoryUs = (bytes: number) => (bytes / H200.bytesPerSecond) * 1e6;
export const mathUs = (flops: number) => (flops / H200.flopsPerSecond) * 1e6;
export const fp32MathUs = (flops: number) => (flops / H200.fp32FlopsPerSecond) * 1e6;
export const floorUs = (c: Cost) => Math.max(memoryUs(c.bytes), mathUs(c.flops));

/** One projection for `tokens` tokens: read the weights and inputs, write the outputs. */
export function projectionCost(p: Projection, tokens: number): Cost {
	const rows = projectionRows(p, tokens);
	return {
		// Inputs are BF16; outputs are BF16 except the router's, which are FP32.
		bytes: weightBytes(p, tokens) + tokens * (p.cols * 2 + rows * (p.id === "router" ? 4 : 2)),
		flops: 2 * tokens * rows * p.cols,
	};
}

const projectionRows = (p: Projection, tokens: number) =>
	p.id === "unpooler" && tokens <= SLICED_MAX_TOKENS ? UNPOOL_SLICE : p.rows;

/** The weight bytes a projection reads for `tokens` tokens. */
export const weightBytes = (p: Projection, tokens: number) =>
	projectionRows(p, tokens) * p.cols * p.bytesPerWeight;

/** Key and value bytes one request reads in one layer, from the FP8 cache. */
export function kvBytes(contextTokens: number, layer: "sliding" | "full"): number {
	const attended = layer === "sliding" ? Math.min(contextTokens, SLIDING_WINDOW) : contextTokens;
	return attended * KV_SIZE * 2;
}

/** The arithmetic the GPU's tensor cores can do in the time it takes to read one byte. */
export const BALANCE_POINT = H200.flopsPerSecond / H200.bytesPerSecond;
/** The FP32 arithmetic the GPU's CUDA cores can do in the time it takes to read one byte. */
export const FP32_BALANCE_POINT = H200.fp32FlopsPerSecond / H200.bytesPerSecond;

/** Layers in one pass. The challenge weights each shape by how often a step runs it:
 *  46 sliding-window layers and 16 full-attention layers. */
export const LAYERS = { sliding: 46, full: 16, total: 62 } as const;

/** The standalone muse_moe_w4a16 benchmark's shape weight. Both muse_decode_layer shapes include experts. */
export const MOE_PROBLEM_LAYERS = 60;

/** Query-key and attention-value FLOPs for one new token across all GPUs and layers. */
export function attentionFlops(contextTokens: number): number {
	const attended = LAYERS.full * contextTokens + LAYERS.sliding * Math.min(contextTokens, SLIDING_WINDOW);
	return 4 * attended * GPUS * Q_HEADS * HEAD_DIM;
}

/** The layers' weights one GPU holds: its LOCAL_EXPERTS experts and every projection in full, in all
 *  LAYERS.total layers. The token lookup table and the final scoring table come on top. */
export const weightBytesPerGpu =
	LAYERS.total *
	(PROJECTIONS.reduce((sum, p) => sum + p.rows * p.cols * p.bytesPerWeight, 0) +
		LOCAL_EXPERTS * EXPERT_BYTES);

/** Clock cycles from a request to its data, measured on an H800 (a Hopper chip like the H200's):
 *  Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", Table IV
 *  (arxiv.org/abs/2402.13499). NVIDIA publishes no on-chip latencies or bandwidths. */
export const MEASURED_LATENCY_CYCLES = { shared: 29, l2: 263, hbm: 479 } as const;
/** H100 SXM5 arithmetic units per compute block: "128 FP32 CUDA Cores per SM" (NVIDIA Hopper
 *  Architecture In-Depth, developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth). */
export const FP32_UNITS_PER_SM = 128;
/** Memory moves in 32-byte pieces: a warp's reads "coalesce into a number of transactions equal to
 *  the number of 32-byte transactions necessary" (CUDA C++ Best Practices Guide, compute capability
 *  6.0 and later, docs.nvidia.com/cuda/cuda-c-best-practices-guide). */
export const MEMORY_SECTOR_BYTES = 32;

/** Short-context, weight-only arithmetic ceiling: two FLOPs per active weight at peak Tensor Core rate. */
export const ARITHMETIC_TOKENS_PER_SECOND = (GPUS * H200.flopsPerSecond) / (2 * ACTIVE_WEIGHTS);

/** Arithmetic ceiling including attention's two multiplications at the contest context length. */
export const LONG_CONTEXT_ARITHMETIC_TOKENS_PER_SECOND =
	(GPUS * H200.flopsPerSecond) / (2 * ACTIVE_WEIGHTS + attentionFlops(CONTEXT_TOKENS));
