// One definition per term, used by <Term> in the margin and by the glossary page.

import { count } from "../lib/format";
import { PROJ_BLOCK_N } from "../lib/kernels";
import {
	GPUS,
	H200,
	HIDDEN,
	LAYERS,
	LOCAL_EXPERTS,
	N_EXPERTS,
	NVFP4_BLOCK,
	SLIDING_WINDOW,
	TOP_K,
} from "../lib/model";
import { href, STOPS } from "./stops";

export type Category = "model" | "gpu" | "server" | "challenge";

export const CATEGORIES: Record<Category, string> = {
	model: "The model",
	gpu: "The GPU",
	server: "The server",
	challenge: "The challenge",
};

export type Entry = {
	term: string;
	short: string;
	category: Category;
	/** The chapter that teaches the term. */
	chapter?: { label: string; href: string };
	/** One canonical outside source for readers who want more. */
	link?: { label: string; href: string };
};

// Chapter references come from STOPS so the labels and paths cannot drift from the chapter list.
const ch = (slug: string) => {
	const stop = STOPS.find((s) => s.slug === slug);
	if (!stop) throw new Error(`glossary: no chapter "${slug}"`);
	return { label: stop.part === "appendix" ? "Appendix" : `Ch ${stop.label}`, href: href(stop) };
};
const OPENING = ch("");
const PASS = ch("pass");
const ATTENTION = ch("attention");
const EXPERTS = ch("experts");
const CHIP = ch("chip");
const KERNEL = ch("kernel");
const LAYER = ch("layer");
const SLICES = ch("attention-kernel");
const BATCHING = ch("batching");
const PREFILL = ch("prefill");
const TRADEOFF = ch("tradeoff");
const APPENDIX = ch("challenge");

export const GLOSSARY = {
	token: {
		term: "token",
		category: "model",
		chapter: OPENING,
		short:
			"The unit of text a model reads and writes: a common word, a piece of a longer one, a space or a punctuation mark.",
	},
	prefill: {
		term: "prefill",
		category: "model",
		chapter: ATTENTION,
		short:
			"The phase where the model reads its new input, all of it together, before writing. A long input is read in a few large chunks.",
	},
	decode: {
		term: "decode",
		category: "model",
		chapter: OPENING,
		short:
			"The phase where the model writes its reply, normally one token per pass (or several at once when verifying a speculative draft).",
	},
	weights: {
		term: "weights",
		category: "model",
		chapter: OPENING,
		short:
			"The numbers a model learned in training. Running the model means multiplying by them. Also called parameters.",
	},
	activations: {
		term: "activations",
		category: "model",
		chapter: PASS,
		short:
			"A token's working numbers: the list each step of a layer reads and changes. Every pass makes them fresh.",
	},
	layer: {
		term: "layer",
		category: "model",
		chapter: PASS,
		short: `One repetition of the model's recipe. Muse Spark Open runs ${LAYERS.total} of them for every token it writes.`,
	},
	gpu: {
		term: "GPU",
		category: "gpu",
		chapter: OPENING,
		short:
			"A chip built to do huge amounts of simple arithmetic at once, with its own memory where the model's weights live.",
	},
	hbm: {
		term: "HBM",
		category: "gpu",
		chapter: CHIP,
		short: `High-bandwidth memory: the stacks of memory next to the GPU chip where the weights live. ${H200.memoryGB} GB on an H200.`,
	},
	sm: {
		term: "compute block",
		category: "gpu",
		chapter: CHIP,
		short: `One of the H200's ${H200.sms} compute blocks, each with its own arithmetic units and a little fast memory. NVIDIA calls them streaming multiprocessors, or SMs.`,
	},
	kernel: {
		term: "kernel",
		category: "gpu",
		chapter: KERNEL,
		link: {
			label: "Triton: vector addition tutorial",
			href: "https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html",
		},
		short: "A function the GPU runs as many copies at once, for one step of the work.",
	},
	sglang: {
		term: "SGLang",
		category: "server",
		chapter: OPENING,
		link: { label: "SGLang and RadixAttention", href: "https://lmsys.org/blog/2024-01-17-sglang/" },
		short:
			"Open-source software that serves language models: it batches requests, manages each conversation's cache and launches the kernels for every pass.",
	},
	projection: {
		term: "projection",
		category: "model",
		chapter: PASS,
		short: "Multiplying a token's numbers by a table of weights to get a new list of numbers.",
	},
	bf16: {
		term: "BF16",
		category: "model",
		chapter: EXPERTS,
		short: "A 16-bit number format: 2 bytes per number, about two to three significant decimal digits.",
	},
	fp32: {
		term: "FP32",
		category: "gpu",
		chapter: KERNEL,
		short: "The standard 32-bit number format: 4 bytes per number, about seven significant digits.",
	},
	fp8: {
		term: "FP8",
		category: "gpu",
		chapter: SLICES,
		short:
			"An 8-bit number format: 1 byte per number, a quarter the size of a standard number and much less precise.",
	},
	nvfp4: {
		term: "NVFP4",
		category: "model",
		chapter: EXPERTS,
		link: {
			label: "NVIDIA: introducing NVFP4",
			href: "https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/",
		},
		short: `NVIDIA's 4-bit format for weights. Each weight is a 4-bit code, and every block of ${NVFP4_BLOCK} shares one 8-bit scale.`,
	},
	bandwidth: {
		term: "memory bandwidth",
		category: "gpu",
		chapter: CHIP,
		short: `How fast data moves from memory to the chip. The H200 is rated at ${H200.bytesPerSecond / 1e12} TB per second.`,
	},
	memoryBound: {
		term: "memory-bound",
		category: "gpu",
		chapter: CHIP,
		short: "Limited by how fast data arrives, not by how fast the chip can do arithmetic.",
	},
	computeBound: {
		term: "compute-bound",
		category: "gpu",
		chapter: CHIP,
		short: "Limited by arithmetic: the data arrives faster than the chip can use it.",
	},
	floor: {
		term: "floor",
		category: "challenge",
		chapter: APPENDIX,
		short: `The contest's ideal minimum time for a set of inputs: the time to move their bytes at ${H200.bytesPerSecond / 1e12} trillion bytes a second, or to do their arithmetic at ${count(H200.flopsPerSecond / 1e12)} trillion operations a second, whichever is longer.`,
	},
	throughput: {
		term: "throughput",
		category: "server",
		chapter: TRADEOFF,
		short: "How many tokens a server produces per second across everyone it is serving.",
	},
	batch: {
		term: "batch",
		category: "server",
		chapter: BATCHING,
		short:
			"Several people's tokens processed together in one pass, sharing one read of the weights. Each conversation keeps its own cache.",
	},
	speculative: {
		term: "speculative decoding",
		category: "server",
		chapter: PREFILL,
		link: {
			label: "Leviathan et al.: Fast Inference via Speculative Decoding",
			href: "https://arxiv.org/abs/2211.17192",
		},
		short:
			"A small, fast model guesses several tokens, and the big model checks them all in one pass, keeping them up to the first guess it disagrees with.",
	},
	kvCache: {
		term: "KV cache",
		category: "model",
		chapter: ATTENTION,
		link: {
			label: "Hugging Face: KV cache explained",
			href: "https://huggingface.co/docs/transformers/main/en/cache_explanation",
		},
		short:
			"What a conversation keeps from its earlier tokens so they need not be worked out again: one entry per token in every layer. Each conversation has its own.",
	},
	context: {
		term: "context",
		category: "model",
		chapter: ATTENTION,
		short: "Everything the model can see: the conversation so far, including its own replies.",
	},
	attention: {
		term: "attention",
		category: "model",
		chapter: ATTENTION,
		link: {
			label: "3Blue1Brown: attention in transformers",
			href: "https://www.3blue1brown.com/lessons/attention",
		},
		short: "The step where each new token looks at itself and earlier tokens and gathers what is relevant.",
	},
	softmax: {
		term: "softmax",
		category: "model",
		chapter: ATTENTION,
		short:
			"The step that turns a list of scores into shares that add up to one, so the shares can weight a blend.",
	},
	slidingWindow: {
		term: "sliding window",
		category: "model",
		chapter: ATTENTION,
		short: `Attention that looks back over only the latest ${count(SLIDING_WINDOW)} tokens, the new one included.`,
	},
	heads: {
		term: "attention heads",
		category: "model",
		chapter: ATTENTION,
		short:
			"Independent copies of attention running side by side, each looking for different patterns. Grouped-query attention lets several query heads share one set of keys and values.",
	},
	rope: {
		term: "RoPE",
		category: "model",
		chapter: ATTENTION,
		link: { label: "EleutherAI: rotary embeddings", href: "https://blog.eleuther.ai/rotary-embeddings/" },
		short: "Rotary position embedding: rotating numbers by an angle that encodes each token's position.",
	},
	page: {
		term: "page table",
		category: "gpu",
		chapter: SLICES,
		short:
			"A lookup list saying where each token's cache entry is stored, so the cache need not sit in one contiguous block.",
	},
	expert: {
		term: "expert",
		category: "model",
		chapter: EXPERTS,
		short: `One of ${N_EXPERTS} small networks in each layer. Each token uses only ${TOP_K} of them.`,
	},
	router: {
		term: "router",
		category: "model",
		chapter: EXPERTS,
		short: `A small projection that scores all ${N_EXPERTS} experts for a token and picks the top ${TOP_K}.`,
	},
	moe: {
		term: "mixture of experts",
		category: "model",
		chapter: EXPERTS,
		link: { label: "Hugging Face: mixture of experts explained", href: "https://huggingface.co/blog/moe" },
		short: "A design where each token passes through a few chosen experts instead of one huge block.",
	},
	expertParallel: {
		term: "expert parallelism",
		category: "model",
		chapter: EXPERTS,
		short: `Dividing a layer's experts among GPUs. Here each of the ${GPUS} GPUs holds ${LOCAL_EXPERTS} of the ${N_EXPERTS}.`,
	},
	allReduce: {
		term: "all-reduce",
		category: "gpu",
		chapter: LAYER,
		link: {
			label: "NVIDIA NCCL: collective operations",
			href: "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html",
		},
		short: "GPUs exchanging partial results so each ends up with the full sum.",
	},
	residual: {
		term: "residual stream",
		category: "model",
		chapter: PASS,
		short: `The running list of ${count(HIDDEN)} numbers per token that every layer reads a copy of and mixes its results into.`,
	},
	norm: {
		term: "RMSNorm",
		category: "model",
		chapter: PASS,
		short: "Rescaling a list of numbers to a standard size, so values stay in a workable range.",
	},
	cudaGraph: {
		term: "CUDA graph",
		category: "gpu",
		chapter: LAYER,
		link: {
			label: "NVIDIA: Getting started with CUDA Graphs",
			href: "https://developer.nvidia.com/blog/cuda-graphs/",
		},
		short: "A recording of a sequence of GPU work that can be replayed without the CPU re-issuing each step.",
	},
	l2: {
		term: "L2 cache",
		category: "gpu",
		chapter: CHIP,
		short: "A small, fast memory on the chip (tens of MB) that keeps recently used data.",
	},
	speedup: {
		term: "speedup",
		category: "challenge",
		chapter: APPENDIX,
		short: "SGLang's time divided by yours. 1.00× ties SGLang; 2.00× takes half the time.",
	},
	geomean: {
		term: "geometric mean",
		category: "challenge",
		chapter: APPENDIX,
		short: "Multiplying n numbers and taking the n-th root. A 10% gain counts the same wherever it happens.",
	},
	load: {
		term: "load",
		category: "challenge",
		chapter: BATCHING,
		short: "One serving situation the challenge times, such as one user writing or 32 users checking drafts.",
	},
	pass: {
		term: "pass",
		category: "model",
		chapter: OPENING,
		short: `One run of the whole model, all ${LAYERS.total} layers. During normal decode, a pass produces one new token for each conversation in the batch.`,
	},
	ttft: {
		term: "time to first token",
		category: "server",
		chapter: PREFILL,
		short:
			"The pause between sending a message and seeing the first word of the reply: mostly the time to read the new input (prefill), plus any wait in the server's queue.",
	},
	continuousBatching: {
		term: "continuous batching",
		category: "server",
		chapter: BATCHING,
		link: {
			label: "Anyscale: continuous batching",
			href: "https://www.anyscale.com/blog/continuous-batching-llm-inference",
		},
		short:
			"Letting people join and leave a batch between passes, instead of waiting for every reply in it to finish.",
	},
	promptCaching: {
		term: "prompt caching",
		category: "server",
		chapter: PREFILL,
		short:
			"Keeping a conversation's cache after a reply ends, so the next turn reads only its new tokens instead of the whole conversation again.",
	},
	thread: {
		term: "thread",
		category: "gpu",
		chapter: CHIP,
		short: `One list of GPU instructions running with its own few numbers. Not a core: a compute block holds up to ${count(H200.maxThreadsPerSm)} of them and runs them ${H200.warpSize} at a time.`,
	},
	warp: {
		term: "warp",
		category: "gpu",
		chapter: CHIP,
		short: `${H200.warpSize} threads that run the same instruction together. A compute block switches between warps while some wait for data.`,
	},
	register: {
		term: "registers",
		category: "gpu",
		chapter: CHIP,
		short:
			"The fastest memory on a GPU: the numbers one thread is working on right now, seen by that thread alone.",
	},
	sharedMemory: {
		term: "shared memory",
		category: "gpu",
		chapter: CHIP,
		short: "A small memory inside each compute block that the threads of one program share with each other.",
	},
	latencyHiding: {
		term: "latency hiding",
		category: "gpu",
		chapter: CHIP,
		short:
			"Keeping many warps on a compute block so that while some wait for data, others have arithmetic to do.",
	},
	program: {
		term: "program",
		category: "gpu",
		chapter: KERNEL,
		short:
			"One running copy of a Triton kernel, working on one tile. A group of threads on one compute block runs it; programs run in no fixed order.",
	},
	launch: {
		term: "launch",
		category: "gpu",
		chapter: KERNEL,
		short:
			"The CPU's request to run a kernel: which function, how many programs, and the arguments to hand each one.",
	},
	tile: {
		term: "tile",
		category: "gpu",
		chapter: KERNEL,
		short: `The block of the work one program owns, such as ${PROJ_BLOCK_N} rows of a weight table.`,
	},
	mask: {
		term: "mask",
		category: "gpu",
		chapter: KERNEL,
		short:
			"A list of true and false that switches off the parts of a load or store that fall past the edge of the data.",
	},
	coalescing: {
		term: "coalescing",
		category: "gpu",
		chapter: KERNEL,
		short:
			"Neighboring threads reading neighboring addresses, so their reads combine into a few wide trips to memory.",
	},
	tensorCore: {
		term: "tensor core",
		category: "gpu",
		chapter: CHIP,
		link: {
			label: "NVIDIA Hopper architecture in depth",
			href: "https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/",
		},
		short: `A unit in each compute block that multiplies small matrices in one step. The H200 has ${H200.tensorCoresPerSm} in each.`,
	},
	fusion: {
		term: "fusion",
		category: "gpu",
		chapter: LAYER,
		link: {
			label: "Triton: fused softmax tutorial",
			href: "https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html",
		},
		short:
			"Doing several steps in one kernel, so the numbers between them stay on the chip instead of going to memory and back.",
	},
	splitContext: {
		term: "split attention",
		category: "gpu",
		chapter: SLICES,
		link: { label: "PyTorch: Flash-Decoding", href: "https://pytorch.org/blog/flash-decoding/" },
		short:
			"Cutting a conversation's cache into slices that separate programs attend over at once, then combining their partial results. Also called flash-decoding.",
	},
	tensorParallel: {
		term: "tensor parallelism",
		category: "gpu",
		chapter: EXPERTS,
		short: `Dividing most of a layer's weight tables among the ${GPUS} GPUs, so each computes part of a result and they exchange the parts to complete it.`,
	},
	roofline: {
		term: "roofline model",
		category: "gpu",
		chapter: CHIP,
		short: `A chart of the fastest a job can go: either its bytes arrive in time or its arithmetic finishes in time, so the slower of the two sets the floor. The H200's turning point is about ${count(H200.flopsPerSecond / H200.bytesPerSecond)} operations per byte.`,
		link: { label: "Horace He: Making Deep Learning Go Brrrr", href: "https://horace.io/brrr_intro.html" },
	},
	arithmeticIntensity: {
		term: "arithmetic intensity",
		category: "gpu",
		chapter: CHIP,
		short:
			"Operations done per byte moved from memory. Low intensity means the chip waits for data; high intensity means it waits for arithmetic.",
		link: { label: "Horace He: Making Deep Learning Go Brrrr", href: "https://horace.io/brrr_intro.html" },
	},
	gqa: {
		term: "grouped-query attention",
		category: "model",
		chapter: ATTENTION,
		short:
			"Attention where several query heads share one set of keys and values, so the cache keeps far fewer entries per token. Often shortened to GQA.",
		link: { label: "Ainslie et al.: GQA", href: "https://arxiv.org/abs/2305.13245" },
	},
	gemv: {
		term: "GEMV",
		category: "gpu",
		chapter: KERNEL,
		short:
			"Matrix-vector multiply: one token's numbers times a table of weights. Writing a reply for one person is mostly this, and it reads every weight to do one multiply-add with it.",
		link: { label: "Horace He: Making Deep Learning Go Brrrr", href: "https://horace.io/brrr_intro.html" },
	},
	gemm: {
		term: "GEMM",
		category: "server",
		chapter: BATCHING,
		short:
			"Matrix-matrix multiply: many tokens' numbers times a table of weights, so each weight read is used many times. Batches and prefill do this, on tensor cores.",
		link: {
			label: "Triton Matrix Multiplication Tutorial",
			href: "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html",
		},
	},
	flashAttention: {
		term: "FlashAttention",
		category: "gpu",
		chapter: SLICES,
		short:
			"An attention kernel that works through the cache in tiles and keeps its running results on the chip, so the table of scores is never written to memory.",
		link: { label: "FlashAttention-2 paper", href: "https://arxiv.org/abs/2307.08691" },
	},
	onlineSoftmax: {
		term: "online softmax",
		category: "gpu",
		chapter: SLICES,
		short:
			"Computing softmax one tile or slice at a time, tracking a running maximum and running total and rescaling earlier values whenever a larger score arrives.",
		link: { label: "Milakov & Gimelshein (2018)", href: "https://arxiv.org/abs/1805.02867" },
	},
	pagedAttention: {
		term: "PagedAttention",
		category: "gpu",
		chapter: SLICES,
		short:
			"Storing each conversation's cache in small fixed-size pages found through a page table, so memory is not wasted on space a conversation may never use.",
		link: { label: "vLLM: PagedAttention", href: "https://blog.vllm.ai/2023/06/20/vllm.html" },
	},
	radixAttention: {
		term: "RadixAttention",
		category: "server",
		chapter: PREFILL,
		short:
			"SGLang's way of keeping caches in a tree of shared prefixes, so conversations that begin with the same text reuse one stored copy of its cache.",
		link: { label: "SGLang and RadixAttention", href: "https://lmsys.org/blog/2024-01-17-sglang/" },
	},
	cudaCore: {
		term: "CUDA core",
		category: "gpu",
		chapter: CHIP,
		short: `A unit that does one ordinary 32-bit multiply-add at a time. Together they reach about ${count(H200.fp32FlopsPerSecond / 1e12)} trillion operations a second on an H200, against ${count(H200.flopsPerSecond / 1e12)} trillion for the tensor cores.`,
		link: { label: "Modal GPU Glossary", href: "https://modal.com/gpu-glossary" },
	},
	splitK: {
		term: "split-K",
		category: "gpu",
		chapter: KERNEL,
		short:
			"Dividing the long dimension a sum runs over among several programs, then adding their partial sums, so a small job still fills the compute blocks.",
		link: { label: "PyTorch: Flash-Decoding", href: "https://pytorch.org/blog/flash-decoding/" },
	},
	quantization: {
		term: "quantization",
		category: "model",
		chapter: EXPERTS,
		short:
			"Storing weights or cache in fewer bits, such as FP8 or NVFP4, so fewer bytes move for each number at some cost in precision.",
		link: {
			label: "NVIDIA: introducing NVFP4",
			href: "https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/",
		},
	},
	chunkedPrefill: {
		term: "chunked prefill",
		category: "server",
		chapter: PREFILL,
		short:
			"Reading a long input in slices, a few at a time between other people's decode passes, so one big prompt does not stall everyone else's replies.",
		link: { label: "Sarathi-Serve paper", href: "https://arxiv.org/abs/2403.02310" },
	},
	triton: {
		term: "Triton",
		category: "gpu",
		chapter: KERNEL,
		short:
			"A language for writing GPU kernels in Python, where you write what one program does to one tile and the compiler handles the threads.",
		link: {
			label: "Triton Tutorials",
			href: "https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html",
		},
	},
} satisfies Record<string, Entry>;

export type TermId = keyof typeof GLOSSARY;
