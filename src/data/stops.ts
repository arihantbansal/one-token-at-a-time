import type { ProblemName } from "../lib/challenge";
import { count } from "../lib/format";
import { ACTIVE_WEIGHTS, TOTAL_WEIGHTS } from "../lib/model";

export type Part = "opening" | "model" | "gpu" | "server" | "appendix";

export const PARTS: Record<Part, string> = {
	opening: "Opening",
	model: "The model",
	gpu: "The GPU",
	server: "The server",
	appendix: "Appendix",
};

export type Stop = {
	/** Reading order, from 0. */
	n: number;
	/** What the reader sees: "1" to "10", or "A". */
	label: string;
	slug: string;
	part: Part;
	title: string;
	/** The reader's question the chapter answers. */
	question: string;
	/** The contest problem whose work this chapter explains, for links from the layer map and the appendix. */
	problem?: ProblemName;
};

export const STOPS: Stop[] = [
	{
		n: 0,
		label: "0",
		slug: "",
		part: "opening",
		title: "One token at a time",
		question: "Where does the time go?",
	},
	{
		n: 1,
		label: "1",
		slug: "pass",
		part: "model",
		title: "One pass",
		question: "What does one pass compute?",
	},
	{
		n: 2,
		label: "2",
		slug: "attention",
		part: "model",
		title: "Remembering",
		question: "How does the model remember the conversation?",
	},
	{
		n: 3,
		label: "3",
		slug: "experts",
		part: "model",
		title: "Experts",
		question: `Why read only ${count(ACTIVE_WEIGHTS / 1e9)} of ${count(TOTAL_WEIGHTS / 1e9)} billion weights?`,
		problem: "muse_moe_w4a16",
	},
	{
		n: 4,
		label: "4",
		slug: "chip",
		part: "gpu",
		title: "Inside the chip",
		question: "What is inside a GPU?",
	},
	{
		n: 5,
		label: "5",
		slug: "kernel",
		part: "gpu",
		title: "A kernel",
		question: "What does a kernel do?",
		problem: "muse_bf16_proj",
	},
	{
		n: 6,
		label: "6",
		slug: "layer",
		part: "gpu",
		title: "A chain of kernels",
		question: "What is a pass to the GPU?",
		problem: "muse_decode_layer",
	},
	{
		n: 7,
		label: "7",
		slug: "attention-kernel",
		part: "gpu",
		title: "Reading the cache",
		question: "How does the chip read a long conversation?",
		problem: "muse_attention_decode",
	},
	{
		n: 8,
		label: "8",
		slug: "batching",
		part: "server",
		title: "Many people",
		question: "How can one server answer 32 people?",
	},
	{
		n: 9,
		label: "9",
		slug: "prefill",
		part: "server",
		title: "Reading the prompt",
		question: "Why is there a pause before the first word?",
		problem: "muse_prefill_layer",
	},
	{
		n: 10,
		label: "10",
		slug: "tradeoff",
		part: "server",
		title: "Where the speed went",
		question: "How much of the arithmetic does a server win back?",
	},
	{
		n: 11,
		label: "A",
		slug: "challenge",
		part: "appendix",
		title: "The challenge, decoded",
		question: "What was the Open Kernel Challenge asking?",
	},
];

export const href = (s: Stop) => `/${s.slug}`;

export function neighbors(slug: string) {
	const i = STOPS.findIndex((s) => s.slug === slug);
	return { prev: STOPS[i - 1], next: STOPS[i + 1] };
}
