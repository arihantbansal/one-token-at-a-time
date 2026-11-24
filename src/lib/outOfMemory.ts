// "Out of memory", the 404 page's game: one GPU's memory at toy size. One square holds one token's
// cache; the weights take their real share of the H200's memory and the rest holds caches.
import { count } from "./format";
import { H200, weightBytesPerGpu } from "./model";
import { seeded } from "./random";

export const SQUARES = 100;
export const WEIGHT_SQUARES = Math.round((SQUARES * weightBytesPerGpu) / (H200.memoryGB * 1e9));
export const CACHE_SQUARES = SQUARES - WEIGHT_SQUARES;

export const RULES = {
	seats: 8,
	queue: 5,
	strikes: 3,
	passes: 120,
	passMs: 600,
	/** Passes a conversation waits in the queue before it leaves. */
	patience: 12,
	finishBonus: 5,
	lostPenalty: 3,
} as const;

/** Reply lengths: the sum of two dice, so they run from `min` to `max` and cluster at `typical`. The
 *  player sees this spread, never a reply's own length. */
const SPREAD = 6;
export const REPLY = { min: 3, typical: 3 + SPREAD, max: 3 + 2 * SPREAD } as const;

export type Conversation = {
	id: number;
	name: string;
	prompt: number;
	reply: number;
	written: number;
	waited: number;
};
export type Game = {
	rand: () => number;
	nextId: number;
	pass: number;
	queue: Conversation[];
	batch: Conversation[];
	score: number;
	tokens: number;
	finished: number;
	lost: number;
	preemptions: number;
	mostInPass: number;
	over: "memory" | "shift" | null;
	/** What happened in the last action, for the status line. */
	events: string[];
};

export const cacheOf = (c: Conversation) => c.prompt + c.written;
export const used = (g: Game) => g.batch.reduce((sum, c) => sum + cacheOf(c), 0);
const progress = (g: Game) => g.pass / RULES.passes;

function arrive(g: Game) {
	// Later prompts run longer and people come more often.
	const id = g.nextId++;
	const die = () => Math.floor(g.rand() * (SPREAD + 1));
	g.queue.push({
		id,
		name: String.fromCharCode(65 + (id % 26)),
		prompt: 3 + Math.floor(g.rand() * (6 + 6 * progress(g))),
		reply: REPLY.min + die() + die(),
		written: 0,
		waited: 0,
	});
}

export function newGame(seed: number): Game {
	const g: Game = {
		rand: seeded(seed),
		nextId: 0,
		pass: 0,
		queue: [],
		batch: [],
		score: 0,
		tokens: 0,
		finished: 0,
		lost: 0,
		preemptions: 0,
		mostInPass: 0,
		over: null,
		events: [],
	};
	for (let i = 0; i < 3; i++) arrive(g);
	return g;
}

/** Why a queued conversation cannot join, or null when it can. Reading this leaves the game alone. */
export function admission(g: Game, slot: number): string | null {
	if (g.over) return "Restart to play again.";
	const c = g.queue[slot];
	if (!c) return "No one is waiting in this slot.";
	if (g.batch.length >= RULES.seats) return "The batch is full. Wait for a reply to finish.";
	const free = CACHE_SQUARES - used(g);
	if (c.prompt > free)
		return `${c.name} needs ${count(c.prompt)} cache squares; only ${count(free)} are free. Wait for a reply to finish.`;
	return null;
}

export function admit(g: Game, slot: number): boolean {
	const reason = admission(g, slot);
	if (reason) {
		g.events = [reason];
		return false;
	}
	const [c] = g.queue.splice(slot, 1);
	g.batch.push(c);
	g.events = [`${c.name} joins with ${c.prompt} squares of prompt.`];
	return true;
}

function lose(g: Game, c: Conversation) {
	g.lost++;
	g.score -= RULES.lostPenalty;
	g.events.push(`${c.name} gave up waiting.`);
}

/** One pass: every conversation in the batch writes a token, unless memory can't hold the new
 *  entries, in which case the latest to join is evicted first and starts again from the queue. */
export function pass(g: Game) {
	if (g.over) return;
	g.events = [];
	while (used(g) + g.batch.length > CACHE_SQUARES) {
		const c = g.batch.pop() as Conversation;
		c.written = 0;
		c.waited = 0;
		g.preemptions++;
		g.events.push(`Out of room: ${c.name} is evicted and starts over.`);
		if (g.preemptions >= RULES.strikes) {
			g.over = "memory";
			return;
		}
		g.queue.unshift(c);
		if (g.queue.length > RULES.queue) lose(g, g.queue.pop() as Conversation);
	}
	for (const c of g.batch) c.written++;
	g.tokens += g.batch.length;
	g.score += g.batch.length;
	g.mostInPass = Math.max(g.mostInPass, g.batch.length);
	for (const c of g.batch.filter((c) => c.written >= c.reply)) {
		g.finished++;
		g.score += RULES.finishBonus;
		g.events.push(`${c.name} is done and its cache is freed.`);
	}
	g.batch = g.batch.filter((c) => c.written < c.reply);
	for (const c of g.queue) c.waited++;
	for (const c of g.queue.filter((c) => c.waited >= RULES.patience)) lose(g, c);
	g.queue = g.queue.filter((c) => c.waited < RULES.patience);
	if (g.queue.length < RULES.queue && g.rand() < 0.3 + 0.4 * progress(g)) arrive(g);
	g.pass++;
	if (g.pass >= RULES.passes) g.over = "shift";
}

/** Every square of memory in order, as its class and label: the weights, each cache in the order its
 *  owner joined, the free squares the next pass will fill, and the rest. */
export function cells(g: Game): [string, string][] {
	const out: [string, string][] = Array.from({ length: WEIGHT_SQUARES }, () => ["obj-weight", ""]);
	for (const c of g.batch) for (let i = 0; i < cacheOf(c); i++) out.push(["obj-cache", c.name]);
	const free = SQUARES - out.length;
	for (let i = 0; i < free; i++) out.push([i < g.batch.length ? "next" : "free", ""]);
	return out;
}

export const readouts = (g: Game, best: number | null) => ({
	score: count(g.score),
	used: `${used(g)} of ${CACHE_SQUARES}`,
	people: `${g.batch.length} of ${RULES.seats}`,
	strikes: count(RULES.strikes - g.preemptions),
	tokens: count(g.tokens),
	finished: count(g.finished),
	most: count(g.mostInPass),
	best: best === null ? "none yet" : count(best),
	pass: `pass ${g.pass} of ${RULES.passes}`,
});

export const slotText = (c: Conversation) => ({
	sizes: `prompt ${c.prompt} · reply ?`,
	wait: `${RULES.patience - c.waited} left`,
});
