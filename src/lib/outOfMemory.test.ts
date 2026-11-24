import { expect, test } from "vitest";
import { weightBytesPerGpu } from "./model";
import {
	admission,
	admit,
	CACHE_SQUARES,
	type Conversation,
	cells,
	type Game,
	newGame,
	pass,
	REPLY,
	RULES,
	SQUARES,
	slotText,
	used,
	WEIGHT_SQUARES,
} from "./outOfMemory";

const person = (id: number, prompt: number, reply: number): Conversation => ({
	id,
	name: String.fromCharCode(65 + id),
	prompt,
	reply,
	written: 0,
	waited: 0,
});

/** A game with an empty queue and no arrivals, holding these people in the order they joined. */
function game(...batch: Conversation[]): Game {
	const g = newGame(1);
	g.queue = [];
	g.batch = batch;
	g.rand = () => 0.99;
	return g;
}

test("the weights take their real share of the H200's memory", () => {
	expect(WEIGHT_SQUARES).toBe(Math.round((SQUARES * weightBytesPerGpu) / 141e9));
	expect(cells(newGame(1)).filter(([kind]) => kind === "obj-weight")).toHaveLength(WEIGHT_SQUARES);
});

test("admitting adds the prompt's cache at once, and only if it fits", () => {
	const g = game(person(0, CACHE_SQUARES - 5, 20));
	g.queue = [person(1, 6, 4), person(2, 5, 4)];
	const reason = "B needs 6 cache squares; only 5 are free. Wait for a reply to finish.";
	expect(admission(g, 0)).toBe(reason);
	expect(admit(g, 0)).toBe(false);
	expect(g.events).toEqual([reason]);
	expect(admission(g, 1)).toBeNull();
	expect(admit(g, 1)).toBe(true);
	expect(used(g)).toBe(CACHE_SQUARES);
});

test("a full batch explains the seat limit even when the prompt fits in memory", () => {
	const g = game(...Array.from({ length: RULES.seats }, (_, i) => person(i, 3, 10)));
	const waiting = person(RULES.seats, 3, 4);
	g.queue = [waiting];
	expect(CACHE_SQUARES - used(g)).toBeGreaterThan(waiting.prompt);
	const reason = "The batch is full. Wait for a reply to finish.";
	expect(admission(g, 0)).toBe(reason);
	expect(admit(g, 0)).toBe(false);
	expect(g.events).toEqual([reason]);
	expect(g.batch).toHaveLength(RULES.seats);
	expect(g.queue).toEqual([waiting]);
});

test("each pass, every person in the batch writes one token and their cache grows by one", () => {
	const g = game(person(0, 3, 10), person(1, 5, 10));
	pass(g);
	pass(g);
	expect(g.batch.map((c) => c.written)).toEqual([2, 2]);
	expect(used(g)).toBe(3 + 5 + 4);
	expect(g.tokens).toBe(4);
	expect(g.score).toBe(4);
	expect(g.mostInPass).toBe(2);
});

test("a finished reply frees its cache and scores the bonus", () => {
	const g = game(person(0, 3, 2), person(1, 4, 5));
	pass(g);
	pass(g);
	expect(g.batch.map((c) => c.name)).toEqual(["B"]);
	expect(used(g)).toBe(4 + 2);
	expect(g.finished).toBe(1);
	expect(g.score).toBe(4 + RULES.finishBonus);
});

test("when the next pass won't fit, the latest to join is evicted, loses its progress and waits again", () => {
	const g = game(person(0, 20, 30), person(1, 15, 30), person(2, 10, 30));
	for (const c of g.batch) c.written = 2;
	// 51 squares hold 20 + 15 + 10 + 3 × 2 = 51, so the next three tokens need room someone must give up.
	pass(g);
	expect(g.preemptions).toBe(1);
	expect(g.batch.map((c) => c.name)).toEqual(["A", "B"]);
	expect(g.queue[0]).toMatchObject({ name: "C", written: 0 });
	expect(g.over).toBeNull();
});

test("the third preemption ends the run", () => {
	const g = game(person(0, 40, 30), person(1, 4, 30), person(2, 3, 30), person(3, 2, 30));
	g.batch[0].written = 2;
	pass(g);
	expect(g.preemptions).toBe(2);
	g.batch.push(person(4, CACHE_SQUARES - used(g), 30));
	pass(g);
	expect(g.preemptions).toBe(RULES.strikes);
	expect(g.over).toBe("memory");
});

test("someone left waiting too long leaves and costs points", () => {
	const g = game();
	g.queue = [person(0, 3, 4)];
	for (let i = 0; i < RULES.patience; i++) pass(g);
	expect(g.queue).toHaveLength(0);
	expect(g.lost).toBe(1);
	expect(g.score).toBe(-RULES.lostPenalty);
});

test("the shift ends after its last pass", () => {
	const g = game();
	for (let i = 0; i < RULES.passes + 5; i++) pass(g);
	expect(g.pass).toBe(RULES.passes);
	expect(g.over).toBe("shift");
});

test("the queue hides each reply's length, and the stated spread matches the generator", () => {
	const replies: number[] = [];
	for (let seed = 1; replies.length < 3000; seed++) {
		const g = newGame(seed);
		expect(slotText(g.queue[0]).sizes).toMatch(/reply \?$/);
		replies.push(...g.queue.map((c) => c.reply));
	}
	expect(Math.min(...replies)).toBe(REPLY.min);
	expect(Math.max(...replies)).toBe(REPLY.max);
	const often = (n: number) => replies.filter((r) => r === n).length;
	expect(often(REPLY.typical)).toBeGreaterThan(often(REPLY.typical - 3));
	expect(often(REPLY.typical)).toBeGreaterThan(often(REPLY.typical + 3));
});
