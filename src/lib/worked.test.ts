import { expect, test } from "vitest";
import { attend, QUERIES, TOKENS } from "./attentionMix";
import { shapeFloor } from "./challenge";
import { CONTEXT_TOKENS, HEAD_DIM, kvBytes, mathUs, memoryUs, Q_HEADS, TURN_TOKENS } from "./model";
import { ATTENTION_RESULT, EXPERTS_RESULT, KEEP, MIX, size } from "./runningTotal";

test.each(QUERIES)("query (%i, %i): the displayed shares add to exactly 1.00", (...query) => {
	expect(attend(query).total).toBe("1.00");
});

test("each query has a different winner, so switching queries moves the blend", () => {
	const winners = QUERIES.map((q) => attend(q).note);
	expect(new Set(winners).size).toBe(QUERIES.length);
	expect(TOKENS.at(-1)?.word).toBe("bank");
});

test("the running total's results are already normalized, and the gate's scales pair up", () => {
	expect(size(ATTENTION_RESULT)).toBe(1);
	expect(size(EXPERTS_RESULT)).toBe(1);
	expect(KEEP ** 2 + MIX ** 2).toBeCloseTo(1, 12);
});

test("the published full-layer floor is consistent with the extra attention arithmetic", () => {
	const extra =
		shapeFloor("muse_prefill_layer", "turn 1", "layer full") -
		shapeFloor("muse_prefill_layer", "turn 1", "layer sliding");
	expect(memoryUs(kvBytes(CONTEXT_TOKENS, "full"))).toBeLessThan(extra / 20);
	// A multiply and add for each query-key score, and again to blend the values.
	const attention = mathUs(4 * TURN_TOKENS * CONTEXT_TOKENS * Q_HEADS * HEAD_DIM);
	expect(attention / extra).toBeGreaterThan(0.8);
	expect(attention / extra).toBeLessThan(1.2);
});
