import { expect, test } from "vitest";
import { decodeThroughput, problem } from "./challenge";
import {
	attentionFlops,
	CONTEXT_TOKENS,
	EXPERT_WEIGHTS,
	GPUS,
	LAYERS,
	LOCAL_EXPERTS,
	mathUs,
	N_EXPERTS,
	PROJECTIONS,
	projectionCost,
	TOP_K,
} from "./model";
import { routing } from "./routing";
import {
	billLines,
	EXAMPLES,
	LENGTH_SPREAD,
	passArithmeticCeilingUs,
	passBill,
	ROUTING_SHARE,
	tokensPerSecond,
} from "./serving";

const PEOPLE = [1, 16, 32];

test.each(PEOPLE)("%i people: the displayed lines add up to the displayed total", (people) => {
	const bill = billLines(people);
	expect(bill.lines.reduce((sum, l) => sum + l.us, 0)).toBe(bill.totalUs);
	expect(bill.lines.every((l) => l.us > 0)).toBe(true);
	expect(bill.each).toBeCloseTo(decodeThroughput(people).each, 0);
});

test.each(PEOPLE)("%i people: arithmetic finishes first, so moving data is the whole floor", (people) => {
	expect(passArithmeticCeilingUs(people)).toBeLessThan(passBill(people).totalUs);
});

test.each(PEOPLE)("%i people: serving and model agree on the attention arithmetic", (people) => {
	const share = people === 1 ? 1 : 1 - LENGTH_SPREAD;
	const attention = (people * share * attentionFlops(CONTEXT_TOKENS)) / GPUS;
	const projections = PROJECTIONS.reduce((sum, p) => sum + projectionCost(p, people).flops, 0);
	const experts = 2 * people * TOP_K * EXPERT_WEIGHTS;
	expect(passArithmeticCeilingUs(people)).toBeCloseTo(
		mathUs(attention + LAYERS.total * (projections + experts)),
		10,
	);
});

test("the one-token decoder floor sums both sets of layers", () => {
	const decoder = problem("muse_decode_layer");
	const shapes = decoder.shapes.filter((s) => s.load === "decode 1");
	expect(shapes).toHaveLength(2);
	expect(shapes.find((s) => s.shape.includes("layer sliding"))?.weight).toBe(LAYERS.sliding);
	expect(shapes.find((s) => s.shape.includes("layer full"))?.weight).toBe(LAYERS.full);
	expect(shapes.reduce((sum, s) => sum + s.weight, 0)).toBe(LAYERS.total);
	expect(passBill(1).totalUs).toBeCloseTo(2_547.2, 10);
});

test("the experts left over are plausible: within the judge's own bounds, and growing with people", () => {
	const counts = PEOPLE.map((n) => passBill(n).expertsPerLayer);
	expect(counts[0]).toBeGreaterThanOrEqual(1);
	expect(counts[1]).toBeGreaterThanOrEqual(2);
	expect(Math.max(...counts)).toBeLessThan(LOCAL_EXPERTS);
	expect(counts[0]).toBeLessThan(counts[1]);
	expect(counts[1]).toBeLessThan(counts[2]);
});

// For one token, the busiest GPU's share of its 8 picks among 8 GPUs of 32 experts, counted
// exactly: P(max <= m) is the coefficient of x^8 in (sum over j <= m of C(32, j) x^j)^8 / C(256, 8).
function busiestOfOneToken() {
	const choose = (n: number, k: number) => {
		let r = 1;
		for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
		return r;
	};
	const atMost = (m: number) => {
		let poly = [1];
		for (let g = 0; g < GPUS; g++) {
			const next = new Array(TOP_K + 1).fill(0);
			poly.forEach((c, i) => {
				for (let j = 0; j <= m && i + j <= TOP_K; j++) next[i + j] += c * choose(LOCAL_EXPERTS, j);
			});
			poly = next;
		}
		return poly[TOP_K] / choose(N_EXPERTS, TOP_K);
	};
	let expected = 0;
	for (let m = 0; m < TOP_K; m++) expected += 1 - atMost(m);
	return expected;
}

test("for one person, the experts the floor leaves match three quarters of the busiest GPU's picks", () => {
	const expected = ROUTING_SHARE * busiestOfOneToken();
	// About 1.87 implied against 1.92 expected: 5% leaves room for the judge's routing targets, which it doesn't publish.
	expect(Math.abs(passBill(1).expertsPerLayer / expected - 1)).toBeLessThan(0.05);
});

test.each(PEOPLE)(
	"the routing figure's %i tokens touch about as many experts as the floor implies",
	(people) => {
		const implied = passBill(people).expertsPerLayer / ROUTING_SHARE;
		expect(Math.abs(routing(people).distinct - implied)).toBeLessThanOrEqual(Math.max(1, implied * 0.2));
	},
);

test("the calculator's first state is the guide's own number for one person", () => {
	const muse = EXAMPLES[0];
	expect(muse.id).toBe("muse");
	expect(Math.round(tokensPerSecond(muse.gbPerSecond, muse.gb))).toBe(Math.round(decodeThroughput(1).each));
});
