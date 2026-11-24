import { expect, test } from "vitest";
import challenge from "../data/challenge.json";
import {
	ACTIVE_WEIGHTS,
	ARITHMETIC_TOKENS_PER_SECOND,
	attentionFlops,
	EXPERT_WEIGHTS,
	FP32_BALANCE_POINT,
	floorUs,
	fp32MathUs,
	getProjection,
	H200,
	LAYERS,
	LONG_CONTEXT_ARITHMETIC_TOKENS_PER_SECOND,
	MOE_PROBLEM_LAYERS,
	mathUs,
	memoryUs,
	N_EXPERTS,
	PREFILL_CHUNK,
	projectionCost,
	TOP_K,
	TOTAL_WEIGHTS,
} from "./model";

test("every muse_bf16_proj shape matches its official floor within 0.06 µs or 1%", () => {
	const problem = challenge.problems.find((p) => p.name === "muse_bf16_proj");
	if (!problem) throw new Error("muse_bf16_proj is missing from challenge.json");
	expect(problem.shapes.length).toBeGreaterThan(0);
	for (const { shape, floorUs: officialFloorUs, weight } of problem.shapes) {
		expect(weight, shape).toBe(LAYERS.total);
		const [, m, layer] = /^m (\d+), layer (\w+)$/.exec(shape) ?? [];
		if (!m || !layer) throw new Error(`Unrecognized projection shape: ${shape}`);
		const actual = floorUs(projectionCost(getProjection(layer), Number(m)));
		expect(Math.abs(actual - officialFloorUs), shape).toBeLessThanOrEqual(
			Math.max(0.06, officialFloorUs * 0.01),
		);
	}
});

test("FP32 CUDA cores sustain about 14 FLOPs in the time it takes to read one byte", () => {
	expect(H200.fp32FlopsPerSecond).toBe(67e12);
	expect(FP32_BALANCE_POINT).toBeCloseTo(13.9583, 4);
	expect(fp32MathUs(67e6)).toBe(1);
	expect(fp32MathUs(FP32_BALANCE_POINT * 1e6)).toBeCloseTo(memoryUs(1e6), 12);
});

test("attention counts all heads and caps only the sliding layers at 2,048 tokens", () => {
	expect(attentionFlops(0)).toBe(0);
	expect(attentionFlops(1_000)).toBe(2_031_616_000);
	expect(attentionFlops(2_049) - attentionFlops(2_048)).toBe(524_288);
	expect(attentionFlops(80_000)).toBe(45_030_047_744);
});

test("the arithmetic ceiling drops from about 94k to 61k tokens per second at 80k context", () => {
	expect(Math.round(ARITHMETIC_TOKENS_PER_SECOND)).toBe(94_190);
	expect(Math.round(LONG_CONTEXT_ARITHMETIC_TOKENS_PER_SECOND)).toBe(61_319);
});

test("the standalone expert benchmark's weight is 60, while the guide counts 62 sets of experts", () => {
	const problem = challenge.problems.find((p) => p.name === "muse_moe_w4a16");
	if (!problem) throw new Error("muse_moe_w4a16 is missing from challenge.json");
	expect(new Set(problem.shapes.map((s) => s.weight))).toEqual(new Set([MOE_PROBLEM_LAYERS]));
	expect(MOE_PROBLEM_LAYERS).toBe(60);

	const experts = LAYERS.total * N_EXPERTS * EXPERT_WEIGHTS;
	const picked = LAYERS.total * TOP_K * EXPERT_WEIGHTS;
	const outside = TOTAL_WEIGHTS - experts;
	expect(Math.round(experts / 1e9)).toBe(799);
	expect(Math.round(picked / 1e9)).toBe(25);
	expect(Math.round(outside / 1e9)).toBe(17);
	expect(Math.round((outside + picked) / 1e9)).toBe(ACTIVE_WEIGHTS / 1e9);
});

test("the illustrated projection waits on memory for one token and arithmetic for a prompt chunk", () => {
	const projection = getProjection("pooler");
	const one = projectionCost(projection, 1);
	const chunk = projectionCost(projection, PREFILL_CHUNK);
	expect(memoryUs(one.bytes)).toBeGreaterThan(mathUs(one.flops));
	expect(mathUs(chunk.flops)).toBeGreaterThan(memoryUs(chunk.bytes));
	expect(floorUs(chunk) / floorUs(one)).toBeLessThan(PREFILL_CHUNK);
});
