import { expect, test, vi } from "vitest";
import { contestTimes } from "./challenge";

const fixture: { best: { speedup: number } | null } = vi.hoisted(() => ({ best: null }));

vi.mock("../data/challenge.json", async (original) => {
	const { default: data } = await original<{ default: typeof import("../data/challenge.json") }>();
	return {
		default: {
			...data,
			problems: [
				{
					...data.problems[0],
					name: "muse_bf16_proj",
					sglangUs: 120,
					get best() {
						return fixture.best;
					},
					shapes: [
						{ load: "first", weight: 2, floorUs: 10 },
						{ load: "first", weight: 1, floorUs: 20 },
						{ load: "second", weight: 3, floorUs: 30 },
					],
				},
			],
		},
	};
});

test.each([
	{ speedup: undefined, best: undefined },
	{ speedup: 0.5, best: 240 },
	{ speedup: 1, best: 120 },
	{ speedup: 1.5, best: 80 },
])("contest times support a best speedup of $speedup", ({ speedup, best }) => {
	fixture.best = speedup === undefined ? null : { speedup };
	const times = contestTimes("muse_bf16_proj");
	// The two loads take 40 and 90 µs, whose geometric mean is 60 µs.
	expect(times.ideal).toBeCloseTo(60, 10);
	expect(times.sglang).toBe(120);
	expect(times.best).toBe(best);
	expect(times.speedup).toBe(speedup);
});
