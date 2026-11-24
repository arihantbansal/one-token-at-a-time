import { expect, test } from "vitest";
import { EXAMPLES } from "./serving";
import { speedReadout } from "./speedCalculator";

test("the guide's first example keeps its 393 tokens a second", () => {
	const example = EXAMPLES[0];
	expect(speedReadout(example.gbPerSecond, example.gb)).toEqual({
		sum: "4,800 GB/s ÷ 12.2 GB =",
		answer: "393",
		each: "one token every 2.5 ms, at best",
	});
});

test("positive speeds and intervals never display as zero", () => {
	expect(speedReadout(1, 100000)).toEqual({
		sum: "1 GB/s ÷ 100,000 GB =",
		answer: "<0.1",
		each: "one token every 100,000 s, at best",
	});
	expect(speedReadout(100000, 0.000001)).toEqual({
		sum: "100,000 GB/s ÷ 1.0e-6 GB =",
		answer: "100,000,000,000",
		each: "one token every <0.1 ms, at best",
	});
});

test.each([
	[0, 1],
	[1, 0],
	[-1, 1],
	[Number.NaN, 1],
	[1, Number.POSITIVE_INFINITY],
	[1e308, 1e-308],
])("invalid input %s / %s asks for usable values", (bandwidth, gigabytes) => {
	expect(speedReadout(bandwidth, gigabytes)).toEqual({
		sum: "enter two positive numbers",
		answer: "–",
		each: " ",
	});
});
