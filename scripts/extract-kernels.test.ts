import { expect, test } from "vitest";
import { kernelSource } from "./extract-kernels";

test("preserves Python indentation and quoted code while excluding explanatory notes", () => {
	const page = `export const body = [
	{ code: "@triton.jit", note: "not Python" },
	{ code: 'def example():', note: "a plain explanation" },
	{ code: '    x = float("inf")' },
];`;
	expect(kernelSource(page, "body")).toBe(
		'import triton\nimport triton.language as tl\n\n@triton.jit\ndef example():\n    x = float("inf")\n',
	);
});

test("refuses missing, empty or computed code instead of verifying a partial kernel", () => {
	expect(() => kernelSource("", "body")).toThrow("Missing kernel array");
	expect(() => kernelSource("export const body = [\n];", "body")).toThrow("empty kernel");
	expect(() => kernelSource("export const body = [{ code: helper() }\n];", "body")).toThrow("literal string");
});
