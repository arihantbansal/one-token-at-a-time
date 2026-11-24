import { afterEach, expect, test, vi } from "vitest";
import { createSnapshot } from "./snapshot";

afterEach(() => vi.unstubAllGlobals());

const tracks = {
	launch: "2026-10-22T11:00:00-07:00",
	kernels: { closes: "2026-11-05T00:00:00-08:00" },
	serving: { opens: "2026-11-10T09:00:00-08:00", closes: "2026-11-20T00:00:00-08:00" },
};
const problem = {
	name: "muse_decode_layer",
	task_md:
		"| decode 1 | layer sliding | 46 | 32.1 µs |\n\n## How a run is timed\n\n- Replayed as a CUDA graph.\n- Cold L2.\n\n## Rules\n\n- Not timing.",
	benchmarks: [{ load: "decode 1", weight: 46 }],
	sglang_us: 123,
	sglang_per: "decode step",
	page: { title: "One whole layer", computation: "y = layer(x)", correctness: ["Within 3%."] },
};

function publicApi(overrides: Record<string, unknown> = {}) {
	const responses: Record<string, unknown> = {
		"/problems": { problems: [{ name: problem.name }], tracks },
		"/serving": { concurrency: [1, 4] },
		[`/problems/${problem.name}`]: problem,
		[`/leaderboard/${problem.name}`]: { rows: [{ speedup: 1.5 }, { speedup: 2 }] },
		...overrides,
	};
	vi.stubGlobal("fetch", async (input: string) => {
		const path = new URL(input).pathname.replace("/api/v1", "");
		if (!(path in responses)) throw new Error(`Unexpected request: ${path}`);
		return new Response(JSON.stringify(responses[path]));
	});
}

test("builds a snapshot with the scored shapes and fastest entry", async () => {
	publicApi();
	const result = await createSnapshot();
	expect(result.tracks).toEqual(tracks);
	expect(result.servingLoads).toEqual([1, 4]);
	expect(result.problems).toEqual([
		{
			name: problem.name,
			title: problem.page.title,
			computation: "y = layer(x)",
			correctness: ["Within 3%."],
			timing: ["Replayed as a CUDA graph.", "Cold L2."],
			sglangUs: 123,
			sglangPer: "decode step",
			shapes: [{ load: "decode 1", shape: "layer sliding", weight: 46, floorUs: 32.1 }],
			entrants: 2,
			best: { speedup: 2 },
		},
	]);
});

test("accepts an empty leaderboard", async () => {
	publicApi({ [`/leaderboard/${problem.name}`]: { rows: [] } });
	expect((await createSnapshot()).problems[0]).toMatchObject({ entrants: 0, best: null });
});

test.each([
	[{ ...problem, page: {} }, /title/],
	[{ ...problem, page: { title: "One whole layer" } }, /computation/],
	[{ ...problem, sglang_us: "123" }, /SGLang time/],
	[{ ...problem, task_md: "| decode 1 | layer sliding | 16 | 32.1 µs |" }, /benchmarks expects/],
	[{ ...problem, task_md: "| decode 1 | layer sliding | 46 | 32.1 µs |" }, /timing rules/],
	[{ ...problem, name: "another_problem" }, /API returned problem/],
] as const)("rejects invalid problem data: %j", async (detail, error) => {
	publicApi({ [`/problems/${problem.name}`]: detail });
	await expect(createSnapshot()).rejects.toThrow(error);
});

test("accepts a problem without checks of its own", async () => {
	const { correctness: _, ...page } = problem.page;
	publicApi({ [`/problems/${problem.name}`]: { ...problem, page } });
	expect((await createSnapshot()).problems[0]?.correctness).toEqual([]);
});

test("rejects serving loads that aren't whole numbers of people", async () => {
	publicApi({ "/serving": { concurrency: [1.5] } });
	await expect(createSnapshot()).rejects.toThrow(/serving concurrency/);
});

test("rejects invalid track dates", async () => {
	publicApi({
		"/problems": { problems: [{ name: problem.name }], tracks: { ...tracks, launch: "tomorrow" } },
	});
	await expect(createSnapshot()).rejects.toThrow(/launch/);
});

test("reports HTTP failures with the failing endpoint", async () => {
	vi.stubGlobal("fetch", async () => new Response(null, { status: 503 }));
	await expect(createSnapshot()).rejects.toThrow("/problems: HTTP 503");
});
