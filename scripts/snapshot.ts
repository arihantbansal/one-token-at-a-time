// Pulls the challenge's public API into src/data/challenge.json.
// Every number the site shows about the challenge comes from this file.
// Run with: bun run snapshot

const BASE = "https://openkernelchallenge.dev/api/v1";
const OUT = new URL("../src/data/challenge.json", import.meta.url);

type ShapeRow = { load: string; shape: string; weight: number; floorUs: number };

async function get<T>(path: string, parse: (value: unknown) => T): Promise<T> {
	const res = await fetch(`${BASE}${path}`, { signal: AbortSignal.timeout(15_000) });
	if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
	return parse(await res.json());
}

function positive(what: string, n: unknown): number {
	if (typeof n !== "number" || !Number.isFinite(n) || n <= 0)
		throw new Error(`${what}: expected a positive number, got ${n}`);
	return n;
}

// TASK.md lists each scored shape as `| load | shape | weight | floor |`.
function parseShapes(taskMd: string): ShapeRow[] {
	const rows: ShapeRow[] = [];
	for (const line of taskMd.split("\n")) {
		const cells = line.split("|").map((c) => c.trim());
		if (cells.length !== 6) continue;
		const [, load, shape, weight, floor] = cells;
		const floorMatch = floor?.match(/^(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*µs$/);
		if (!load || !shape || !floorMatch || !/^\d+$/.test(weight ?? "")) continue;
		rows.push({
			load,
			shape,
			weight: Number(weight),
			floorUs: positive(`floor of ${load} ${shape}`, Number(floorMatch[1]?.replaceAll(",", ""))),
		});
	}
	return rows;
}

// TASK.md's "How a run is timed" section, one bullet per rule.
function parseTiming(taskMd: string, name: string): string[] {
	const section = taskMd.split(/^## /m).find((s) => s.startsWith("How a run is timed"));
	const rules = (section ?? "")
		.split("\n")
		.filter((line) => line.startsWith("- "))
		.map((line) => line.slice(2).trim());
	if (!rules.length) throw new Error(`${name}: no timing rules in TASK.md`);
	return rules;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function record(value: unknown, what: string): Record<string, unknown> {
	if (!isRecord(value)) throw new Error(`${what}: expected an object`);
	return value;
}

function text(value: unknown, what: string): string {
	if (typeof value !== "string" || !value.trim()) throw new Error(`${what}: expected nonempty text`);
	return value;
}

function list(value: unknown, what: string): unknown[] {
	if (!Array.isArray(value)) throw new Error(`${what}: expected an array`);
	return value;
}

function date(value: unknown, what: string) {
	if (!Number.isFinite(Date.parse(text(value, what)))) throw new Error(`${what}: expected a valid date`);
}

function parseIndex(value: unknown) {
	const index = record(value, "problem index");
	const problems = list(index.problems, "problems").map((value) => {
		const name = text(record(value, "problem").name, "problem name");
		if (!/^[a-z0-9_]+$/.test(name)) throw new Error(`problem name: invalid ${name}`);
		return { name };
	});
	if (!problems.length) throw new Error("problem index: no problems");
	if (new Set(problems.map((p) => p.name)).size !== problems.length)
		throw new Error("problem index: duplicate names");
	const tracks = record(index.tracks, "tracks");
	date(tracks.launch, "tracks.launch");
	date(record(tracks.kernels, "kernel track").closes, "kernels.closes");
	const serving = record(tracks.serving, "serving track");
	date(serving.opens, "serving.opens");
	date(serving.closes, "serving.closes");
	return { problems, tracks };
}

function parseProblem(value: unknown) {
	const p = record(value, "problem");
	const name = text(p.name, "problem name");
	const page = record(p.page, `${name} page`);
	const benchmarks = list(p.benchmarks, `${name} benchmarks`).map((value) => {
		const b = record(value, `${name} benchmark`);
		const weight = positive(`${name} benchmark weight`, b.weight);
		if (!Number.isSafeInteger(weight)) throw new Error(`${name} benchmark weight: expected an integer`);
		return { load: text(b.load, `${name} benchmark load`), weight };
	});
	return {
		name,
		task_md: text(p.task_md, `${name} task_md`),
		benchmarks,
		sglang_us: positive(`${name} SGLang time`, p.sglang_us),
		sglang_per: text(p.sglang_per, `${name} SGLang unit`),
		page: {
			title: text(page.title, `${name} title`),
			computation: text(page.computation, `${name} computation`),
			// Problems that share another's checks have none of their own.
			correctness:
				page.correctness === undefined
					? []
					: list(page.correctness, `${name} correctness`).map((c) => text(c, `${name} correctness`)),
		},
	};
}

function parseServing(value: unknown) {
	const loads = list(record(value, "serving").concurrency, "serving concurrency").map((n) => {
		const users = positive("serving concurrency", n);
		if (!Number.isSafeInteger(users)) throw new Error("serving concurrency: expected an integer");
		return users;
	});
	if (!loads.length) throw new Error("serving concurrency: no loads");
	return loads;
}

function parseBoard(value: unknown) {
	const board = record(value, "leaderboard");
	return list(board.rows, "leaderboard rows").map((value) => {
		const row = record(value, "leaderboard row");
		return positive("leaderboard speedup", row.speedup);
	});
}

export async function createSnapshot() {
	const [index, servingLoads] = await Promise.all([
		get("/problems", parseIndex),
		get("/serving", parseServing),
	]);
	const problems = await Promise.all(
		index.problems.map(async ({ name }) => {
			const [p, board] = await Promise.all([
				get(`/problems/${name}`, parseProblem),
				get(`/leaderboard/${name}`, parseBoard),
			]);
			if (p.name !== name) throw new Error(`${name}: API returned problem ${p.name}`);
			const shapes = parseShapes(p.task_md);
			if (shapes.length === 0) throw new Error(`${name}: no shape table in TASK.md`);
			if (shapes.length !== p.benchmarks.length) {
				throw new Error(
					`${name}: TASK.md has ${shapes.length} shapes; benchmarks has ${p.benchmarks.length}`,
				);
			}
			for (const [i, benchmark] of p.benchmarks.entries()) {
				const row = shapes[i];
				if (row?.load !== benchmark.load || row?.weight !== benchmark.weight) {
					throw new Error(
						`${name}: TASK.md row ${i + 1} has load ${row?.load}, weight ${row?.weight}; benchmarks expects load ${benchmark.load}, weight ${benchmark.weight}`,
					);
				}
			}
			const best = board.reduce((top, speedup) => Math.max(top, speedup), 0);
			return {
				name,
				title: p.page.title,
				// The task brief's pseudocode for what the kernel computes, line by line.
				computation: p.page.computation,
				correctness: p.page.correctness,
				timing: parseTiming(p.task_md, name),
				sglangUs: p.sglang_us,
				sglangPer: p.sglang_per,
				shapes,
				entrants: board.length,
				best: best ? { speedup: best } : null,
			};
		}),
	);

	// The serving track's loads: how many people the server answers at once.
	return { fetchedAt: new Date().toISOString(), tracks: index.tracks, servingLoads, problems };
}

if (import.meta.main) {
	const data = await createSnapshot();
	await Bun.write(OUT, `${JSON.stringify(data, null, "\t")}\n`);
	console.log(
		`wrote ${data.problems.length} problems, ${data.problems.reduce((n, p) => n + p.shapes.length, 0)} shapes`,
	);
}
