import data from "../data/challenge.json";
import { H200 } from "./model";

export type ProblemName =
	| "muse_bf16_proj"
	| "muse_attention_decode"
	| "muse_moe_w4a16"
	| "muse_decode_layer"
	| "muse_prefill_layer";

type Problem = (typeof data.problems)[number];

export const fetchedAt = new Date(data.fetchedAt);

const pacificDay = (d: Date) =>
	d.toLocaleDateString("en-US", { month: "long", day: "numeric", timeZone: "America/Los_Angeles" });
/** A track's first and last days in Pacific time, from the snapshot; it closes at the start of the day after its last. */
const trackDays = (opens: string, closes: string) =>
	`${pacificDay(new Date(opens))} to ${pacificDay(new Date(Date.parse(closes) - 1))}`;
export const kernelTrack = trackDays(data.tracks.launch, data.tracks.kernels.closes);
export const servingTrack = trackDays(data.tracks.serving.opens, data.tracks.serving.closes);
export const contestYear = new Date(data.tracks.launch).getFullYear();

export function problem(name: ProblemName): Problem {
	const p = data.problems.find((x) => x.name === name);
	if (!p) throw new Error(`challenge.json has no ${name}; run bun run snapshot`);
	return p;
}

/** A load's time at the floor: each shape's floor times how often one step runs it. */
export function loadFloors(name: ProblemName): { load: string; us: number }[] {
	const loads = new Map<string, number>();
	for (const s of problem(name).shapes) loads.set(s.load, (loads.get(s.load) ?? 0) + s.weight * s.floorUs);
	return [...loads].map(([load, us]) => ({ load, us }));
}

/**
 * Bytes one GPU moves in one pass for one person, at the floor. With one token every step is bound by
 * moving data, so the floor times the bandwidth is the bytes it reads and writes.
 */
export function passBytesPerGpu(): number {
	const load = loadFloors("muse_decode_layer").find((l) => l.load === "decode 1");
	if (!load) throw new Error("muse_decode_layer has no decode 1 load");
	return load.us * 1e-6 * H200.bytesPerSecond;
}

/** Ideal tokens per second through a pass's layers, without serving overhead. */
export function decodeThroughput(people: number) {
	const load = loadFloors("muse_decode_layer").find((l) => l.load === `decode ${people}`);
	if (!load) throw new Error(`muse_decode_layer has no decode ${people} load`);
	return { total: (people * 1e6) / load.us, each: 1e6 / load.us };
}

/** A problem's times per pass, averaged over its loads the way the judge averages speedups (the
 *  geometric mean): the ideal from its floors, SGLang's measured time, and the best entry's (SGLang's
 *  divided by the best speedup, as the board shows it). `best` and `speedup` are undefined until an
 *  entry is judged. */
export function contestTimes(name: ProblemName) {
	const p = problem(name);
	const loads = loadFloors(name);
	const ideal = Math.exp(loads.reduce((sum, l) => sum + Math.log(l.us), 0) / loads.length);
	const speedup = p.best?.speedup;
	return {
		ideal,
		sglang: p.sglangUs,
		best: speedup === undefined ? undefined : p.sglangUs / speedup,
		speedup,
	};
}

export const officialUrl = (name: string) => `https://openkernelchallenge.dev/problems/${name}`;

export const formatDate = (d: Date) =>
	d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric", timeZone: "UTC" });

/** One shape's floor, found by its load and a fragment of its shape description that matches exactly one row. */
export function shapeFloor(name: ProblemName, load: string, shape: string): number {
	const matches = problem(name).shapes.filter((x) => x.load === load && x.shape.includes(shape));
	if (matches.length !== 1)
		throw new Error(`${name} has ${matches.length} "${load}" shapes matching "${shape}"; expected 1`);
	return matches[0].floorUs;
}
