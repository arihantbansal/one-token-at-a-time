// The judge's timing, illustrated: ten pairs of runs of an entry `speed` times faster than SGLang,
// with made-up noise and the occasional slow run. The server renders the first race with it and
// the figure reruns it, so both show the same runs.
import { percent, times } from "./format";
import { seeded } from "./random";

const PAIRS = 10;
export const FIRST_SEED = 7;
const BASE = 100;
const FLUSH = 12;
const SCALE = 240;
const SIGMA = 0.03;
const SLOW_ODDS = 1 / 12;
const SLOW = 1.15;

export type Who = "theirs" | "yours";
type Run = { width: string; slow: boolean };

/** A bar's or flush's length, as a share of the track. */
const width = (time: number) => `${(time / SCALE) * 100}%`;
export const FLUSH_WIDTH = width(FLUSH);

const median = (xs: number[]) => {
	const s = [...xs].sort((a, b) => a - b);
	const m = Math.floor(s.length / 2);
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export function race(seed: number, speed: number) {
	const rand = seeded(seed);
	const noise = () => {
		const gauss = Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
		const slow = rand() < SLOW_ODDS;
		return { factor: Math.exp(SIGMA * gauss) * (slow ? SLOW : 1), slow };
	};
	const rows = Array.from({ length: PAIRS }, () => {
		const theirs = noise();
		const yours = noise();
		const theirTime = BASE * theirs.factor;
		const yourTime = (BASE / speed) * yours.factor;
		const runs: Record<Who, Run> = {
			theirs: { width: width(theirTime), slow: theirs.slow },
			yours: { width: width(yourTime), slow: yours.slow },
		};
		return { ratio: theirTime / yourTime, runs };
	});
	const order = rows.map((_, i) => i).sort((a, b) => rows[a].ratio - rows[b].ratio);
	return { rows, middle: [order[PAIRS / 2 - 1], order[PAIRS / 2]] };
}

/** What the judge would report from these ratios, and how far that is from the true speed. */
export function report(ratios: number[], speed: number) {
	const measured = median(ratios);
	const error = measured / speed - 1;
	return { measured: times(measured), error: `${error < 0 ? "−" : "+"}${percent(Math.abs(error))}` };
}
