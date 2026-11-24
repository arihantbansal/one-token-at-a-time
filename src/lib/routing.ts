// The invented routing RouterGrid draws: each token picks TOP_K of the N_EXPERTS experts,
// skewed the way the contest's routings are. The first LOCAL_EXPERTS are this GPU's. The server renders
// one batch size with it and the figure redraws with it, so both show the same routing.
import { bytes, count } from "./format";
import { EXPERT_WEIGHTS, LOCAL_EXPERTS, N_EXPERTS, NVFP4_BYTES, TOP_K } from "./model";
import { seeded } from "./random";

// Strong skew, as in the contest's routings, where a few experts per layer take most tokens. Tuned
// so this GPU's distinct experts track what the judge's floors imply (serving.test.ts).
const POPULARITY_SKEW = 1.6;
// Popularity ranks given to experts on this GPU, so the GPU drawn in full is the busiest.
const LOCAL_TOP: [rank: number, expert: number][] = [
	[0, 21],
	[3, 5],
];
const SEED = 7;

function shuffled(n: number, rand: () => number): number[] {
	const a = Array.from({ length: n }, (_, i) => i);
	for (let i = n - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		[a[i], a[j]] = [a[j], a[i]];
	}
	return a;
}

function popularity(total: number): number[] {
	const pinned = new Set(LOCAL_TOP.map(([, e]) => e));
	const order = shuffled(total, seeded(SEED)).filter((e) => !pinned.has(e));
	for (const [rank, expert] of LOCAL_TOP) order.splice(rank, 0, expert);
	const weight = new Array<number>(total).fill(0);
	order.forEach((e, rank) => {
		weight[e] = 1 / (rank + 1) ** POPULARITY_SKEW;
	});
	return weight;
}

/** Each token's picks, drawn from one fixed stream so a larger batch adds tokens to a smaller one. */
function route(tokens: number, total: number, top: number): number[] {
	const weight = popularity(total);
	const rand = seeded(SEED * 7919);
	const counts = new Array<number>(total).fill(0);
	for (let t = 0; t < tokens; t++) {
		const left = [...weight];
		let sum = left.reduce((a, b) => a + b, 0);
		for (let k = 0; k < top; k++) {
			let r = rand() * sum;
			let pick = 0;
			for (; pick < total - 1; pick++) {
				r -= left[pick];
				if (r < 0 && left[pick] > 0) break;
			}
			counts[pick]++;
			sum -= left[pick];
			left[pick] = 0;
		}
	}
	return counts;
}

/** One batch's routing, and what it means for this GPU. */
export function routing(tokens: number) {
	const counts = route(tokens, N_EXPERTS, TOP_K);
	const here = counts.slice(0, LOCAL_EXPERTS);
	const used = here.filter((n) => n > 0).length;
	const pairs = here.reduce((a, b) => a + b, 0);
	return {
		counts,
		distinct: used,
		used: `${count(used)} of ${count(LOCAL_EXPERTS)}`,
		pairs: count(pairs),
		read: bytes(used * EXPERT_WEIGHTS * NVFP4_BYTES),
		label: `An example routing of ${count(tokens)} ${tokens === 1 ? "token" : "tokens"}: ${count(pairs)} of their picks land on this GPU, on ${count(used)} of its ${count(LOCAL_EXPERTS)} experts.`,
	};
}

/** How a count shows in a cell: a dot for an expert nobody chose. */
export const cellText = (n: number) => (n === 0 ? "·" : count(n));
