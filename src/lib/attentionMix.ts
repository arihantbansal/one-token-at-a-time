// The worked attention step AttentionMix shows: three tokens, two numbers each, small enough to
// check by hand. The server renders the first query with it and the figure redraws with it, so
// both show the same numbers. Everything is worked in hundredths, so every displayed sum is exact.
import { hundredths } from "./format";

type Token = { word: string; key: number[]; value: number[] };

// "bank" is the new token; it looks at itself and the two tokens before it. Each query has a
// different clear winner (river, the, bank), and for each one the rounded shares add to 1.00.
export const TOKENS: Token[] = [
	{ word: "the", key: [0, 2], value: [1, 0] },
	{ word: "river", key: [4, 0], value: [0, 2] },
	{ word: "bank", key: [3, 1], value: [3, 1] },
];
export const QUERIES = [
	[1, 0],
	[0, 1],
	[1, 2],
];

export const pair = ([a, b]: number[], format: (n: number) => string = String) =>
	`(${format(a)}, ${format(b)})`;
const cents = (x: number) => Math.round(x * 100);

export function attend(query: number[]) {
	const scores = TOKENS.map((t) => t.key.reduce((sum, k, i) => sum + k * query[i], 0));
	const exps = scores.map((s) => cents(Math.exp(s)));
	const expTotal = exps.reduce((a, b) => a + b, 0);
	const shares = exps.map((e) => cents(e / expTotal));
	const products = TOKENS.map((t, j) => t.value.map((v) => shares[j] * v));
	const blend = [0, 1].map((i) => products.reduce((sum, p) => sum + p[i], 0));
	const best = shares.indexOf(Math.max(...shares));
	return {
		rows: TOKENS.map((t, j) => ({
			word: t.word,
			score: `${t.key.map((k, i) => `${k}×${query[i]}`).join(" + ")} = ${scores[j]}`,
			power: String(scores[j]),
			exp: hundredths(exps[j]),
			share: hundredths(shares[j]),
			width: `${shares[j]}%`,
			product: pair(products[j], hundredths),
		})),
		expTotal: hundredths(expTotal),
		total: hundredths(shares.reduce((a, b) => a + b, 0)),
		blend: pair(blend, hundredths),
		note: `"${TOKENS[best].word}" has the highest score, so it gets the largest share (${hundredths(shares[best])}) and the blend leans toward its value.`,
	};
}
