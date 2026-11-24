// The running-total example that the whole-layer stop follows: the toy token's four numbers, and
// what attention and the experts mix into them. The results are made up and already normalized
// (their typical size is 1); every displayed number is computed from the rounded ones before it.
import { X } from "./toy";

/** One gate setting for both mixes: the old total is scaled by KEEP and the normalized result by
 *  MIX, and a learned gate always pairs them so that KEEP² + MIX² = 1. */
export const KEEP = 0.6;
export const MIX = 0.8;

export const ATTENTION_RESULT = [1.0, -1.0, 1.4, -0.2];
export const EXPERTS_RESULT = [-0.2, 1.4, 1.0, -1.0];

const round = (n: number) => Math.round(n * 100) / 100;
export const scale = (xs: number[], k: number) => xs.map((x) => round(x * k));
const add = (a: number[], b: number[]) => a.map((x, i) => round(x + b[i]));

/** The typical size a normalization divides by (the root of the mean square), as displayed. */
export const size = (xs: number[]) => round(Math.sqrt(xs.reduce((s, x) => s + x * x, 0) / xs.length));
export const normalized = (xs: number[]) => xs.map((x) => round(x / size(xs)));

const mix = (total: number[], result: number[]) => add(scale(total, KEEP), scale(result, MIX));

const T1 = mix(X, ATTENTION_RESULT);
export const TOTALS = [X, T1, mix(T1, EXPERTS_RESULT)];
