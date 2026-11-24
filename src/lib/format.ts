const en = (n: number, digits: number) => n.toLocaleString("en-US", { maximumFractionDigits: digits });

/** Decimal input values without rounding a nonzero quantity to zero. */
export const decimal = (n: number, digits = 2) =>
	n !== 0 && Math.abs(n) < 10 ** -digits ? n.toExponential(1) : en(n, digits);

/** Microseconds, switching to milliseconds from 1,000 µs. */
export function us(n: number): string {
	if (n >= 1000) return `${en(n / 1000, 2)} ms`;
	return `${en(n, n < 10 ? 2 : 1)} µs`;
}

/** Microseconds at every size, for times shown side by side: 565 µs beside 5,520 µs, never 5.52 ms. */
export const microseconds = (n: number) => `${en(n, n < 10 ? 2 : n < 1000 ? 1 : 0)} µs`;

/** Bytes as MB or GB. */
export function bytes(n: number): string {
	if (n >= 1e9) return `${en(n / 1e9, 2)} GB`;
	if (n >= 1e6) return `${en(n / 1e6, 1)} MB`;
	return `${en(n / 1e3, 1)} KB`;
}

export const times = (x: number, digits = 2) => `${x.toFixed(digits)}×`;

export const count = (n: number) => en(n, 0);

export const percent = (fraction: number) => {
	const p = fraction * 100;
	return `${p < 1 ? p.toFixed(1) : Math.round(p)}%`;
};

/** A whole number of hundredths as a two-decimal number: 7 → "0.07". Worked examples count in
 *  hundredths so their displayed sums are exact. */
export const hundredths = (c: number) => (c / 100).toFixed(2);

/** Rounded to hundredths with both decimals shown and a real minus sign: 0.5 → "0.50", -0.155 → "−0.16". */
export const twoPlaces = (n: number) => (Math.round(n * 100) / 100).toFixed(2).replace("-", "−");

/** A hand-checkable number: rounded to two decimals, at least one shown, with a real minus sign.
 *  1 → "1.0", -0.155 → "−0.16". */
export function worked(n: number): string {
	const r = Math.round(n * 100) / 100;
	const s = String(Math.abs(r));
	return `${r < 0 ? "−" : ""}${s.includes(".") ? s : `${s}.0`}`;
}
