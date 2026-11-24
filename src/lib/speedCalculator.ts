import { count, decimal } from "./format";

/** The calculator's displayed result, shared by the first render and reader input. */
export function speedReadout(bandwidth: number, gigabytes: number) {
	const speed = bandwidth / gigabytes;
	const milliseconds = 1000 / speed;
	if (
		!Number.isFinite(bandwidth) ||
		!Number.isFinite(gigabytes) ||
		bandwidth <= 0 ||
		gigabytes <= 0 ||
		!Number.isFinite(speed) ||
		!Number.isFinite(milliseconds) ||
		speed <= 0
	) {
		return { sum: "enter two positive numbers", answer: "–", each: " " };
	}
	const interval =
		milliseconds < 0.1
			? "<0.1 ms"
			: milliseconds < 1000
				? `${decimal(milliseconds, 1)} ms`
				: `${decimal(milliseconds / 1000, 1)} s`;
	return {
		sum: `${decimal(bandwidth)} GB/s ÷ ${decimal(gigabytes)} GB =`,
		answer: speed < 0.1 ? "<0.1" : speed < 10 ? decimal(speed, 1) : count(speed),
		each: `one token every ${interval}, at best`,
	};
}
