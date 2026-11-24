// The guide's toy projection: four inputs, three outputs, twelve weights.
export const W = [
	[0.2, -0.5, 1.0, 0.3],
	[0.7, 0.1, -0.4, 0.6],
	[-0.3, 0.8, 0.5, -0.1],
];
export const X = [1.0, 0.5, -1.0, 2.0];

export const products = (row: number) => W[row].map((w, i) => w * X[i]);
export const output = (row: number) => products(row).reduce((a, b) => a + b, 0);
