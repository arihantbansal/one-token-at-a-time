import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import ts from "typescript";
import { ATTN_BLOCK, ATTN_SPLITS, PROJ_BLOCK_K, PROJ_BLOCK_N } from "../src/lib/kernels";
import { CONTEXT_TOKENS, getProjection, HEAD_DIM, Q_HEADS } from "../src/lib/model";

export function kernelSource(page: string, name: string): string {
	const array = new RegExp(`^export const ${name} = (\\[[\\s\\S]*?^\\]);`, "m").exec(page)?.[1];
	if (!array) throw new Error(`Missing kernel array ${name}; update the extractor if its location changed`);
	const file = ts.createSourceFile("kernel.ts", `const lines = ${array};`, ts.ScriptTarget.Latest, true);
	const statement = file.statements[0];
	const value = ts.isVariableStatement(statement) && statement.declarationList.declarations[0].initializer;
	if (!value || !ts.isArrayLiteralExpression(value)) throw new Error(`${name}: expected an array of lines`);
	const lines = value.elements.map((line) => {
		const code =
			ts.isObjectLiteralExpression(line) &&
			line.properties.find(
				(property) => ts.isPropertyAssignment(property) && property.name.getText(file) === "code",
			);
		if (
			!code ||
			!ts.isPropertyAssignment(code) ||
			!(ts.isStringLiteral(code.initializer) || ts.isNoSubstitutionTemplateLiteral(code.initializer))
		)
			throw new Error(`${name}: each code line must be a literal string; extract the displayed code exactly`);
		return code.initializer.text;
	});
	if (!lines.length) throw new Error(`${name}: empty kernel; restore the displayed code`);
	return `import triton\nimport triton.language as tl\n\n${lines.join("\n")}\n`;
}

if (import.meta.main) {
	const directory = resolve(process.argv[2] ?? ".astro/kernel-check");
	await mkdir(directory, { recursive: true });
	for (const [page, name, filename] of [
		["kernel", "toy", "k_toy.py"],
		["kernel", "tiled", "k_tiled.py"],
		["layer", "fused", "k_fused.py"],
		["attention-kernel", "body", "k_slice.py"],
	]) {
		const source = await readFile(new URL(`../src/pages/${page}.mdx`, import.meta.url), "utf8");
		await writeFile(resolve(directory, filename), kernelSource(source, name));
	}
	const { rows, cols } = getProjection("pooler");
	await writeFile(
		resolve(directory, "consts.json"),
		JSON.stringify({
			PROJ_BLOCK_N,
			PROJ_BLOCK_K,
			ATTN_BLOCK,
			ATTN_SPLITS,
			HEAD_DIM,
			Q_HEADS,
			CONTEXT_TOKENS,
			rows,
			cols,
		}),
	);
	console.log(`Extracted four displayed kernels and model constants into ${directory}`);
}
