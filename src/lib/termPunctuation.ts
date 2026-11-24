// A <Term> renders a <button>, and browsers may break a line right after one, stranding the comma
// that follows at the start of the next line. A word joiner doesn't stop that in Chrome; only a
// nowrap box around the term and its punctuation does, and the component can't reach the text after
// it. So this MDX plugin wraps them together in `.nobreak`, climbing out of bold, italic or a link
// that ends with the term.

import type { SatteriProcessorOptions } from "@astrojs/markdown-satteri";

const PUNCTUATION = /^[,.;:!?)\]’”]+/;
const INLINE = new Set(["strong", "emphasis", "delete", "link"]);

export const keepTermsWithPunctuation: NonNullable<SatteriProcessorOptions["mdastPlugins"]>[number] = {
	name: "keep-terms-with-punctuation",
	mdxJsxTextElement(node, ctx) {
		if (node.name !== "Term") return;
		let end: Parameters<typeof ctx.parent>[0] = node;
		let parent = ctx.parent(end);
		while (parent && INLINE.has(parent.type) && ctx.indexOf(end) === parent.children.length - 1) {
			end = parent;
			parent = ctx.parent(end);
		}
		const at = ctx.indexOf(end);
		const next = parent && at !== undefined ? parent.children[at + 1] : undefined;
		if (next?.type !== "text") return;
		const mark = PUNCTUATION.exec(next.value)?.[0];
		if (!mark) return;
		ctx.setField(next, "value", next.value.slice(mark.length));
		ctx.wrapNode(end, {
			type: "mdxJsxTextElement",
			name: "span",
			attributes: [{ type: "mdxJsxAttribute", name: "class", value: "nobreak" }],
			children: [{ type: "text", value: mark }],
		});
	},
};
