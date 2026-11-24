// @ts-check
import { satteri } from "@astrojs/markdown-satteri";
import mdx from "@astrojs/mdx";
import { defineConfig } from "astro/config";
import { keepTermsWithPunctuation } from "./src/lib/termPunctuation";

const site = "https://tokens.arihantbansal.com";

// Markdown links to other sites open in a new tab. Raw <a> tags in MDX are handled in Base.astro.
const externalLinks = {
	name: "external-links",
	element: {
		filter: ["a"],
		/** @param {any} node @param {any} ctx */
		visit(node, ctx) {
			const href = String(node.properties?.href ?? "");
			if (!/^https?:\/\//.test(href) || new URL(href).origin === new URL(site).origin) return;
			ctx.setProperty(node, "target", "_blank");
			ctx.setProperty(node, "rel", "noopener noreferrer");
		},
	},
};

export default defineConfig({
	site,
	// Pages build to `kernel.html` and are linked as `/kernel`, so the served address,
	// the links and the canonical URL agree without a trailing-slash redirect.
	trailingSlash: "never",
	build: { format: "file" },
	markdown: {
		processor: satteri({ mdastPlugins: [keepTermsWithPunctuation], hastPlugins: [externalLinks] }),
	},
	integrations: [mdx()],
});
