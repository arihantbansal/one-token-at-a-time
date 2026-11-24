import { expect, test } from "vitest";
import { keepTermsWithPunctuation } from "./termPunctuation";

// The same Satteri the site builds with, resolved through the Astro package that depends on it.
const { mdxToJs } = await import(
	import.meta.resolve("satteri", import.meta.resolve("@astrojs/markdown-satteri"))
);

const compile = (mdx: string): string =>
	mdxToJs(mdx, { mdastPlugins: [keepTermsWithPunctuation] }).code.replace(/\s+/g, "");

test("wraps a term and the punctuation after it", () => {
	const out = compile('Reading it is <Term id="prefill">prefill</Term>, and more.');
	expect(out).toContain('class:"nobreak",children:[_jsx(Term,{id:"prefill",children:"prefill"}),","]');
	expect(out).not.toContain(",andmore");
});

test("wraps the bold or italic text that ends with the term", () => {
	expect(compile('A **<Term id="pass">pass</Term>**, next.')).toMatch(
		/class:"nobreak",children:\[_jsx\(_components\.strong,\{children:_jsx\(Term,[^)]*\)\}\),","\]/,
	);
	expect(compile('An _early <Term id="pass">pass</Term>_. Next.')).toMatch(
		/class:"nobreak",children:\[_jsxs\(_components\.em,.*"\."\]/,
	);
});

test("leaves a term alone when no punctuation follows it", () => {
	expect(compile('A <Term id="pass">pass</Term> ends.')).not.toContain("nobreak");
	expect(compile('A **<Term id="pass">pass</Term> ends**, here.')).not.toContain("nobreak");
});
