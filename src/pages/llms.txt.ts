import type { APIRoute } from "astro";
import { GLOSSARY } from "../data/glossary";
import { PARTS, STOPS } from "../data/stops";
import { fetchedAt, officialUrl, problem } from "../lib/challenge";
import { times } from "../lib/format";

export const GET: APIRoute = ({ site }) => {
	const url = (slug: string) => new URL(`/${slug}`, site).href;
	const stops = STOPS.map((s) => {
		const p = s.problem ? problem(s.problem) : undefined;
		const contest = p
			? ` Contest problem: ${p.name} (${officialUrl(p.name)}), best entry ${p.best ? times(p.best.speedup) : "none yet"} faster than SGLang.`
			: "";
		return `- ${PARTS[s.part]}, ${s.label}. [${s.title}](${url(s.slug)}): ${s.question}${contest}`;
	});
	const terms = Object.values(GLOSSARY).map((e) => `- ${e.term}: ${e.short}`);
	const body = [
		"# One token at a time",
		"",
		"> An illustrated guide to what actually happens when an AI writes a reply: what the model computes for each token, how a GPU computes it (memory, compute blocks, Triton kernels read line by line), and how a server shares that work among many people. Real sizes and timings come from the Open Kernel Challenge (https://openkernelchallenge.dev/). Independent; not affiliated with Meta, Paradigm or Andromeda. Public information only.",
		"",
		`Contest figures read from the challenge's public API on ${fetchedAt.toISOString().slice(0, 10)}.`,
		"",
		"## The guide, in order",
		"",
		...stops,
		"",
		"## Terms",
		"",
		...terms,
		"",
	].join("\n");
	return new Response(body, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
};
