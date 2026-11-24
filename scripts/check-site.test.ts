import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, expect, test } from "vitest";

const directories: string[] = [];
afterEach(() => {
	for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

function check(files: Record<string, string>) {
	const directory = mkdtempSync(join(tmpdir(), "guide-site-"));
	directories.push(directory);
	for (const [file, content] of Object.entries(files)) {
		const path = join(directory, file);
		mkdirSync(dirname(path), { recursive: true });
		writeFileSync(path, content);
	}
	return spawnSync("bun", [fileURLToPath(new URL("./check-site.ts", import.meta.url)), directory], {
		encoding: "utf8",
	});
}

test("resolves clean URLs, relative links and controls while reporting a missing anchor", () => {
	const result = check({
		"index.html": `<a href="/chapter?from=contents#one%20pass">chapter</a>
			<button popovertarget="contents" aria-controls="contents">contents</button>
			<nav id="contents"></nav><a href="/nested/">nested</a>`,
		"chapter.html": `<h1 id="one pass">one pass</h1><a href="/#main">start</a>`,
		"nested/index.html": `<a href="../chapter#one%20pass">chapter</a>`,
	});
	expect(result.status, result.stderr).toBe(1);
	expect(result.stderr).toContain('missing anchor "/#main"');
	expect(result.stderr.trim().split("\n")).toHaveLength(1);
});

test("accepts external URLs, local downloads and existing anchor references", () => {
	const result = check({
		"index.html": `<main id="main"></main><a href="#main">article</a><a href="/llms.txt">text</a>
			<a href="https://example.com/elsewhere">external</a><a href="mailto:reader@example.com">mail</a>
			<label for="read">read</label><input id="read" aria-describedby="hint unit">
			<span id="hint"></span><span id="unit"></span>`,
		"llms.txt": "guide",
	});
	expect(result.status, result.stderr).toBe(0);
	expect(result.stdout).toContain("Checked 1 built pages");
});

test("fails on broken paths, duplicate ids and disconnected control targets", () => {
	const result = check({
		"index.html": `<a href="/missing">missing</a><span id="same"></span><span id="same"></span>
			<button popovertarget="menu">contents</button><label for="field">value</label>`,
	});
	expect(result.status).toBe(1);
	expect(result.stderr).toContain('broken link "/missing"');
	expect(result.stderr).toContain('duplicate id "same"');
	expect(result.stderr).toContain('popovertarget="menu" has no target');
	expect(result.stderr).toContain('for="field" has no target');
});

test("fails if the build output is missing", () => {
	const result = check({});
	expect(result.status).toBe(1);
	expect(result.stderr).toContain("no HTML pages; run bun run build first");
});
