import { resolve } from "node:path";

type Page = { ids: Set<string>; links: string[]; targets: { attribute: string; id: string }[] };

async function checkSite(directory: string) {
	const files = new Set(new Bun.Glob("**/*").scanSync({ cwd: directory, onlyFiles: true }));
	const pages = new Map<string, Page>();
	const errors: string[] = [];
	for (const file of [...files].filter((f) => f.endsWith(".html")).sort()) {
		const page: Page = { ids: new Set(), links: [], targets: [] };
		new HTMLRewriter()
			.on("[id]", {
				element(el) {
					const id = el.getAttribute("id") ?? "";
					if (page.ids.has(id)) errors.push(`${file}: duplicate id "${id}"; give each target a unique id`);
					page.ids.add(id);
				},
			})
			.on("a[href]", {
				element(el) {
					page.links.push(el.getAttribute("href") ?? "");
				},
			})
			.on("[popovertarget], [aria-controls], [aria-labelledby], [aria-describedby], label[for]", {
				element(el) {
					for (const attribute of [
						"popovertarget",
						"aria-controls",
						"aria-labelledby",
						"aria-describedby",
						"for",
					]) {
						const value = el.getAttribute(attribute);
						if (value) for (const id of value.split(/\s+/)) page.targets.push({ attribute, id });
					}
				},
			})
			.transform(await Bun.file(resolve(directory, file)).text());
		pages.set(file, page);
	}
	if (pages.size === 0) throw new Error(`${directory}: no HTML pages; run bun run build first`);

	const origin = "https://guide.invalid";
	for (const [file, page] of pages) {
		for (const { attribute, id } of page.targets) {
			if (!page.ids.has(id))
				errors.push(`${file}: ${attribute}="${id}" has no target; add or correct its id`);
		}
		for (const href of page.links) {
			let url: URL;
			let path: string;
			let fragment: string;
			try {
				url = new URL(href, `${origin}/${file}`);
				if (url.origin !== origin) continue;
				path = decodeURIComponent(url.pathname).slice(1);
				fragment = decodeURIComponent(url.hash.slice(1));
			} catch {
				errors.push(`${file}: invalid link "${href}"; correct its URL`);
				continue;
			}
			const target = [path, `${path}.html`, `${path.replace(/\/$/, "")}${path ? "/" : ""}index.html`].find(
				(candidate) => files.has(candidate),
			);
			if (!target) errors.push(`${file}: broken link "${href}"; point it at a built page or file`);
			else if (fragment && pages.has(target) && !pages.get(target)?.ids.has(fragment)) {
				errors.push(`${file}: missing anchor "${href}"; add or correct the target id in ${target}`);
			}
		}
	}
	return { pages: pages.size, errors };
}

const result = await checkSite(resolve(process.argv[2] ?? "dist"));
if (result.errors.length) {
	console.error(result.errors.join("\n"));
	process.exitCode = 1;
} else {
	console.log(`Checked ${result.pages} built pages: local links, unique ids and control targets resolve.`);
}
