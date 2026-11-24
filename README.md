# One token at a time

An unofficial illustrated guide to what happens when an AI model writes a reply: what the model computes for each token, how a GPU executes that work (down to short Triton kernels read line by line), and how a server shares it among many people. Written for programmers who use AI every day and have never looked inside a model or a GPU.

**[Read the guide at tokens.arihantbansal.com](https://tokens.arihantbansal.com)**

Eight NVIDIA H200 GPUs have enough tensor-core arithmetic to write about 94,000 tokens a second from weight multiplications alone in a short conversation, yet for one person in an 80,000-token conversation the hardware's ideal ceiling drops to about 390. Every chapter answers one question along that path, grounding all shapes and timings in public data from Meta's Muse Spark Open and the [Open Kernel Challenge](https://openkernelchallenge.dev/).

This is an independent guide, not affiliated with Meta, Paradigm, or Andromeda. It explains general mechanisms (tiling, coalescing, fusion, CUDA graphs, split attention, continuous batching, prompt caching, and speculative decoding) and says nothing about private contest submissions or problem-specific recipes.

## What is inside

The guide zooms from a reply into the chip, then back out to the server. Every chapter ends on the question the next one answers.

| #   | Part       | Chapter                | Question                                           |
| --- | ---------- | ---------------------- | -------------------------------------------------- |
| 0   | Opening    | One token at a time    | Where does the time go?                            |
| 1   | The model  | One pass               | What does one pass compute?                        |
| 2   | The model  | Remembering            | How does the model remember the conversation?      |
| 3   | The model  | Experts                | Why read only 42 of 816 billion weights?           |
| 4   | The GPU    | Inside the chip        | What is inside a GPU?                              |
| 5   | The GPU    | A kernel               | What does a kernel do?                             |
| 6   | The GPU    | A chain of kernels     | What is a pass to the GPU?                         |
| 7   | The GPU    | Reading the cache      | How does the chip read a long conversation?        |
| 8   | The server | Many people            | How can one server answer 32 people?               |
| 9   | The server | Reading the prompt     | Why is there a pause before the first word?        |
| 10  | The server | Where the speed went   | How much of the arithmetic does a server win back? |
| A   | Appendix   | The challenge, decoded | What was the Open Kernel Challenge asking?         |

Also included:

- **Inline term cards and a searchable glossary (`/glossary`)** with category filters, chapter references, and primary outside sources.
- **Interactive figures** built as lightweight Web Components (tile-height scheduling in `TileEdit`, BF16 tensor-core vs. FP32 CUDA-core rooflines in `TippingPoint`, KV-cache routing and capacity, a personal-machine speed calculator, and an interactive layer map).
- **Machine-readable index (`/llms.txt`)** and a small KV-cache memory-packing game on `/404`.

## Run locally

Install the Node version in [`.node-version`](.node-version) and the Bun version in [`package.json`](package.json), then run:

```sh
bun install --frozen-lockfile
bun run dev
```

Open the URL printed in the terminal (usually `http://localhost:4321`).

| Command            | Purpose                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------- |
| `bun run dev`      | Start the local Astro dev server.                                                        |
| `bun run open`     | Start or reuse a background dev server and open it in the browser.                       |
| `bun run close`    | Stop the background dev server.                                                          |
| `bun run check`    | Run Astro type checks, Biome lint/format checks, and Vitest unit tests.                  |
| `bun run build`    | Build the static site into `dist/` and verify all local links, IDs, and control targets. |
| `bun run verify`   | Run `check` and `build` together, matching CI and Vercel.                                |
| `bun run preview`  | Serve the built `dist/` output locally.                                                  |
| `bun run snapshot` | Refresh `src/data/challenge.json` from the public Open Kernel Challenge API.             |

## Project layout

Built with static Astro and MDX, styled with plain CSS, and zero UI framework overhead.

- [`src/pages/`](src/pages/): one MDX file per chapter, plus [`glossary.astro`](src/pages/glossary.astro), [`404.astro`](src/pages/404.astro), and [`llms.txt.ts`](src/pages/llms.txt.ts).
- [`src/data/`](src/data/): chapter order and questions ([`stops.ts`](src/data/stops.ts)), glossary definitions and sources ([`glossary.ts`](src/data/glossary.ts)), and the saved contest snapshot ([`challenge.json`](src/data/challenge.json)).
- [`src/components/`](src/components/): bespoke static HTML/SVG figures and interactive custom elements extending `FigureElement`.
- [`src/lib/`](src/lib/): single source of truth for model shapes, sourced H200 limits, and cost calculations ([`model.ts`](src/lib/model.ts)), kernel grids and worked examples ([`kernels.ts`](src/lib/kernels.ts)), serving math ([`serving.ts`](src/lib/serving.ts)), the shared twelve-weight projection ([`toy.ts`](src/lib/toy.ts)), and number formatting ([`format.ts`](src/lib/format.ts)).
- [`scripts/`](scripts/): static build validator ([`check-site.ts`](scripts/check-site.ts)), API snapshot refresher ([`snapshot.ts`](scripts/snapshot.ts)), and the Triton kernel extraction and verification harness ([`extract-kernels.ts`](scripts/extract-kernels.ts), [`verify-kernels.py`](scripts/verify-kernels.py)).

## Working on the guide

Follow [`AGENTS.md`](AGENTS.md) for prose, vocabulary, and figure conventions, and [`PLAN.md`](PLAN.md) for the guide's design record and chapter structure. Run `bun run verify` before committing; [`.github/workflows/check.yml`](.github/workflows/check.yml) runs the same command on pull requests and pushes to `main`.

For visible changes, check affected pages at 320px and desktop widths in both light and dark themes, open the contents menu and folds, exercise interactive controls, and confirm zero console errors, zero horizontal overflow, and stable figure heights (including with reduced motion enabled).

### Verify the teaching kernels

The verification harness extracts all four displayed Triton kernels directly from the MDX code arrays and reads sizes from `src/lib/`. Run it whenever any kernel example changes (requires Linux, Bun, and Python 3.12; no GPU needed):

```sh
uv venv --python 3.12 .venv-kernels
uv pip install --python .venv-kernels/bin/python -r scripts/requirements-kernels.txt
bun scripts/extract-kernels.ts &&
  .venv-kernels/bin/python scripts/verify-kernels.py interpret &&
  .venv-kernels/bin/python scripts/verify-kernels.py compile
```

`interpret` runs the toy, real, and ragged projections, the tile-height edit, fused gated norm, and split attention (including empty slices) under Triton's CPU interpreter against float64 references. `compile` compiles each kernel for `sm_90` down to a cubin. Generated files live in `.astro/kernel-check/`.

### Refresh contest data

```sh
bun run snapshot
```

Review the diff before committing. The script validates API fields and checks the shape table against benchmark weights before replacing [`src/data/challenge.json`](src/data/challenge.json); failed requests leave the saved file untouched. Production builds always use the committed snapshot and never contact the API at build time.

## Deployment

The site deploys as static files on Vercel with no server adapter or environment variables. [`vercel.json`](vercel.json) pins the install (`bun install --frozen-lockfile`), build (`bun run verify`), output (`dist`), and `cleanUrls: true` so generated HTML files such as `kernel.html` serve cleanly at `/kernel`.

To deploy:

1. Import the repository into Vercel (root directory `.`, production branch `main`). Branch pushes automatically create preview deployments.
2. Add your custom domain in Vercel's Domains settings (`tokens.arihantbansal.com`, matching `site` in [`astro.config.mjs`](astro.config.mjs)) and configure the CNAME record Vercel provides at your DNS host.

## Credits

- **Departure Mono** by Helena Zhang, under the [SIL Open Font License](public/fonts/DepartureMono-LICENSE.txt).
- **Source Serif 4** by Adobe, under the SIL Open Font License.
- Public benchmark specifications and timings from the [Open Kernel Challenge](https://openkernelchallenge.dev/).
