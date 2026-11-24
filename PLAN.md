# Design record

What the guide is for, why it is built the way it is, and how it was checked. `AGENTS.md` holds the
rules every edit follows; read this before changing the structure or adding a chapter.

## What the guide is for

A programmer who uses AI every day, and perhaps pointed an agent at the Open Kernel Challenge without
understanding what it edited, wants to know what actually happens when a model writes a reply: what the
model computes for each token, how a GPU computes it, and how a server shares that work among many
people. Write for that one person. The contest supplies a complete, public, measured system (model
shapes, SGLang's kernels, ideal times and measured baselines). It is the source of real numbers and is
mentioned where a chapter's idea is exactly what a problem timed; it is not the guide's skeleton, so the
guide stays useful after the contest ends.

## Why it is built this way

The first draft (October 2026) read as generated. The causes, and what replaced them:

- **Chapters were the contest's five problems,** so everything shared had to be taught on the opening
  page (14 terms, the layer map, the hardware) and pages opened with recaps. Chapters now answer the
  reader's questions, in three parts, and each term arrives where it is first needed.
- **The writing rules were almost all prohibitions,** so the prose became an inventory of facts joined
  by semicolons, with no questions and no "because" or "but". The rules are now positive, each with a
  before/after pair, plus measured concision rules.
- **The opening stated what the reader already knew** and buried the surprising fact 900 words in. It now
  opens on a puzzle in the reader's units: enough arithmetic for about 94,000 tokens a second, about 390
  for one person.
- **A rule forbade explaining techniques,** so a kernel stayed a black box. Part 2 now explains the GPU and
  reads three Triton kernels line by line; only recipes for specific contest problems are out.
- **No absolute speeds, and ideal never set beside measured.** Tokens per second appear throughout, and
  the contest's ideal, SGLang and best times close the GPU chapters.

## Decisions

- **Reader:** a programmer with no GPU or model-internals background.
- **Voice:** no author persona; "you" and "we".
- **Techniques:** explain how kernels and servers generally work (tiling, fusion, splitting work across
  compute blocks, overlapping data movement with arithmetic, CUDA graphs). Contest problems may be posed
  as open questions. No recipe for a specific problem, nothing about what entries did. Kernel 3 is
  therefore a simplified teaching kernel, not the contest's attention.
- **Running object:** the twelve-weight projection (`src/lib/toy.ts`) in all three parts: computed by
  hand (ch 1), as tiles by a kernel (ch 5), shared by a batch (ch 8).
- **Numbers:** every number comes from `model.ts` or `challenge.json`; each states its scope once
  (ideal or measured, the busiest GPU or each GPU).
- **Tooling:** no Tailwind, shadcn/ui or Astryx. The interface is prose, bespoke figures and about a
  dozen small controls; a React kit would add shipped JavaScript to replace them. Astro and the platform
  cover the rest: native popover with anchor positioning for glossary cards, `<details>` folds, CSS
  counters for figure and section numbers, container queries. Vercel's build runs `bun run verify`
  because Vercel deploys on every push regardless of CI.

## Structure

Zoom in from the reply to the chip, then back out to the server.

| #   | Part       | Slug                | Chapter asks                                       | Covers                                                                                                     |
| --- | ---------- | ------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 0   | Opening    | `/`                 | Where does the time go?                            | The puzzle, the reply as a loop of passes, 412 operations of time per weight and 2 used                    |
| 1   | The model  | `/pass`             | What does one pass compute?                        | Token to numbers, weighted sums, the twelve weights, layers, scores and sampling, the running total        |
| 2   | The model  | `/attention`        | How does the model remember the conversation?      | The waste without a cache, attention, heads, the cache, sliding windows                                    |
| 3   | The model  | `/experts`          | Why read only 42 of 816 billion weights?           | Experts and the router, active weights, one whole layer                                                    |
| 4   | The GPU    | `/chip`             | What is inside a GPU?                              | The memory ladder, why weights can't stay on chip, compute blocks, threads and warps, latency hiding       |
| 5   | The GPU    | `/kernel`           | What does a kernel do?                             | Programs and tiles, kernel 1, masks, coalescing, what sets its time, one edit that changes only speed      |
| 6   | The GPU    | `/layer`            | What is a pass to the GPU?                         | 18 kernels per layer, the CPU queue, CUDA graphs, fusion (kernel in a fold), eight GPUs and all-reduce     |
| 7   | The GPU    | `/attention-kernel` | How does the chip read a long conversation?        | Reading the cache, too little parallel work, splitting and recombining, the slice kernel                   |
| 8   | The server | `/batching`         | How can one server answer 32 people?               | Shared reads, the tipping point, caches never shared, more experts, the bill, memory capacity and queueing |
| 9   | The server | `/prefill`          | Why is there a pause before the first word?        | A prompt shares every read, the triangle, prompt caching, speculative decoding                             |
| 10  | The server | `/tradeoff`         | How much of the arithmetic does a server win back? | The ledger from 94,000 to 390, total against each, ideal against measured, a calculator                    |
| A   | Appendix   | `/challenge`        | What was the Open Kernel Challenge asking?         | What each problem asked, the layer map, how runs were judged, one brief line by line, About                |

The 404 page is a small game, "Out of memory": chapter 8's capacity trade-off, played.

Each chapter ends on the question the next one answers:

- 0 → 1: what is the chip doing with the weights it reads?
- 1 → 2: does every pass have to reread the earlier tokens?
- 2 → 3: 816 billion weights, but a token reads 42 billion: which ones, and who decides?
- 3 → 4: every pass moves about 12 GB on the busiest GPU: where is it kept, and why not on the chip?
- 4 → 5: what program moves it through the chip?
- 5 → 6: a layer is 18 kernels in a row: what does each handoff cost?
- 6 → 7: attention's mixing step reads no weights: what does it read, and how does it keep the chip busy?
- 7 → 8: with one person most of the chip waits: what if a second person's token rides along?
- 8 → 9: why can the prompt be read so much faster than the reply is written?
- 9 → 10: who gets the speed, each person or the server?

## Kernels

Triton, because a Triton kernel is written per tile, which matches the drawings, and SGLang's baseline
files use it. Present each one this way (from research on Boehm, the Triton tutorials and Horace He):
say that one copy runs per program and works on a whole tile; show the launch grid before the body; give
the trips to memory before the code; pseudocode first, then real lines with a plain gloss and the toy's
values; mark moving data and arithmetic with the two signals; show later kernels as changes to earlier
ones. Head off: a thread is not a core; peak arithmetic doesn't set the time; fixed launch costs don't
shrink with the work; one program per head leaves most of the chip idle; programs run in no fixed order.

No GPU was available. The arithmetic the kernels imply is computed in `src/lib/kernels.ts` and tested.
In November 2026 the four kernels, extracted exactly as the pages show them, were run under Triton
3.9's CPU interpreter (`TRITON_INTERPRET=1`) against float64 references at the toy size, the real size
and ragged sizes, and compiled for sm_90 down to a cubin. All passed, including the tile-height edit
giving identical outputs and the slice kernel's empty slices (blend NaN, number −∞) through a
test-only recombination. They have never been timed on a GPU.

The harness is preserved in `scripts/extract-kernels.ts` and `scripts/verify-kernels.py`;
the Linux setup and commands are in `README.md`. Extraction uses the current MDX
code arrays and model constants, so future runs check the displayed examples.

## Design

A technical manual, quiet and exact, with Making Software as inspiration, not a template: one serif
(Source Serif 4) and one pixel mono (Departure Mono); a sticky contents rail grouped by part and a
scroll-progress ruler on desktop, a slim bar with a contents popover on phones; numbered figures
("fig 5.2") on a faint dot grid, wider than the text; mono section numbers; one quiet style for the
three folds (? predict, + deeper, ◆ challenge); a typographic end mark and a next-chapter link that
shows the next question. The only colours are the two signals. Not borrowed: its accent colour,
justified long lines, commercial serif, unreadable phone figures.

## How it was checked

- Seven fresh-perspective Codex reviews: the plan, each part, Part 2's technical accuracy, a cold read
  of the whole guide in order, and the codebase. Every finding was fixed.
- The headline numbers (94,000; 393; 2.55 ms; 412; 206; 139 and 4,434; 12 GB; 256 bytes; 20.5 MB;
  4.5 bits; 10.6 µs; 1.54) were recomputed from NVIDIA's published rates and the contest snapshot,
  independently of `model.ts`, and matched against the built pages.
- A Chrome DevTools sweep of every page at 320, 390 and 1440px in both themes: no horizontal overflow
  with folds open or closed, no console errors, no figure changing size while it animates.
- Every internal link and anchor resolves; every outside source returns 200.
- **November 2026 technical, interactive & editorial revision:** Dual-perspective audit (GPU/inference
  architect + novice OKC reader), followed by an adversarial GPT-6.2 Sol audit. Scoped 94,000 tok/s
  as short-context weight arithmetic and added 80k-context attention FLOPs (~61,000 tok/s ceiling);
  added FP32 CUDA-core (~67 TFLOPS, ~14-token ridge) vs. BF16 tensor-core (989 TFLOPS, ~206-token
  ridge) toggle in `TippingPoint` and `tl.dot` batched GEMM fold; updated `SplitContext` and Ch. 7
  to real GQA KV-head grouping (2 KV heads × 64 slices = 128 programs); replaced Ch. 5's static tile
  edit with interactive `TileEdit`; upgraded `<Term>` hover cards with external resource links and
  `/glossary` with instant search and category filters; added motivation/history beats, Ch. 3's
  8-GPU bridge, Ch. 10's pocket equations, and Appendix A's diff Rosetta Stone; standardized American
  English and ragged-right desktop body typography.
- **Council & Codex cold-read polish:** Two-model council review (Opus 5.5 + GPT-6.1 Sol) followed by
  a full voice restoration and a fresh cold-read audit via Codex (`google/gemini-4-neon`, high
  reasoning). Removed AI-cadence crutches across all chapters, fixed attribution and ratio-wording
  slips, aligned speculative-decode verification wording with `DraftCheck`, eliminated early jargon
  forward-references, and verified every figure, link, and responsive layout across both themes.

Mistakes the reviews caught, worth avoiding again: a number quoted with the wrong scope (the busiest GPU
called "each GPU"; an ideal read as measured); "never shared" said of expert weights, which tokens that
choose the same expert do share; a break-even that left out a cost (drafting time); a kernel that only
handled the first tile of tokens; a contest-shaped kernel that came close to a recipe; and a test that
passed by construction.
