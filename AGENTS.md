# One token at a time

An unofficial illustrated guide for a programmer who uses AI but has never looked inside a model or
GPU. Astro + MDX, Bun and Biome; no UI framework. Read `PLAN.md` before changing the argument,
structure or chapter list.

## Commands and verification

- `bun run verify`: full check before committing; types, lint, tests, build and built-page checks.
- `bun run check`: faster types, lint and tests. `bun run build`: build and check links, IDs and targets.
- `bun run snapshot`: refresh public contest data; review the diff. Builds use the saved snapshot.
- Changed Triton examples: run the Linux interpreter and compiler checks in `README.md`.
- Visible changes: check the affected pages at 320px and desktop in both themes, open folds, exercise
  controls, check the console and reduced motion. Frames must keep their height and pages must not
  scroll sideways. The static checker does not verify browser behavior.
- Check `bunx astro dev status` before starting a server. Use `bunx astro dev --background` and stop
  only a server you started, with `bunx astro dev stop`.

## Where things live

- `src/pages/*.mdx`: chapters; `src/data/stops.ts`: their order, parts and questions.
- `src/components/`: one figure per file. Start from `TimeBars.astro` for static figures or
  `AttentionMix.astro` for live ones. `Figure` frames and numbers them; `KernelCode` renders code.
- `src/lib/model.ts`: model shapes, sourced GPU limits and cost calculations. Add missing constants
  here with sources; never duplicate them in a page or figure.
- `src/lib/challenge.ts` and `src/data/challenge.json`: official numbers and their public API snapshot.
- `src/lib/toy.ts`: the twelve-weight projection used across all three parts.
- `src/lib/format.ts`: format numbers with its helpers, including `us`, `bytes`, `times` and `worked`.
- `src/lib/figure.ts`: `FigureElement` and lifecycle helpers. Put a live figure's shared server/client
  calculations in a pure module in `src/lib/`; render its first state and redraw with the same module.
- `src/data/glossary.ts`: definitions for the glossary and `Term` cards.
- `src/styles/global.css`: shared objects, signals, `.fold`, `.meta` and `.page-head`.

## The argument and vocabulary

The guide zooms from a reply into the chip, then out to the server: enough arithmetic for about
94,000 tokens a second, but about 390 for one person. Each chapter answers one reader question and
ends on the next chapter's question (`PLAN.md`). Derive each idea once, then refer to it by name.

Use **weights** for fixed numbers shared by everyone, **working numbers** for per-pass activations,
and **cache** for each conversation's saved entries. Batching shares weights, never another person's
cache or working numbers. A **pass** runs the whole model; a **step** is a stage within a layer.
Say **compute blocks** for SMs outside code and contest asides, **program** for one running Triton
kernel copy, and **moving data** and **arithmetic** for the two costs.

## Writing and examples

- Open on the reader's question or the news. Relate facts with because, but or so; avoid recaps,
  itinerary cross-references and sentences that merely inventory facts.
- Describe a thing before naming it. Teach the small case before real sizes and introduce an idea
  before any example or aside relies on it. A token sees only itself and earlier tokens.
- Use “you” for the reader and “we” for working something out together; no author persona.
- Give each sentence one idea and each paragraph a role in the chapter's question. Cut stacked clauses,
  repeated derivations and caveats; keep every mechanism and qualification the argument needs.
- State scope beside the first relevant number: ideal lower bound or measured, busiest GPU or each
  GPU. Use values from `model.ts` or `challenge.json` through MDX, with absolute speeds as well as ratios.
  Keep a number in main text when the reader compares it or meets it again.
- Explain what is on the path from 94,000 to 390 or what an entrant's agent edits. For a side topic,
  prefer a sentence and an inline link. Folds must earn their place too.
- Wrap a recurring term in `Term` on its first use per page; its sentence already explains it.
  Use `Predict` before revealing an answer the reader can work out, especially a plausible wrong belief.
- Add prose around a figure only when it poses its question or explains its implication. Cut captions
  and instructions that repeat the drawing. No “watch the counter” or “this stop asks” scaffolding.
- End on the next question. A part's final chapter may first have a labelled “what you now have” box
  of at most three sentences. No end-of-page link lists or chapter summaries.
- Explain general techniques, including tiling, fusion and CUDA graphs. Contest links and short
  `InTheChallenge` asides belong where relevant; main text stands alone. Never describe entries or
  give a recipe for a specific contest problem.
- Kernels are short, real Triton. Count memory trips and show the launch grid before the body, naming
  one program per tile. Kernel 1 gets pseudocode, real lines, plain glosses and toy values; later
  kernels get one presentation beyond the figure. Keep the tile-height edit that changes work
  distribution without changing the answer. Check changed Triton calls against the official docs.

## Figures

- Prefer static HTML/SVG and side-by-side filmstrips. Use a custom element extending `FigureElement`
  only when motion or a control teaches something; register with `define()` and pass server values
  through `data-*`. A live figure has a meaningful state before it plays.
- Give each `Figure` a short `label`; folds are unnumbered and prose never refers to figure numbers.
- Reuse `.obj-weight`, `.obj-work`, `.obj-cache`, `.obj-op` and `.dim`, including their SVG equivalents.
- Two signals only: `--bytes` with squares for moving data, `--math` with dots for arithmetic. Use
  `Key` in text and neutral tokens for other states; never hard-code colors. The first signal figure
  shows one weight and one operation.
- Draw ideal data/arithmetic times from a common origin on a fixed labelled scale (`TimeBars`);
  the longer sets the floor. Never present their ratio as measured utilization.
- Compute worked sums from the rounded values actually displayed.
- Labels are lowercase plain-language fragments in `--mono` at 0.8125rem. Code names belong in code,
  detail folds and contest asides.
- Controls are labelled native inputs/buttons. A canvas has `role="img"`, an `aria-label` and visible
  text for essential state. Animate on input or once on visibility; show the final state for reduced
  motion. Keep frames stable at 320px and in both themes. Use existing dependencies.
