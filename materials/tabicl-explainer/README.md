# TabICL Explainer

Interactive visualization of TabICLv2 inference on fixed UCI Iris examples.

The 12-row context is intentionally compact for tracing. It is outside the
officially documented TabICLv2 pretraining range of 300 to 48K rows, and the
upstream authors state that sub-300-row generalization has not been tested.
Treat its measured output as an out-of-regime illustration, not a quality claim.
The interface adapts the MIT-licensed
[Transformer Explainer](https://github.com/poloclub/transformer-explainer)
layout while replacing GPT-2 generation with tabular in-context learning.

## Requirements

- Node.js 20 or newer
- npm 10 or newer
- Sibling browser runtime at `../website/js/tabicl/nanotabicl.js`
- Sibling model assets at `../website/model/`

## Run locally

```bash
npm ci
npm run dev
```

Open http://localhost:5173/. Vite serves the sibling checkpoint at `/model/`
during development and preview.

## Build

```bash
npm run build
```

The static adapter writes output to `../website/tabicl-explainer/`. That generated
folder is ignored by Git and rebuilt by the Pages workflow.

## Verification

```bash
npm run check
npm run build
```

## Attribution and scope

The interface is adapted from *[Transformer Explainer: Learning LLM
Transformers with Interactive Visual Explanation and
Experimentation](https://doi.org/10.1145/3772318.3791725)* by Aeree Cho,
Grace C. Kim, Alexander Karpekov, Seongmin Lee, Alec Helbling, Benjamin
Hoover, Zijie J. Wang, Minsuk Kahng, and Duen Horng (Polo) Chau (CHI 2026).
The original interface is used under the MIT License reproduced in
[`LICENSE`](LICENSE).

This adaptation preserves the interface composition while replacing GPT
inference and examples with fixed UCI Iris records and a browser TabICLv2
checkpoint. The explorer uses a nanoTabICL-derived bridge for selected-view
inspection with the released TabICLv2 checkpoint's feature-group offsets. It is
not the standalone nanoTabICL model and does not expose every preprocessing and
ensemble option in the official `TabICLClassifier`.

Checkpoint source, hashes, runtime behavior, and limitations are documented in
[`../website/model/PROVENANCE.md`](../website/model/PROVENANCE.md). TabICL code
and checkpoint terms remain governed by their upstream sources.
