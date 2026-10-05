# TabICL Explainer

Interactive visualization of TabICLv2 inference on selected
[UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) records.

Dataset citation: Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning
Repository. [DOI: 10.24432/C56C76](https://doi.org/10.24432/C56C76).
The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Measurements remain in centimeters; species names are encoded as numeric
context labels, and query labels are withheld.

The 12-row context is intentionally compact for tracing. It is outside the
officially documented TabICLv2 pretraining range of 300 to 48K rows. The revised
paper includes sub-300-row few-shot evaluations
([Appendix L.3–L.4, September 2026 revision](https://arxiv.org/html/2602.11139v2#A12.SS3)).
Those results do not establish the quality of this particular 12-row Iris demo.
Treat its measured output as an illustration of computation, not a quality claim.
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
checkpoint. The explorer uses a nanoTabICL-derived bridge for fixed single-pass
core inspection with the released TabICLv2 checkpoint's feature-group offsets. It is
not the standalone nanoTabICL model. It uses context-only z-score standardization,
original feature and class order, and softmax temperature 1. It is not a selected
view of the main playground's eight-view `TabICLClassifier`, which applies its own
preprocessing, permutations, and temperature 0.9.

Checkpoint source, hashes, runtime behavior, and limitations are documented in
[`../website/model/PROVENANCE.md`](../website/model/PROVENANCE.md). TabICL code
and checkpoint terms remain governed by their upstream sources.

The deployed site includes [third-party notices](../website/THIRD_PARTY_NOTICES.txt)
and [full explorer dependency licenses](../website/licenses/EXPLORER-DEPENDENCIES.txt).
