# Teaching materials

This archive contains the interactive tutorial and a self-guided notebook for
*Table as Prompt: An Interactive Guide to Tabular Foundation Models*.

## Interactive tutorial

Requirements: Python 3 and a modern browser.

```bash
cd website
./serve.sh 8000
```

Open <http://localhost:8000/>. The roughly 28 MB TabICLv2 checkpoint loads in
the browser; inference is local and requires no GPU or remote service.

## Self-guided notebook

### Google Colab

Open the
[primer in Colab](https://colab.research.google.com/github/Affirm/tabular-foundation-models-tutorial/blob/main/materials/notebooks/01_tabicl_primer.ipynb)
and run all cells. A GPU runtime is recommended but not required. The first cell
installs the pinned tutorial packages while retaining Colab's managed Python and
PyTorch environment. TabICL automatically selects CUDA in Colab and CUDA, XPU,
or Apple MPS on supported local hardware, with CPU as the fallback.

### Local or CI

Use Python 3.11 in a fresh virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
jupyter lab notebooks/01_tabicl_primer.ipynb
```

`requirements.txt` lists the direct dependencies; `requirements-lock.txt`
fully pins transitive packages and distribution hashes for Python 3.11. This is
the exact reproducibility path; Colab is the convenience path. The
notebook verifies its immutable Hugging Face revision and checkpoint checksum
before model loading. An internet connection is needed for package, dataset,
and checkpoint downloads.

## Source and attribution

- Public tutorial: <https://affirm.github.io/tabular-foundation-models-tutorial/>
- Source repository: <https://github.com/Affirm/tabular-foundation-models-tutorial>
- Browser checkpoint details: [`website/model/PROVENANCE.md`](website/model/PROVENANCE.md)
- Adapted explorer details: [`tabicl-explainer/README.md`](tabicl-explainer/README.md)

## Licensing

Unless otherwise noted, original software and code in this standalone materials package are
licensed under the [Apache License, Version 2.0](LICENSE), and original educational content is
licensed under [Creative Commons Attribution 4.0 International](LICENSE-CONTENT). Copyright
(c) 2026, Affirm, Inc. All rights reserved. See [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) for project and third-party terms, including those
for the browser runtime, model artifact, adapted explorer, and datasets.
