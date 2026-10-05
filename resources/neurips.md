# NeurIPS resources for tabular foundation models

[Tutorial home](../README.md) · [Interactive guide](https://affirm.github.io/tabular-foundation-models-tutorial/) · [FMSD workshop resources](fmsd.md)

Start with our interactive guide, then use these selected NeurIPS resources to study pretraining, text features, and evaluation. Entries are grouped by conference year. Checked October 4, 2026.

## Our guide at NeurIPS 2026

**Table as Prompt: An Interactive Guide to Tabular Foundation Models** is accepted for presentation at the NeurIPS 2026 Education Track.

The [interactive guide](https://affirm.github.io/tabular-foundation-models-tutorial/) introduces task-specific fitting, adaptation through labeled examples, and PFN theory. It then follows TabICLv2 through training and inference, with a browser playground and model inspection. Basic supervised learning is enough to start. Use the [Python notebook](../materials/notebooks/01_tabicl_primer.ipynb) to work through prediction in code.

The [Education Track website](https://neurips-education-track.github.io/) collects track information. The [official call](https://neurips.cc/Conferences/2026/CallforEducationalResources) describes the track's educational scope and formats.

## Selected NeurIPS 2025 models

| Model | What to study | Paper and implementation |
| --- | --- | --- |
| TabDPT | Pretraining on real tables and retrieval for inference context. | [Paper](https://www.cs.toronto.edu/~mvolkovs/NeurIPS2025_TabDPT.pdf) · [GitHub](https://github.com/layer6ai-labs/TabDPT-inference) · [Weights](https://huggingface.co/Layer6/TabDPT) |
| Mitra | How a mixture of synthetic priors shapes classification and regression. | [Paper](https://arxiv.org/abs/2510.21204) · [AutoGluon](https://github.com/autogluon/autogluon) · [Classifier](https://huggingface.co/autogluon/mitra-classifier) · [Regressor](https://huggingface.co/autogluon/mitra-regressor) |
| ConTextTab | Semantic embeddings and real-table pretraining for tabular ICL. | [Paper](https://arxiv.org/abs/2506.10707) · [GitHub](https://github.com/SAP-samples/sap-rpt-1-oss) · [Weights](https://huggingface.co/SAP/sap-rpt-1-oss) |
| TabSTAR | Target-aware representations for tables containing text fields. | [Paper](https://arxiv.org/abs/2505.18125) · [GitHub](https://github.com/alanarazi7/TabSTAR) · [Weights](https://huggingface.co/alana89/TabSTAR) |

Compare the pretraining data, use of feature semantics, and adaptation procedure. TabSTAR studies transfer learning; the models here do not all use the same inference-only adaptation procedure.

## Evaluation

[TabArena](https://arxiv.org/abs/2506.16791), a NeurIPS 2025 Datasets and Benchmarks spotlight, provides a maintained evaluation system. Read its validation and ensembling protocol before comparing results. Start with the [code](https://github.com/autogluon/tabarena), [datasets](https://github.com/tabarena/data-foundry), and [leaderboard](https://tabarena.ai/).

For additional conference papers, search the [NeurIPS proceedings](https://proceedings.neurips.cc/). The repository's [model landscape](../README.md#model-landscape) covers models across venues and distinguishes paper dates from model releases.
