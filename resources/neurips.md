# NeurIPS 2026 resources for tabular foundation models

[Tutorial home](../README.md) · [Interactive guide](https://affirm.github.io/tabular-foundation-models-tutorial/) · [FMSD workshop resources](fmsd.md)

Start with our interactive guide, then explore selected NeurIPS 2026 work on tabular models, broader prediction tasks, and evaluation. The papers below appear in the [official 2026 conference directory](https://neurips.cc/Downloads/2026). Earlier work is collected separately at the end. Checked October 4, 2026.

## Our guide at NeurIPS 2026

**Table as Prompt: An Interactive Guide to Tabular Foundation Models** is accepted for presentation at the NeurIPS 2026 Education Track.

The [interactive guide](https://affirm.github.io/tabular-foundation-models-tutorial/) introduces task-specific fitting, adaptation through labeled examples, and PFN theory. It then follows TabICLv2 through training and inference, with a browser playground and model inspection. Basic supervised learning is enough to start. Use the [Python notebook](../materials/notebooks/01_tabicl_primer.ipynb) to work through prediction in code.

The [Education Track website](https://neurips-education-track.github.io/) collects track information. The [official call](https://neurips.cc/Conferences/2026/CallforEducationalResources) describes the track's educational scope and formats.

## NeurIPS 2026 models and methods

| Paper | What to study | Sources and resources |
| --- | --- | --- |
| FlexTab | A shared encoder produces target-agnostic row representations; task-specific decoders support classification, regression, clustering, and other tabular tasks. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/153378) · [Paper](https://arxiv.org/abs/2606.30336) |
| SurvivalPFN | PFN pretraining for time-to-event prediction with right-censored observations. Study how the prior and prediction target change for survival analysis. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/151336) · [Paper](https://arxiv.org/abs/2605.15488) · [GitHub](https://github.com/rgklab/SurvivalPFN) · [Weights](https://huggingface.co/shi-ang/SurvivalPFN) |

FlexTab's paper announces code and checkpoints, but its linked repository was unavailable when checked. Use the paper for now.

## NeurIPS 2026 benchmarks and evaluation

| Paper | What to study | Sources and resources |
| --- | --- | --- |
| Beyond IID / BeyondArena | Evaluate generalization across IID, temporal, and grouped tasks, different table sizes, and varied feature types. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139490) · [Paper](https://arxiv.org/abs/2606.30410) · [TabArena code](https://github.com/autogluon/tabarena) · [Data Foundry](https://github.com/tabarena/data-foundry) |
| MulTaBench | Compare models on tables that combine structured features with text or images. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/138893) · [Paper](https://arxiv.org/abs/2605.10616) · [Code and datasets](https://github.com/alanarazi7/MulTaBench) |
| TabPrep | Study how feature engineering changes comparisons among tree-based, neural, linear, and foundation models. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139065) · [Paper](https://arxiv.org/abs/2606.02384) · [GitHub](https://github.com/atschalz/tabprep) |
| Benchmarking Attention for Tabular Foundation Models | Compare attention backends for row and column operations across table shapes and GPU hardware. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139839) · [Paper](https://arxiv.org/abs/2609.31306) · [Code and results](https://github.com/SAP-samples/tabular-attention-benchmark) |

These papers address different questions: generalization, multimodal inputs, preprocessing, and computational cost. Read their splits, tuning budgets, and hardware settings before comparing results. Strong task-specific baselines remain essential.

## Earlier NeurIPS work: 2025

| Model | What to study | Paper and implementation |
| --- | --- | --- |
| TabDPT | Pretraining on real tables and retrieval for inference context. | [Paper](https://www.cs.toronto.edu/~mvolkovs/NeurIPS2025_TabDPT.pdf) · [GitHub](https://github.com/layer6ai-labs/TabDPT-inference) · [Weights](https://huggingface.co/Layer6/TabDPT) |
| Mitra | How a mixture of synthetic priors shapes classification and regression. | [Paper](https://arxiv.org/abs/2510.21204) · [AutoGluon](https://github.com/autogluon/autogluon) · [Classifier](https://huggingface.co/autogluon/mitra-classifier) · [Regressor](https://huggingface.co/autogluon/mitra-regressor) |
| ConTextTab | Semantic embeddings and real-table pretraining for tabular ICL. | [Paper](https://arxiv.org/abs/2506.10707) · [GitHub](https://github.com/SAP-samples/sap-rpt-1-oss) · [Weights](https://huggingface.co/SAP/sap-rpt-1-oss) |
| TabSTAR | Target-aware representations for tables containing text fields. | [Paper](https://arxiv.org/abs/2505.18125) · [GitHub](https://github.com/alanarazi7/TabSTAR) · [Weights](https://huggingface.co/alana89/TabSTAR) |

Compare the pretraining data, use of feature semantics, and adaptation procedure. TabSTAR studies transfer learning; the models here do not all use the same inference-only adaptation procedure.

### TabArena

[TabArena](https://arxiv.org/abs/2506.16791), a NeurIPS 2025 Datasets and Benchmarks spotlight, provides a maintained evaluation system. Read its validation and ensembling protocol before comparing results. Start with the [code](https://github.com/autogluon/tabarena), [datasets](https://github.com/tabarena/data-foundry), and [leaderboard](https://tabarena.ai/).

For additional conference papers, search the [NeurIPS proceedings](https://proceedings.neurips.cc/). The repository's [model landscape](../README.md#model-landscape) covers models across venues and distinguishes paper dates from model releases.
