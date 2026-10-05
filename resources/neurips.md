# NeurIPS 2026 resources for tabular foundation models

[Tutorial home](../README.md) · [Interactive guide](https://affirm.github.io/tabular-foundation-models-tutorial/) · [Official paper directory](https://neurips.cc/Downloads/2026)

Selected NeurIPS 2026 papers on tabular models, prediction tasks, and evaluation, grouped by research topic. Paper labels are shortened for scanning; conference records provide the full titles. Checked October 5, 2026.

**Browse:** [Our guide](#our-guide) · [Models and prediction tasks](#models-and-prediction-tasks) · [Benchmarks and efficiency](#benchmarks-and-efficiency) · [Related research](#related-research)

## Our guide

**Table as Prompt: An Interactive Guide to Tabular Foundation Models** is accepted for presentation at the NeurIPS 2026 Education Track.

The [interactive guide](https://affirm.github.io/tabular-foundation-models-tutorial/) introduces task-specific fitting, adaptation through labeled examples, and PFN theory. It then follows TabICLv2 through training and inference, with a browser playground and model inspection. Basic supervised learning is enough to start. Use the [Python notebook](../materials/notebooks/01_tabicl_primer.ipynb) to work through prediction in code.

The [Education Track website](https://neurips-education-track.github.io/) collects track information. The [official call](https://neurips.cc/Conferences/2026/CallforEducationalResources) describes the track's educational scope and formats.

## Models and prediction tasks

| Paper | Focus | Links |
| --- | --- | --- |
| FlexTab | A shared encoder produces target-agnostic row representations; task-specific decoders support classification, regression, clustering, and other tabular tasks. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/153378) · [Paper](https://arxiv.org/abs/2606.30336) |
| Semantic feature embeddings | Feature meaning and context | [NeurIPS record](https://neurips.cc/virtual/2026/poster/149526) |
| TabClustPFN | Infer cluster assignments and the number of clusters using a PFN trained on synthetic clustering tasks. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/150101) · [Paper](https://arxiv.org/abs/2601.21656) · [Code and checkpoint access](https://github.com/Tianqi-Zhao/TabClustPFN) |
| TabK | Cluster-count inference | [NeurIPS record](https://neurips.cc/virtual/2026/poster/148754) |
| SurvivalPFN | PFN pretraining for time-to-event prediction with right-censored observations. Study how the prior and prediction target change for survival analysis. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/151336) · [Paper](https://arxiv.org/abs/2605.15488) · [GitHub](https://github.com/rgklab/SurvivalPFN) · [Weights](https://huggingface.co/shi-ang/SurvivalPFN) |
| DynaPFN | Dynamical forecasting | [NeurIPS record](https://neurips.cc/virtual/2026/poster/149937) |
| CausalTab | Causal discovery | [NeurIPS record](https://neurips.cc/virtual/2026/poster/150884) |
| Causal ordering for prediction | Causal structure in ICL | [NeurIPS record](https://neurips.cc/virtual/2026/poster/153895) |

## Benchmarks and efficiency

| Paper | Focus | Links |
| --- | --- | --- |
| Beyond IID / BeyondArena | Evaluate generalization across IID, temporal, and grouped tasks, different table sizes, and varied feature types. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139490) · [Paper](https://arxiv.org/abs/2606.30410) · [TabArena code](https://github.com/autogluon/tabarena) · [Data Foundry](https://github.com/tabarena/data-foundry) |
| TRL-Bench | Compare row, column, and table representations under shared downstream evaluation conditions. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139385) · [Paper](https://arxiv.org/abs/2606.09323) · [Code and datasets](https://github.com/LOGO-CUHKSZ/TRL-Bench) |
| MulTaBench | Compare models on tables that combine structured features with text or images. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/138893) · [Paper](https://arxiv.org/abs/2605.10616) · [Code and datasets](https://github.com/alanarazi7/MulTaBench) |
| STRABLE | String-valued features | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139430) |
| TabBioMed | Biomedical tables | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139791) |
| Genetic-data limitations | Failure cases in genetics | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139799) |
| TabPrep | Study how feature engineering changes comparisons among tree-based, neural, linear, and foundation models. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139065) · [Paper](https://arxiv.org/abs/2606.02384) · [GitHub](https://github.com/atschalz/tabprep) |
| Benchmarking Attention for Tabular Foundation Models | Compare attention backends for row and column operations across table shapes and GPU hardware. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139839) · [Paper](https://arxiv.org/abs/2609.31306) · [Code and results](https://github.com/SAP-samples/tabular-attention-benchmark) |

Read the evaluation splits, tuning budgets, and hardware settings before comparing results. Strong task-specific baselines remain essential.

## Related research

### Relational data

| Paper | Focus | Links |
| --- | --- | --- |
| Few-shot relational pretraining | Prediction across linked tables | [NeurIPS record](https://neurips.cc/virtual/2026/poster/149169) |
| RelAgent | Agent-driven relational analysis | [NeurIPS record](https://neurips.cc/virtual/2026/poster/151742) |

### Generative modeling

| Paper | Focus | Links |
| --- | --- | --- |
| TabDLM | Numeric and text generation | [NeurIPS record](https://neurips.cc/virtual/2026/poster/155779) |
| TabWorld | World models for tables | [NeurIPS record](https://neurips.cc/virtual/2026/poster/150304) |
| Generative TFM | Generative modeling | [NeurIPS record](https://neurips.cc/virtual/2026/poster/149211) |
| Quality and privacy in generation | ICL for synthetic tables | [NeurIPS record](https://neurips.cc/virtual/2026/poster/152928) |

### Robustness, privacy, and theory

| Paper | Focus | Links |
| --- | --- | --- |
| RAD-TFM | Domain adaptation | [NeurIPS record](https://neurips.cc/virtual/2026/poster/155295) |
| Harmful-shift detection | PFN reliability under shift | [NeurIPS record](https://neurips.cc/virtual/2026/poster/150327) |
| Embedding privacy audit | Information leakage | [NeurIPS record](https://neurips.cc/virtual/2026/poster/149397) |
| TFM generalization theory | Generalization behavior | [NeurIPS record](https://neurips.cc/virtual/2026/poster/150354) |

## Related venues

[Foundation Models for Structured Data (FMSD), ICML 2026](https://openreview.net/group?id=ICML.cc/2026/Workshop/FMSD#tab-your-consoles).
