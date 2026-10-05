# NeurIPS 2026 resources for tabular foundation models

[Tutorial home](../README.md) · [Interactive guide](https://affirm.github.io/tabular-foundation-models-tutorial/) · [FMSD 2026 workshop](https://openreview.net/group?id=ICML.cc/2026/Workshop/FMSD#tab-your-consoles)

Start with our interactive guide, then explore selected NeurIPS 2026 work on tabular models, broader prediction tasks, and evaluation. The collection below covers papers listed in the [official conference directory](https://neurips.cc/Downloads/2026). Checked October 5, 2026.

## Our guide at NeurIPS 2026

**Table as Prompt: An Interactive Guide to Tabular Foundation Models** is accepted for presentation at the NeurIPS 2026 Education Track.

The [interactive guide](https://affirm.github.io/tabular-foundation-models-tutorial/) introduces task-specific fitting, adaptation through labeled examples, and PFN theory. It then follows TabICLv2 through training and inference, with a browser playground and model inspection. Basic supervised learning is enough to start. Use the [Python notebook](../materials/notebooks/01_tabicl_primer.ipynb) to work through prediction in code.

The [Education Track website](https://neurips-education-track.github.io/) collects track information. The [official call](https://neurips.cc/Conferences/2026/CallforEducationalResources) describes the track's educational scope and formats.

## NeurIPS 2026 models and methods

| Paper | What to study | Sources and resources |
| --- | --- | --- |
| FlexTab | A shared encoder produces target-agnostic row representations; task-specific decoders support classification, regression, clustering, and other tabular tasks. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/153378) · [Paper](https://arxiv.org/abs/2606.30336) |
| TabClustPFN | Infer cluster assignments and the number of clusters using a PFN trained on synthetic clustering tasks. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/150101) · [Paper](https://arxiv.org/abs/2601.21656) · [Code and checkpoint access](https://github.com/Tianqi-Zhao/TabClustPFN) |
| SurvivalPFN | PFN pretraining for time-to-event prediction with right-censored observations. Study how the prior and prediction target change for survival analysis. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/151336) · [Paper](https://arxiv.org/abs/2605.15488) · [GitHub](https://github.com/rgklab/SurvivalPFN) · [Weights](https://huggingface.co/shi-ang/SurvivalPFN) |

## NeurIPS 2026 benchmarks and evaluation

| Paper | What to study | Sources and resources |
| --- | --- | --- |
| Beyond IID / BeyondArena | Evaluate generalization across IID, temporal, and grouped tasks, different table sizes, and varied feature types. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139490) · [Paper](https://arxiv.org/abs/2606.30410) · [TabArena code](https://github.com/autogluon/tabarena) · [Data Foundry](https://github.com/tabarena/data-foundry) |
| TRL-Bench | Compare row, column, and table representations under shared downstream evaluation conditions. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139385) · [Paper](https://arxiv.org/abs/2606.09323) · [Code and datasets](https://github.com/LOGO-CUHKSZ/TRL-Bench) |
| MulTaBench | Compare models on tables that combine structured features with text or images. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/138893) · [Paper](https://arxiv.org/abs/2605.10616) · [Code and datasets](https://github.com/alanarazi7/MulTaBench) |
| TabPrep | Study how feature engineering changes comparisons among tree-based, neural, linear, and foundation models. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139065) · [Paper](https://arxiv.org/abs/2606.02384) · [GitHub](https://github.com/atschalz/tabprep) |
| Benchmarking Attention for Tabular Foundation Models | Compare attention backends for row and column operations across table shapes and GPU hardware. | [NeurIPS record](https://neurips.cc/virtual/2026/poster/139839) · [Paper](https://arxiv.org/abs/2609.31306) · [Code and results](https://github.com/SAP-samples/tabular-attention-benchmark) |

These papers address different questions: generalization, multimodal inputs, preprocessing, and computational cost. Read their splits, tuning budgets, and hardware settings before comparing results. Strong task-specific baselines remain essential.

## Further NeurIPS 2026 reading by topic

This index broadens the collection beyond the worked examples above. Links point to entries in the official 2026 directory; labels are shortened for scanning. Use each paper to check its methods and results.

### Prediction, clustering, and causal structure

| Reading | Topic |
| --- | --- |
| [TabK](https://neurips.cc/virtual/2026/poster/148754) | Cluster-count inference |
| [CausalTab](https://neurips.cc/virtual/2026/poster/150884) | Causal discovery |
| [DynaPFN](https://neurips.cc/virtual/2026/poster/149937) | Dynamical forecasting |
| [Causal ordering for prediction](https://neurips.cc/virtual/2026/poster/153895) | Causal structure in ICL |
| [Semantic feature embeddings](https://neurips.cc/virtual/2026/poster/149526) | Feature meaning and context |

### Relational data

| Reading | Topic |
| --- | --- |
| [Few-shot relational pretraining](https://neurips.cc/virtual/2026/poster/149169) | Prediction across linked tables |
| [RelAgent](https://neurips.cc/virtual/2026/poster/151742) | Agent-driven relational analysis |

### Generation

| Reading | Topic |
| --- | --- |
| [TabDLM](https://neurips.cc/virtual/2026/poster/155779) | Numeric and text generation |
| [TabWorld](https://neurips.cc/virtual/2026/poster/150304) | World models for tables |
| [Generative TFM](https://neurips.cc/virtual/2026/poster/149211) | Generative modeling |
| [Quality and privacy in generation](https://neurips.cc/virtual/2026/poster/152928) | ICL for synthetic tables |

### Robustness and privacy

| Reading | Topic |
| --- | --- |
| [RAD-TFM](https://neurips.cc/virtual/2026/poster/155295) | Domain adaptation |
| [Harmful-shift detection](https://neurips.cc/virtual/2026/poster/150327) | PFN reliability under shift |
| [Embedding privacy audit](https://neurips.cc/virtual/2026/poster/149397) | Information leakage |
| [TFM generalization theory](https://neurips.cc/virtual/2026/poster/150354) | Generalization behavior |

### Specialized evaluation

| Reading | Topic |
| --- | --- |
| [STRABLE](https://neurips.cc/virtual/2026/poster/139430) | String-valued features |
| [TabBioMed](https://neurips.cc/virtual/2026/poster/139791) | Biomedical tables |
| [Genetic-data limitations](https://neurips.cc/virtual/2026/poster/139799) | Failure cases in genetics |
