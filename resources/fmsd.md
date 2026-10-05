# FMSD 2026 workshop resources

[Tutorial home](../README.md) · [Interactive guide](https://affirm.github.io/tabular-foundation-models-tutorial/) · [NeurIPS 2026 resources](neurips.md)

Foundation Models for Structured Data (FMSD) is an ICML workshop covering tabular and time-series prediction. This collection connects our tutorial to the 2026 workshop's research discussions. Checked October 4, 2026.

## Start with the workshop program

The [2026 workshop](https://icml-structured-fm-workshop.github.io/) took place on July 11 in Seoul. Its program brings together model design, training data, evaluation, and deployment. Use the [accepted-paper collection](https://icml-structured-fm-workshop.github.io/accepted-papers/) for paper links and the [ICML event page](https://icml.cc/virtual/2026/workshop/54066) for event materials.

| Program topic | Connection to the tutorial |
| --- | --- |
| TabICLv2 invited talk | Continue from our worked example into its training and scaling choices. |
| Scientific tabular modeling invited talk | Explore how dataset size and scientific applications change modeling requirements. |
| Multimodal time-series invited talk | Extend the discussion beyond individual tables to forecasting and additional modalities. |
| Industry spotlights | Compare the practical concerns raised by SAP, Layer 6, Prior Labs, AWS, and other participating teams. |

The [official schedule](https://icml-structured-fm-workshop.github.io/#schedule) lists the sessions and presentation titles.

## Selected papers by research question

These entries are drawn from the workshop's [accepted papers](https://icml-structured-fm-workshop.github.io/accepted-papers/). Workshop presentation is a separate venue designation from main-conference acceptance.

| Question | Reading |
| --- | --- |
| How can we audit the pretraining data behind a TFM? | [Dataset inference and provenance auditing](https://openreview.net/forum?id=u2uOPq1u6I) |
| How does prior-fitted prediction extend to survival outcomes? | [SurvivalPFN](https://openreview.net/forum?id=PDik7bpFhE) |
| How should attention designs be compared? | [Attention benchmarking](https://openreview.net/forum?id=rwtcugrpDq) |
| How representative are public benchmarks of enterprise tables? | [Enterprise data and public benchmarks](https://openreview.net/forum?id=PXSBtjo3Gd) |

For each paper, inspect its task definition, context construction, baselines, and evaluation splits. Follow author-provided code and data links from the paper record where available.

## Hands-on study

For implementation work, pair the [TabICLv2 paper](https://arxiv.org/abs/2602.11139) with [official code and weights](https://github.com/soda-inria/tabicl), or study the smaller [nanoTabICL implementation](https://github.com/soda-inria/nanotabicl). Our [interactive guide](https://affirm.github.io/tabular-foundation-models-tutorial/) introduces the computation visually. The [benchmark collection](../README.md#benchmarks-and-evaluation) links to code, datasets, and evaluation protocols.
