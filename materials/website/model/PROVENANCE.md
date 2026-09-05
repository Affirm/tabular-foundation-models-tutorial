# Browser classifier provenance

The live website runs the TabICLv2 browser classifier artifact produced by the
parity-checked JavaScript runtime:

- Manifest schema: `tabicl-browser-js/flat-tensors-v1`
- Task: `classifier`
- Browser-export package: `tabicl==2.0.2`
- Notebook package: `tabicl==2.2.0` (separately locked; not used to produce this artifact)
- Repository: https://github.com/soda-inria/tabicl
- Checkpoint host: https://huggingface.co/jingang/TabICL
- Pinned repository revision: `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`
- Checkpoint: `tabicl-classifier-v2-20260212.ckpt`
- Checkpoint SHA-256:
  `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`
- Browser binary: `tabicl.bin` (27,894,972 bytes)
- Binary SHA-256:
  `ee6294132561242077b50dd2da7d0d42794fe0edb1cb53fb67512280d21d71fb`
- Runtime: `../js/tabicl/` with orchestration in `../js/tabicl/worker.js`

All browser loaders compute the binary SHA-256 with Web Crypto and compare it
with the manifest before dequantizing or executing the model. A matching byte
length alone is not accepted.

Each live query prediction uses the same eight-view classifier procedure as the
exported package and is parity-checked within the documented quantization tolerances:
two normalization methods, deterministic Latin
feature shuffles, shifted class labels, logit averaging, temperature 0.9, and
random state 42. The selectable trace is the real inspection of one chosen
ensemble view; its probability and final-block attention are selected-view
values and are never presented as ensemble-averaged attention. The displayed
prediction remains the eight-view ensemble.

The 2D slice and 3D volume are intentionally labeled as a **view 1 preview
field**. Their `p = 0.5` boundary is computed from view 1 only so interactive
field generation remains bounded; it is not the final eight-view query output.
The playground's 20-row context is outside the documented TabICLv2 pretraining
range of 300 to 48K rows. The upstream FAQ says generalization below 300 rows
has not been tested, so the displayed result is an out-of-regime illustration.

The legacy `../js/tabicl/nanotabicl.js` bridge remains only for the Svelte explainer's
selected-view core inspection. It adapts the browser manifest's upstream tensor names and keeps
core input standardization enabled; it does not synthesize values or claim
ensemble behavior. It follows the released TabICLv2 checkpoint's feature-group
offsets `[1, 2, 4]`; the standalone nanoTabICL source uses `[0, 1, 3]`.

This browser scope intentionally excludes regression and classification with
more than 10 classes. The browser artifact stores upstream-named tensors as int8/fp16 for
browser delivery. TabICL source is BSD 3-Clause licensed; the upstream
repository and checkpoint host remain authoritative for current terms.

The 27.9 MB compressed artifact expands to about 105 MiB when its 27,552,258
parameters are materialized as float32 arrays. The main playground and embedded
Iris inspector each initialize their own model instance.
