import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { buildModel as buildLegacyInspectionModel } from "../materials/website/js/tabicl/nanotabicl.js";
import { TabICLClassifier } from "../materials/website/js/tabicl/classifier.js";
import { CoreBackend, toNanoClassifierManifest } from "../materials/website/js/tabicl/core.js";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const modelDir = join(root, "materials/website/model");
const manifestPath = join(modelDir, "manifest.json");
const binaryPath = join(modelDir, "tabicl.bin");

const X = [
  [-1.8, -1.1, 0.2],
  [-1.2, -0.7, -0.4],
  [-0.6, -1.4, 0.6],
  [-0.2, -0.3, -0.8],
  [0.3, 0.5, 0.7],
  [0.8, 1.3, -0.5],
  [1.4, 0.6, 0.4],
  [1.9, 1.5, -0.1],
];
const y = [0, 0, 0, 0, 1, 1, 1, 1];
const query = [[0.25, -0.15, 0.35]];

test("browser manifest translation is idempotent for stale mapped manifests", () => {
  const mapped = toNanoClassifierManifest({
    config: { embed_dim: 128, n_cls_cols: 4, feature_group_size: 3 },
    tensors: [
      { name: "row_cls_tokens", shape: [1, 1, 4, 128] },
      { name: "x_embed.weight", shape: [128, 3] },
      { name: "icl_blocks.0.in_proj_weight", shape: [1536, 512] },
    ],
  });
  assert.deepEqual(mapped.tensors.map((tensor) => tensor.name), [
    "row_cls_tokens",
    "x_embed.weight",
    "icl_blocks.0.in_proj_weight",
  ]);
});

test("legacy Svelte bridge inspects the browser manifest without fabricated values", () => {
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const bytes = readFileSync(binaryPath);
  const model = buildLegacyInspectionModel(
    manifest,
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const cache = model.prepareContext(X, y);
  const trace = model.inspectQuery(cache, query[0], 2);

  assert.equal(trace.iclBlocks.length, 12);
  assert.equal(trace.iclBlocks.at(-1).attention.heads.length, 8);
  assert.equal(trace.output.probabilities.length, 2);
  assert.ok(trace.output.logits.every(Number.isFinite));
  assert.ok(Math.abs(sum(trace.output.probabilities) - 1) < 1e-12);
  assert.ok(Math.abs(
    trace.input.standardized[0]
    - (query[0][0] - trace.input.mean[0]) / trace.input.std[0],
  ) < 1e-6);
});

test("promoted browser classifier supports exact eight-view inspection", { timeout: 600_000 }, async (t) => {
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const bytes = readFileSync(binaryPath);
  const hash = createHash("sha256").update(bytes).digest("hex");

  await t.test("preserves browser artifact identity", () => {
    assert.equal(manifest.schema, "tabicl-browser-js/flat-tensors-v1");
    assert.equal(manifest.task, "classifier");
    assert.equal(manifest.package, "tabicl==2.0.2");
    assert.equal(manifest.binary, basename(binaryPath));
    assert.equal(manifest.binary_sha256, "ee6294132561242077b50dd2da7d0d42794fe0edb1cb53fb67512280d21d71fb");
    assert.equal(hash, manifest.binary_sha256);
    assert.equal(bytes.byteLength, manifest.total_bytes);
  });

  const timings = {};
  const backend = new CoreBackend({ task: "classifier" });
  const loadStart = performance.now();
  await backend.loadClassifierArtifact(
    manifest,
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  timings.loadMs = performance.now() - loadStart;

  const classifier = new TabICLClassifier({
    nEstimators: 8,
    normMethods: null,
    featShuffleMethod: "latin",
    classShuffleMethod: "shift",
    outlierThreshold: 4,
    softmaxTemperature: 0.9,
    averageLogits: true,
    supportManyClasses: false,
    batchSize: 1,
    randomState: 42,
    backend,
  });
  const fitStart = performance.now();
  classifier.fit(X, y, { kvCache: true });
  timings.fitMs = performance.now() - fitStart;

  await t.test("prepares all eight deterministic views", () => {
    const numericQuery = classifier.XEncoder.transform(query);
    const views = classifier._enumerateTransformedViews(numericQuery);
    assert.equal(views.length, 8);
    assert.deepEqual(views.map((view) => view.viewIndex), [0, 1, 2, 3, 4, 5, 6, 7]);
    assert.ok(views.every((view) => view.cache?.native));
  });

  const inspectStart = performance.now();
  const inspection = classifier.inspectQuery(query, {
    viewIndex: 3,
    full: false,
    includeEnsemble: true,
  });
  timings.inspectMs = performance.now() - inspectStart;

  await t.test("separates ensemble output from the selected-view trace", () => {
    assert.equal(inspection.selectedView.viewIndex, 3);
    assert.deepEqual(
      Object.keys(inspection).sort(),
      [
        "ensembleProbability",
        "probabilityClassOrder",
        "selectedView",
        "selectedViewProbability",
        "selectedViewTrace",
      ],
    );
    assert.deepEqual(inspection.ensembleProbability, classifier.predictProba(query)[0]);
    assert.deepEqual(inspection.selectedViewProbability, classifier.predictProbaView(query, 3)[0]);
    assert.equal(inspection.ensembleProbability.length, 2);
    assert.equal(inspection.selectedViewProbability.length, 2);
  });

  await t.test("returns real final-block selected-view attention", () => {
    const trace = inspection.selectedViewTrace;
    assert.equal(trace.iclBlocks.length, 12);
    const finalBlock = trace.iclBlocks.at(-1);
    assert.equal(finalBlock.block, 12);
    assert.equal(finalBlock.attention.average.length, X.length);
    assert.equal(finalBlock.attention.heads.length, 8);
    assert.ok(finalBlock.attention.heads.every((head) => head.length === X.length));
    assert.ok(Math.abs(sum(finalBlock.attention.average) - 1) < 1e-6);
  });

  const fullStart = performance.now();
  const full = classifier.inspectQuery(query, {
    viewIndex: 3,
    full: true,
    includeEnsemble: false,
  });
  timings.fullInspectMs = performance.now() - fullStart;

  await t.test("full mode preserves outputs and expands activations", () => {
    assert.ok(!Object.hasOwn(full, "ensembleProbability"));
    assert.deepEqual(full.selectedViewProbability, inspection.selectedViewProbability);
    assert.deepEqual(full.selectedViewTrace.output.logits, inspection.selectedViewTrace.output.logits);
    assert.equal(full.selectedViewTrace.rowVector.shown, full.selectedViewTrace.rowVector.dim);
    assert.equal(full.selectedViewTrace.output.hidden.shown, full.selectedViewTrace.output.hidden.dim);
  });

  t.diagnostic(
    `small fixture timings: load=${timings.loadMs.toFixed(1)}ms, fit8=${timings.fitMs.toFixed(1)}ms, `
    + `inspect8=${timings.inspectMs.toFixed(1)}ms, fullSelected=${timings.fullInspectMs.toFixed(1)}ms`,
  );
});

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}
