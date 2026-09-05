/* Web Worker: hosts the TabICLv2 browser classifier (int8) off the main thread.
 * Protocol (postMessage):
 *   in  {type:'load'}                         -> streams {type:'progress',loaded,total}, then {type:'loaded',bytes}
 *   in  {type:'prepare', X, y, tag}           -> {type:'prepared', tag, viewCount}
 *   in  {type:'predict', X, tag}              -> {type:'result', tag, proba}
 *   in  {type:'inspect', X, tag, viewIndex}   -> {type:'inspection', tag, proba, selectedViewProbability, selectedView, selectedViewTrace, attention, block, heads}
 *   in  {type:'grid', X, tag, chunk}          -> repeated {type:'gridChunk', tag, start, proba} then {type:'gridDone', tag}
 *   in  {type:'cancelGrid'}                    -> stop the current grid between chunks
 *   any failure                               -> {type:'error', message}
 */
import { TabICLClassifier } from "./classifier.js";
import { CoreBackend } from "./core.js";
import { verifySha256 } from "./tensor.js";

let backend = null, estimator = null, activeGridTag = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      if (backend) { self.postMessage({ type: "loaded", bytes: 0 }); return; }
      const base = new URL(m.modelBase || "../../model/", import.meta.url);
      const manifest = await (await fetch(new URL("manifest.json", base), { cache: "no-store" })).json();
      if (manifest.schema !== "tabicl-browser-js/flat-tensors-v1") {
        throw new Error("Expected the TabICLv2 browser manifest; clear the stale site cache and reload");
      }
      const resp = await fetch(new URL(manifest.binary || "tabicl.bin", base), { cache: "force-cache" });
      const total = +(resp.headers.get("Content-Length") || 0);
      const reader = resp.body.getReader();
      const chunks = []; let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); loaded += value.length;
        self.postMessage({ type: "progress", loaded, total });
      }
      const buf = new Uint8Array(loaded); let off = 0;
      for (const c of chunks) { buf.set(c, off); off += c.length; }
      if (manifest.total_bytes && loaded !== manifest.total_bytes) {
        throw new Error(`Model byte length ${loaded} does not match manifest ${manifest.total_bytes}`);
      }
      await verifySha256(buf.buffer, manifest.binary_sha256, manifest.binary || "tabicl.bin");
      backend = new CoreBackend({ task: "classifier" });
      await backend.loadClassifierArtifact(manifest, buf.buffer);
      self.postMessage({ type: "loaded", bytes: loaded });
    } else if (m.type === "prepare") {
      activeGridTag = null;
      if (!backend) throw new Error("Load the browser classifier before preparing context");
      estimator = new TabICLClassifier({
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
      estimator.fit(m.X, m.y, { kvCache: true });
      self.postMessage({ type: "prepared", tag: m.tag, viewCount: 8 });
    } else if (m.type === "predict") {
      if (!estimator) throw new Error("Prepare the classifier before prediction");
      const proba = estimator.predictProba(m.X);
      self.postMessage({ type: "result", tag: m.tag, proba });
    } else if (m.type === "inspect") {
      if (!estimator) throw new Error("Prepare the classifier before inspection");
      const inspection = estimator.inspectQuery([m.X], {
        viewIndex: m.viewIndex || 0,
        full: m.full === true,
        includeEnsemble: true,
      });
      const trace = inspection.selectedViewTrace;
      const finalBlock = trace.iclBlocks[trace.iclBlocks.length - 1];
      self.postMessage({
        type: "inspection",
        tag: m.tag,
        proba: inspection.ensembleProbability,
        selectedViewProbability: inspection.selectedViewProbability,
        selectedView: inspection.selectedView,
        selectedViewTrace: trace,
        attention: finalBlock.attention.average,
        block: finalBlock.block,
        heads: trace.config.heads,
        attentionScope: "selected-view",
      });
    } else if (m.type === "grid") {
      if (!estimator) throw new Error("Prepare the classifier before field prediction");
      const X = m.X, chunk = m.chunk || 48, gridEstimator = estimator;
      activeGridTag = m.tag;
      for (let s = 0; s < X.length; s += chunk) {
        if (activeGridTag !== m.tag) return;
        const proba = gridEstimator.predictProbaView(X.slice(s, s + chunk), m.viewIndex || 0);
        self.postMessage({ type: "gridChunk", tag: m.tag, start: s, proba });
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      if (activeGridTag !== m.tag) return;
      activeGridTag = null;
      self.postMessage({ type: "gridDone", tag: m.tag });
    } else if (m.type === "cancelGrid") {
      activeGridTag = null;
    }
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.stack) || err) });
  }
};
