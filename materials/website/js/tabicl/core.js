import { loadFlatTensors, loadManifestAndBuffer, loadManifestAndWeights } from "./tensor.js";

export class UpstreamTensorStore {
  constructor(manifest, tensors) {
    this.manifest = manifest;
    this.tensors = tensors;
    this.config = manifest?.config || {};
    this.task = manifest?.task || (this.config.max_classes === 0 ? "regressor" : "classifier");
  }

  static fromBuffer(manifest, buffer, options = {}) {
    return new UpstreamTensorStore(manifest, loadFlatTensors(manifest, buffer, options));
  }

  static async load(baseUrl, options = {}) {
    const { manifest, tensors } = await loadManifestAndWeights(baseUrl, options);
    return new UpstreamTensorStore(manifest, tensors);
  }

  get(name) {
    const tensor = this.tensors.get(name);
    if (!tensor) throw new Error(`Missing tensor: ${name}`);
    return tensor;
  }
}

export class CoreBackend {
  constructor({ task = "classifier", tensorStore = null, model = null } = {}) {
    this.task = task;
    this.tensorStore = tensorStore;
    this.model = model;
    this.inputMode = "preprocessed";
  }

  static async fromBaseUrl(baseUrl, options = {}) {
    const { manifest, buffer } = await loadManifestAndBuffer(baseUrl, options);
    return new CoreBackend().loadArtifact(manifest, buffer);
  }

  async loadArtifact(manifest, buffer) {
    const module = await import("./browser-core.js");
    this.model = module.buildModel(toNanoManifest(manifest), buffer);
    this.task = manifest.task || (manifest.config?.max_classes === 0 ? "regressor" : "classifier");
    this.inputMode = "preprocessed";
    return this;
  }

  async loadClassifierArtifact(manifest, buffer) {
    if ((manifest.task || "classifier") !== "classifier") throw new Error("Expected a classifier manifest");
    return this.loadArtifact(manifest, buffer);
  }

  async loadRegressorArtifact(manifest, buffer) {
    if ((manifest.task || "regressor") !== "regressor") throw new Error("Expected a regressor manifest");
    return this.loadArtifact(manifest, buffer);
  }

  prepare(XTrain, yTrain, options = {}) {
    this.assertModel();
    if (options.cacheMode === "repr") {
      throw new Error("repr cache mode is not implemented for the JavaScript core yet");
    }
    return { mode: options.cacheMode || "kv", native: this.model.prepareContext(XTrain, yTrain), options };
  }

  predictClassifier(XTrain, yTrain, XTest, nClasses) {
    this.assertModel();
    const logits = this.model.predict(XTrain, yTrain, XTest);
    return logits.map((row) => row.slice(0, nClasses));
  }

  predictClassifierWithCache(cache, XTest, nClasses) {
    this.assertModel();
    if (!cache?.native) throw new Error("A fitted model cache is required for prediction");
    return this.model.predictQueries(cache.native, XTest).map((row) => row.slice(0, nClasses));
  }

  predictRegressor(XTrain, yTrain, XTest) {
    this.assertModel();
    return this.model.predict(XTrain, yTrain, XTest);
  }

  predictRegressorWithCache(cache, XTest) {
    this.assertModel();
    if (!cache?.native) throw new Error("A fitted model cache is required for prediction");
    return this.model.predictQueries(cache.native, XTest);
  }

  assertModel() {
    if (!this.model) throw new Error("Load a model artifact before inference");
  }
}

export function toNanoClassifierManifest(manifest) {
  return toNanoManifest(manifest);
}

export function toNanoManifest(manifest) {
  return {
    ...manifest,
    config: toNanoConfig(manifest.config || {}),
    tensors: manifest.tensors
      .map((tensor) => ({ ...tensor, name: translateUpstreamKey(tensor.name) }))
      .filter((tensor) => tensor.name),
  };
}

function toNanoConfig(config) {
  return {
    embed_dim: config.embed_dim ?? 128,
    col_num_blocks: config.col_num_blocks ?? 3,
    row_num_blocks: config.row_num_blocks ?? 3,
    icl_num_blocks: config.icl_num_blocks ?? 12,
    col_nhead: config.col_nhead ?? 8,
    row_nhead: config.row_nhead ?? 8,
    icl_nhead: config.icl_nhead ?? 8,
    n_cls_cols: config.row_num_cls ?? config.n_cls_cols ?? 4,
    n_inducing: config.col_num_inds ?? config.n_inducing ?? 128,
    feature_group_size: config.col_feature_group_size ?? config.feature_group_size ?? 3,
    group_offsets: Array.from({ length: config.col_feature_group_size ?? config.feature_group_size ?? 3 }, (_, i) => 2 ** i),
    rope_theta: config.row_rope_base ?? config.rope_theta ?? 100000,
    max_classes: config.max_classes ?? 10,
    num_quantiles: config.num_quantiles ?? 999,
    standardize_input: false,
  };
}

function mapTransformerBlockSuffix(suffix) {
  return suffix
    .replace("attn.in_proj_weight", "in_proj_weight")
    .replace("attn.in_proj_bias", "in_proj_bias")
    .replace("attn.out_proj.", "out_proj.")
    .replace("attn.ssmax_layer.", "ssmax_layer.")
    .replace("linear1.", "mlp.0.")
    .replace("linear2.", "mlp.2.")
    .replace("norm1.", "ln_attn.")
    .replace("norm2.", "ln_mlp.");
}

function translateUpstreamKey(key) {
  if (
    key === "row_cls_tokens" ||
    /^(x_embed|y_embed_in|y_embed_icl|row_ln|out_ln|out_mlp)\./.test(key) ||
    /^(col_blocks|row_blocks|icl_blocks)\.\d+\./.test(key)
  ) return key;
  if (key === "col_embedder.in_linear.weight") return "x_embed.weight";
  if (key === "col_embedder.in_linear.bias") return "x_embed.bias";
  if (key === "col_embedder.y_encoder.weight") return "y_embed_in.weight";
  if (key === "col_embedder.y_encoder.bias") return "y_embed_in.bias";
  let match = key.match(/^col_embedder\.tf_col\.blocks\.(\d+)\.(.*)$/);
  if (match) {
    if (match[2] === "ind_vectors") return `col_blocks.${match[1]}.inducing_vectors`;
    const inner = match[2].match(/^multihead_attn([12])\.(.*)$/);
    if (!inner) throw new Error(`Unmapped column block key: ${key}`);
    return `col_blocks.${match[1]}.${inner[1] === "1" ? "tfm1" : "tfm2"}.${mapTransformerBlockSuffix(inner[2])}`;
  }
  if (key === "row_interactor.cls_tokens") return "row_cls_tokens";
  if (key === "row_interactor.out_ln.weight") return "row_ln.weight";
  if (key === "row_interactor.out_ln.bias") return "row_ln.bias";
  if (key === "row_interactor.tf_row.rope.freqs") return null;
  match = key.match(/^row_interactor\.tf_row\.blocks\.(\d+)\.(.*)$/);
  if (match) return `row_blocks.${match[1]}.${mapTransformerBlockSuffix(match[2])}`;
  if (key === "icl_predictor.ln.weight") return "out_ln.weight";
  if (key === "icl_predictor.ln.bias") return "out_ln.bias";
  if (key === "icl_predictor.y_encoder.weight") return "y_embed_icl.weight";
  if (key === "icl_predictor.y_encoder.bias") return "y_embed_icl.bias";
  if (key === "icl_predictor.decoder.0.weight") return "out_mlp.0.weight";
  if (key === "icl_predictor.decoder.0.bias") return "out_mlp.0.bias";
  if (key === "icl_predictor.decoder.2.weight") return "out_mlp.2.weight";
  if (key === "icl_predictor.decoder.2.bias") return "out_mlp.2.bias";
  match = key.match(/^icl_predictor\.tf_icl\.blocks\.(\d+)\.(.*)$/);
  if (match) return `icl_blocks.${match[1]}.${mapTransformerBlockSuffix(match[2])}`;
  throw new Error(`Unmapped upstream tensor key: ${key}`);
}
