import { CoreBackend } from "./core.js";
import { EnsembleGenerator, unshuffleAndAverage } from "./ensemble.js";
import { LabelEncoder, TransformToNumerical } from "./preprocessing.js";

export class TabICLClassifier {
  constructor({
    nEstimators = 8,
    normMethods = null,
    featShuffleMethod = "latin",
    classShuffleMethod = "shift",
    outlierThreshold = 4,
    softmaxTemperature = 0.9,
    averageLogits = true,
    supportManyClasses = true,
    batchSize = 8,
    randomState = 42,
    backend = new CoreBackend({ task: "classifier" }),
    schema = null,
  } = {}) {
    Object.assign(this, {
      nEstimators,
      normMethods,
      featShuffleMethod,
      classShuffleMethod,
      outlierThreshold,
      softmaxTemperature,
      averageLogits,
      supportManyClasses,
      batchSize,
      randomState,
      backend,
      schema,
    });
    this.maxClasses = 10;
  }

  fit(X, y, { kvCache = false } = {}) {
    if (X.length !== y.length) throw new Error("X and y must have the same number of rows");
    this.yEncoder = new LabelEncoder();
    const yEncoded = this.yEncoder.fitTransform(y);
    this.classes = this.yEncoder.classes;
    this.nClasses = this.classes.length;
    if (this.nClasses > this.maxClasses && kvCache) {
      throw new Error(`KV caching is not supported for ${this.nClasses} classes`);
    }
    if (this.nClasses > this.maxClasses && !this.supportManyClasses) {
      throw new Error(`Number of classes exceeds native maxClasses=${this.maxClasses}`);
    }
    if (this.nClasses > this.maxClasses && this.backend.model) {
      throw new Error("Many-class artifact inference is not implemented in the JavaScript core yet");
    }
    this.XEncoder = new TransformToNumerical({ schema: this.schema });
    const XNumeric = this.XEncoder.fitTransform(X);
    this.ensembleGenerator = new EnsembleGenerator({
      classification: true,
      nEstimators: this.nEstimators,
      normMethods: this.normMethods,
      featShuffleMethod: this.featShuffleMethod,
      classShuffleMethod: this.classShuffleMethod,
      outlierThreshold: this.outlierThreshold,
      randomState: this.randomState,
      preprocess: this.backend.inputMode !== "raw-core-standardizes",
    }).fit(XNumeric, yEncoded);
    this.cacheMode = normalizeCacheMode(kvCache);
    this.modelCache = this.cacheMode ? this.buildCache(this.cacheMode) : null;
    return this;
  }

  buildCache(cacheMode = "kv") {
    const cache = new Map();
    const trainData = this.ensembleGenerator.transform(null, "train");
    for (const [norm, data] of trainData) {
      cache.set(norm, data.Xs.map((XTrain, idx) => this.backend.prepare(XTrain, data.ys[idx], { cacheMode })));
    }
    return { cacheMode, byNorm: cache };
  }

  _enumerateTransformedViews(XNumeric) {
    this.assertFitted();
    if (!this.modelCache) throw new Error("A fitted model cache is required to enumerate transformed views");
    const views = [];
    const testData = this.ensembleGenerator.transform(XNumeric, "test");
    for (const [normName, data] of testData) {
      const caches = this.modelCache.byNorm.get(normName);
      if (!caches || caches.length !== data.Xs.length) {
        throw new Error(`KV cache does not match transformed ensemble views for norm "${normName}"`);
      }
      data.Xs.forEach((transformedTestRows, normViewIndex) => {
        views.push({
          viewIndex: views.length,
          normName,
          normViewIndex,
          featureShuffle: data.featureShuffles[normViewIndex].slice(),
          classShuffle: data.classShuffles[normViewIndex].slice(),
          transformedTestRows,
          cache: caches[normViewIndex],
        });
      });
    }
    return views;
  }

  predictProba(X) {
    this.assertFitted();
    const XNumeric = this.XEncoder.transform(X);
    const outputs = [];
    const classShuffles = [];
    if (this.modelCache) {
      for (const view of this._enumerateTransformedViews(XNumeric)) {
        outputs.push(this.backend.predictClassifierWithCache(
          view.cache,
          view.transformedTestRows,
          this.nClasses,
        ));
        classShuffles.push(view.classShuffle);
      }
    } else {
      const data = this.ensembleGenerator.transform(XNumeric, "both");
      const nTrain = this.ensembleGenerator.X.length;
      for (const batch of data.values()) {
        batch.Xs.forEach((XAll, idx) => {
          outputs.push(this.backend.predictClassifier(
            XAll.slice(0, nTrain),
            batch.ys[idx],
            XAll.slice(nTrain),
            this.nClasses,
          ));
          classShuffles.push(batch.classShuffles[idx]);
        });
      }
    }
    return unshuffleAndAverage(outputs, classShuffles, {
      averageLogits: this.averageLogits,
      temperature: this.softmaxTemperature,
    });
  }

  predictProbaView(X, viewIndex = 0) {
    this.assertFitted();
    this.assertKVCache();
    const XNumeric = this.XEncoder.transform(X);
    const views = this._enumerateTransformedViews(XNumeric);
    const view = selectView(views, viewIndex);
    const output = this.backend.predictClassifierWithCache(
      view.cache,
      view.transformedTestRows,
      this.nClasses,
    );
    return unshuffleAndAverage([output], [view.classShuffle], {
      averageLogits: true,
      temperature: this.softmaxTemperature,
    });
  }

  inspectQuery(X, { viewIndex = 0, full = false, includeEnsemble = true } = {}) {
    this.assertFitted();
    this.assertKVCache();
    if (!Array.isArray(X) || X.length !== 1) {
      throw new Error(`inspectQuery requires exactly one query row; received ${Array.isArray(X) ? X.length : "non-array input"}`);
    }
    if (!this.backend?.model?.inspectQuery) {
      throw new Error("inspectQuery requires a loaded browser artifact model");
    }

    const XNumeric = this.XEncoder.transform(X);
    const views = this._enumerateTransformedViews(XNumeric);
    const view = selectView(views, viewIndex);
    if (!view.cache?.native) {
      throw new Error("inspectQuery requires a browser fitted KV cache");
    }
    const selectedViewTrace = this.backend.model.inspectQuery(
      view.cache.native,
      view.transformedTestRows[0],
      this.nClasses,
      { full: full === true },
    );
    const selectedViewProbability = unshuffleAndAverage(
      [[selectedViewTrace.output.logits]],
      [view.classShuffle],
      {
        averageLogits: true,
        temperature: this.softmaxTemperature,
      },
    )[0];
    const result = {
      selectedView: {
        viewIndex: view.viewIndex,
        normName: view.normName,
        normViewIndex: view.normViewIndex,
        featureShuffle: view.featureShuffle.slice(),
        classShuffle: view.classShuffle.slice(),
        transformedQuery: view.transformedTestRows[0].slice(),
      },
      probabilityClassOrder: this.classes.slice(),
      selectedViewProbability,
      selectedViewTrace,
    };
    if (includeEnsemble) result.ensembleProbability = this.predictProba(X)[0];
    return result;
  }

  predict(X) {
    const probabilities = this.predictProba(X);
    const encoded = probabilities.map((row) => argmax(row));
    return this.yEncoder.inverseTransform(encoded);
  }

  assertFitted() {
    if (!this.ensembleGenerator || !this.yEncoder) throw new Error("TabICLClassifier is not fitted");
  }

  assertKVCache() {
    if (!this.modelCache || this.cacheMode !== "kv") {
      throw new Error("A fitted KV cache is required; call fit(X, y, { kvCache: \"kv\" })");
    }
  }
}

function selectView(views, viewIndex) {
  if (!Number.isInteger(viewIndex) || viewIndex < 0 || viewIndex >= views.length) {
    throw new Error(`Invalid viewIndex ${viewIndex}; expected an integer from 0 to ${views.length - 1}`);
  }
  return views[viewIndex];
}

function normalizeCacheMode(value) {
  if (!value) return null;
  if (value === true || value === "kv") return "kv";
  if (value === "repr") return "repr";
  throw new Error(`Invalid kvCache value: ${value}`);
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
