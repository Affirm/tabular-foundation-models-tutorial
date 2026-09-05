import { Random } from "./random.js";
import { PreprocessingPipeline, UniqueFeatureFilter } from "./preprocessing.js";

export class Shuffler {
  constructor({ nElements, method = "latin", randomState = 42, maxElementsForLatin = 4000 } = {}) {
    this.nElements = nElements;
    this.method = method;
    this.randomState = randomState;
    this.maxElementsForLatin = maxElementsForLatin;
  }

  shuffle(n) {
    if (this.nElements <= 0) return [];
    if (this.method === "none" || n === 1) return [range(this.nElements)];
    const method = this.method === "latin" && this.nElements > this.maxElementsForLatin ? "random" : this.method;
    if (method === "shift") {
      const base = range(this.nElements);
      return unique(Array.from({ length: this.nElements }, (_, i) => base.slice(-i).concat(base.slice(0, -i))));
    }
    if (method === "random") {
      const rng = new Random(this.randomState);
      const indices = range(this.nElements);
      if (this.nElements <= 5) {
        const allPermutations = permutations(indices);
        return rng.sample(allPermutations, Math.min(n, allPermutations.length));
      }
      return Array.from({ length: n }, () => rng.sample(indices, this.nElements));
    }
    if (method === "latin") {
      return latinSquares(this.nElements, new Random(this.randomState));
    }
    throw new Error(`Invalid shuffle method: ${method}`);
  }
}

export class EnsembleGenerator {
  constructor({
    classification,
    nEstimators = 8,
    normMethods = null,
    featShuffleMethod = "latin",
    classShuffleMethod = "shift",
    outlierThreshold = 4,
    randomState = 42,
    preprocess = true,
  } = {}) {
    this.classification = classification;
    this.nEstimators = nEstimators;
    this.normMethods = Array.isArray(normMethods) ? normMethods : (normMethods ? [normMethods] : ["none", "power"]);
    this.featShuffleMethod = featShuffleMethod;
    this.classShuffleMethod = classShuffleMethod;
    this.outlierThreshold = outlierThreshold;
    this.randomState = randomState;
    this.preprocess = preprocess;
  }

  fit(X, y) {
    this.uniqueFilter = new UniqueFeatureFilter();
    this.X = this.uniqueFilter.fitTransform(X);
    this.y = y.slice();
    this.nFeatures = this.X[0].length;
    this.nClasses = this.classification ? new Set(y).size : 0;
    this.configs = this.generateConfigs();
    this.preprocessors = new Map();
    if (this.preprocess) {
      for (const norm of this.configs.keys()) {
        const pipeline = new PreprocessingPipeline({
          normalizationMethod: norm,
          outlierThreshold: this.outlierThreshold,
          randomState: this.randomState,
        });
        pipeline.fit(this.X);
        this.preprocessors.set(norm, pipeline);
      }
    }
    return this;
  }

  generateConfigs() {
    const featureShuffles = new Shuffler({
      nElements: this.nFeatures,
      method: this.featShuffleMethod,
      randomState: this.randomState,
    }).shuffle(this.nEstimators);
    const classShuffles = this.classification
      ? new Shuffler({
        nElements: this.nClasses,
        method: this.classShuffleMethod,
        randomState: this.randomState,
      }).shuffle(this.nEstimators)
      : [null];
    const pairs = cartesian(featureShuffles, classShuffles);
    const rng = new Random(this.randomState);
    const shuffledPairs = rng.shuffle(pairs);
    const selected = cartesian(shuffledPairs, this.normMethods).slice(0, this.nEstimators);
    const grouped = new Map();
    for (const [[featureShuffle, classShuffle], norm] of selected) {
      if (!grouped.has(norm)) grouped.set(norm, []);
      grouped.get(norm).push({ featureShuffle, classShuffle });
    }
    return grouped;
  }

  transform(X = null, mode = "both") {
    const test = X ? this.uniqueFilter.transform(X) : null;
    const result = new Map();
    for (const [norm, configs] of this.configs.entries()) {
      const pipeline = this.preprocessors.get(norm);
      const trainNorm = pipeline ? pipeline.transform(this.X) : this.X.map((row) => row.slice());
      const testNorm = test ? (pipeline ? pipeline.transform(test) : test.map((row) => row.slice())) : null;
      const Xs = [];
      const ys = [];
      const featureShuffles = [];
      const classShuffles = [];
      for (const config of configs) {
        featureShuffles.push(config.featureShuffle);
        classShuffles.push(config.classShuffle);
        if (mode === "train") {
          Xs.push(applyFeatureShuffle(trainNorm, config.featureShuffle));
          ys.push(this.applyClassShuffle(this.y, config.classShuffle));
        } else if (mode === "test") {
          Xs.push(applyFeatureShuffle(testNorm, config.featureShuffle));
        } else if (mode === "both") {
          Xs.push([
            ...applyFeatureShuffle(trainNorm, config.featureShuffle),
            ...applyFeatureShuffle(testNorm, config.featureShuffle),
          ]);
          ys.push(this.applyClassShuffle(this.y, config.classShuffle));
        } else {
          throw new Error(`Invalid transform mode: ${mode}`);
        }
      }
      result.set(norm, { Xs, ys, featureShuffles, classShuffles });
    }
    return result;
  }

  applyClassShuffle(y, shuffle) {
    if (!this.classification || !shuffle) return y.slice();
    return y.map((label) => shuffle[Number(label)]);
  }

  classShuffles() {
    return Array.from(this.configs.values()).flat().map((config) => config.classShuffle);
  }
}

export function unshuffleAndAverage(outputs, classShuffles, { averageLogits = true, temperature = 0.9 } = {}) {
  if (!outputs.length) return [];
  const nSamples = outputs[0].length;
  const nClasses = outputs[0][0].length;
  const avg = Array.from({ length: nSamples }, () => new Array(nClasses).fill(0));
  for (let est = 0; est < outputs.length; est++) {
    const shuffle = classShuffles[est] || range(nClasses);
    for (let row = 0; row < nSamples; row++) {
      for (let cls = 0; cls < nClasses; cls++) avg[row][cls] += outputs[est][row][shuffle[cls]] || 0;
    }
  }
  for (const row of avg) for (let cls = 0; cls < nClasses; cls++) row[cls] /= outputs.length;
  if (!averageLogits) return renormalizeRows(avg);
  return avg.map((row) => softmax(row, temperature));
}

export function applyFeatureShuffle(X, shuffle) {
  return X.map((row) => shuffle.map((idx) => row[idx]));
}

export function range(n) {
  return Array.from({ length: n }, (_, i) => i);
}

function cartesian(a, b) {
  const out = [];
  for (const av of a) for (const bv of b) out.push([av, bv]);
  return out;
}

function unique(shuffles) {
  const seen = new Set();
  const out = [];
  for (const shuffle of shuffles) {
    const key = shuffle.join(",");
    if (!seen.has(key)) {
      seen.add(key);
      out.push(shuffle);
    }
  }
  return out;
}

function permutations(values) {
  if (values.length === 0) return [[]];
  const result = [];
  const used = new Array(values.length).fill(false);
  const current = [];
  function visit() {
    if (current.length === values.length) {
      result.push(current.slice());
      return;
    }
    for (let i = 0; i < values.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      current.push(values[i]);
      visit();
      current.pop();
      used[i] = false;
    }
  }
  visit();
  return result;
}

function latinSquares(nElements, rng) {
  function recursiveLatin(symbols) {
    if (symbols.length === 1) return [symbols.slice()];
    const sym = rng.choice(symbols);
    const remaining = symbols.filter((value) => value !== sym);
    const square = recursiveLatin(remaining);
    square.push(square[0].slice());
    for (let i = 0; i < symbols.length; i++) square[i].splice(i, 0, sym);
    return square;
  }

  const square = recursiveLatin(range(nElements));
  const shuffledRows = rng.shuffle(square);
  const transposed = shuffledRows[0].map((_, col) => shuffledRows.map((row) => row[col]));
  return rng.shuffle(transposed);
}

function softmax(values, temperature) {
  let max = -Infinity;
  for (const value of values) max = Math.max(max, value / temperature);
  const exps = values.map((value) => Math.exp(value / temperature - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / (sum || 1));
}

function renormalizeRows(rows) {
  return rows.map((row) => {
    const sum = row.reduce((acc, value) => acc + value, 0);
    return row.map((value) => value / (sum || 1));
  });
}
