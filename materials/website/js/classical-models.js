function sigmoid(value) {
  if (value >= 0) {
    const exp = Math.exp(-value);
    return 1 / (1 + exp);
  }
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

function featureStats(X) {
  const dimensions = X[0]?.length || 0;
  const mean = Array(dimensions).fill(0);
  const scale = Array(dimensions).fill(0);
  X.forEach((row) => row.forEach((value, feature) => { mean[feature] += value; }));
  mean.forEach((_, feature) => { mean[feature] /= Math.max(1, X.length); });
  X.forEach((row) => row.forEach((value, feature) => {
    scale[feature] += (value - mean[feature]) ** 2;
  }));
  scale.forEach((_, feature) => {
    scale[feature] = Math.sqrt(scale[feature] / Math.max(1, X.length)) || 1;
  });
  return { mean, scale };
}

export function fitLogisticRegression(X, y, options = {}) {
  if (!X.length || X.length !== y.length) throw new Error("Linear model requires aligned X and y.");
  const { mean, scale } = featureStats(X);
  const weights = Array(mean.length + 1).fill(0);
  const steps = options.steps ?? 240;
  const learningRate = options.learningRate ?? 0.16;
  const l2 = options.l2 ?? 0.012;

  for (let step = 0; step < steps; step += 1) {
    const gradient = Array(weights.length).fill(0);
    X.forEach((row, index) => {
      let score = weights[0];
      for (let feature = 0; feature < mean.length; feature += 1) {
        score += weights[feature + 1] * ((row[feature] - mean[feature]) / scale[feature]);
      }
      const error = sigmoid(score) - y[index];
      gradient[0] += error;
      for (let feature = 0; feature < mean.length; feature += 1) {
        gradient[feature + 1] += error * ((row[feature] - mean[feature]) / scale[feature]);
      }
    });
    weights[0] -= learningRate * gradient[0] / X.length;
    for (let feature = 1; feature < weights.length; feature += 1) {
      const regularized = gradient[feature] / X.length + l2 * weights[feature];
      weights[feature] -= learningRate * regularized;
    }
  }
  return { mean, scale, weights };
}

export function predictLogisticRegression(model, row) {
  let score = model.weights[0];
  for (let feature = 0; feature < model.mean.length; feature += 1) {
    score += model.weights[feature + 1]
      * ((row[feature] - model.mean[feature]) / model.scale[feature]);
  }
  return sigmoid(score);
}

function leaf(indices, y) {
  const positives = indices.reduce((sum, index) => sum + y[index], 0);
  return {
    probability: (positives + 1) / (indices.length + 2),
    size: indices.length,
  };
}

function gini(indices, y) {
  if (!indices.length) return 0;
  const positives = indices.reduce((sum, index) => sum + y[index], 0);
  const p = positives / indices.length;
  return 2 * p * (1 - p);
}

export function fitDecisionTree(X, y, options = {}) {
  if (!X.length || X.length !== y.length) throw new Error("Decision tree requires aligned X and y.");
  const maxDepth = options.maxDepth ?? 3;
  const minLeaf = options.minLeaf ?? 2;

  function build(indices, depth) {
    const base = leaf(indices, y);
    if (depth >= maxDepth || indices.length < minLeaf * 2 || base.probability <= 1 / (indices.length + 2)
        || base.probability >= (indices.length + 1) / (indices.length + 2)) {
      return base;
    }

    let best = null;
    for (let feature = 0; feature < X[0].length; feature += 1) {
      const values = [...new Set(indices.map((index) => X[index][feature]))].sort((a, b) => a - b);
      for (let position = 1; position < values.length; position += 1) {
        const threshold = (values[position - 1] + values[position]) / 2;
        const left = indices.filter((index) => X[index][feature] <= threshold);
        const right = indices.filter((index) => X[index][feature] > threshold);
        if (left.length < minLeaf || right.length < minLeaf) continue;
        const impurity = (left.length * gini(left, y) + right.length * gini(right, y)) / indices.length;
        if (!best || impurity < best.impurity) {
          best = { feature, threshold, left, right, impurity };
        }
      }
    }
    if (!best) return base;
    return {
      feature: best.feature,
      threshold: best.threshold,
      probability: base.probability,
      size: indices.length,
      left: build(best.left, depth + 1),
      right: build(best.right, depth + 1),
    };
  }

  return build(X.map((_, index) => index), 0);
}

export function predictDecisionTree(model, row) {
  let node = model;
  while (node.left && node.right) {
    node = row[node.feature] <= node.threshold ? node.left : node.right;
  }
  return node.probability;
}
