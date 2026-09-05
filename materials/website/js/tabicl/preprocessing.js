import { ensure2D } from "./tensor.js";

export class LabelEncoder {
  fit(values) {
    this.classes = Array.from(new Set(values)).sort();
    this.index = new Map(this.classes.map((value, idx) => [value, idx]));
    return this;
  }

  transform(values) {
    return values.map((value) => {
      if (!this.index.has(value)) throw new Error(`Unknown class label: ${value}`);
      return this.index.get(value);
    });
  }

  inverseTransform(indices) {
    return indices.map((idx) => this.classes[idx]);
  }

  fitTransform(values) {
    this.fit(values);
    return this.transform(values);
  }
}

export class StandardScaler1D {
  fit(values) {
    this.mean = values.reduce((acc, value) => acc + Number(value), 0) / values.length;
    const variance = values.reduce((acc, value) => acc + (Number(value) - this.mean) ** 2, 0) / values.length;
    this.scale = Math.sqrt(variance) || 1;
    return this;
  }

  transform(values) {
    return values.map((value) => (Number(value) - this.mean) / this.scale);
  }

  inverseTransform(values) {
    return values.map((value) => Number(value) * this.scale + this.mean);
  }

  fitTransform(values) {
    this.fit(values);
    return this.transform(values);
  }
}

export class TransformToNumerical {
  constructor({ schema = null } = {}) {
    this.schema = schema;
  }

  fit(X) {
    ensure2D(X);
    const nCols = X[0].length;
    this.columns = [];
    for (let col = 0; col < nCols; col++) {
      const declared = this.schema?.[col]?.type;
      const values = X.map((row) => row[col]);
      const numeric = declared ? declared === "number" : values.every((value) => value == null || value === "" || Number.isFinite(Number(value)));
      if (numeric) {
        const nums = values.filter((value) => value != null && value !== "").map((value) => Number(value)).filter(Number.isFinite);
        const mean = nums.length ? nums.reduce((acc, value) => acc + value, 0) / nums.length : 0;
        this.columns.push({ type: "number", mean });
      } else {
        const categories = Array.from(new Set(values.filter((value) => value != null && value !== "")))
          .sort((a, b) => String(a).localeCompare(String(b)));
        this.columns.push({ type: "category", categories, index: new Map(categories.map((value, idx) => [value, idx])) });
      }
    }
    return this;
  }

  transform(X) {
    ensure2D(X);
    if (X[0].length !== this.columns.length) throw new Error("Feature count changed after fit");
    return X.map((row) => row.map((value, col) => {
      const spec = this.columns[col];
      if (spec.type === "number") {
        if (value == null || value === "") return spec.mean;
        const number = Number(value);
        return Number.isFinite(number) ? number : spec.mean;
      }
      return spec.index.has(value) ? spec.index.get(value) : -1;
    }));
  }

  fitTransform(X) {
    this.fit(X);
    return this.transform(X);
  }
}

export class UniqueFeatureFilter {
  constructor({ threshold = 1 } = {}) {
    this.threshold = threshold;
  }

  fit(X) {
    ensure2D(X);
    const nRows = X.length;
    const nCols = X[0].length;
    this.keep = [];
    for (let col = 0; col < nCols; col++) {
      const unique = new Set(X.map((row) => row[col]));
      this.keep.push(nRows <= this.threshold || unique.size > this.threshold);
    }
    if (!this.keep.some(Boolean)) this.keep.fill(true);
    return this;
  }

  transform(X) {
    ensure2D(X);
    return X.map((row) => row.filter((_, col) => this.keep[col]));
  }

  fitTransform(X) {
    this.fit(X);
    return this.transform(X);
  }
}

export class PreprocessingPipeline {
  constructor({ normalizationMethod = "none", outlierThreshold = 4, randomState = 42 } = {}) {
    this.normalizationMethod = normalizationMethod;
    this.outlierThreshold = outlierThreshold;
    this.randomState = randomState;
  }

  fit(X) {
    ensure2D(X);
    this.mean = columnMeans(X);
    this.std = columnStd(X, this.mean).map((value) => value + 1e-6);
    const standardized = applyStandard(X, this.mean, this.std);
    this.norm = fitNormalizer(standardized, this.normalizationMethod);
    const normalized = this.applyNormalizer(standardized);
    this.clip = fitClip(normalized, this.outlierThreshold);
    return this;
  }

  transform(X) {
    ensure2D(X);
    let standardized = applyStandard(X, this.mean, this.std);
    let normalized = this.applyNormalizer(standardized);
    if (this.normalizationMethod === "power" && normalized.some((row) => row.some((value) => !Number.isFinite(value)))) {
      standardized = standardized.map((row) => row.map(
        (value, col) => Math.min(this.norm.max[col], Math.max(this.norm.min[col], value)),
      ));
      normalized = this.applyNormalizer(standardized);
    }
    return applyClip(normalized, this.clip);
  }

  fitTransform(X) {
    this.fit(X);
    return this.transform(X);
  }

  applyNormalizer(X) {
    if (this.normalizationMethod === "none") return X.map((row) => row.slice());
    if (this.normalizationMethod === "robust") {
      return X.map((row) => row.map((value, col) => (value - this.norm.median[col]) / (this.norm.iqr[col] || 1)));
    }
    if (this.normalizationMethod === "power") {
      return X.map((row) => row.map((value, col) => (
        yeoJohnson(value, this.norm.lambda[col]) - this.norm.mean[col]
      ) / this.norm.scale[col]));
    }
    if (this.normalizationMethod === "quantile" || this.normalizationMethod === "quantile_rtdl") {
      return X.map((row) => row.map((value, col) => quantileToNormal(value, this.norm.sorted[col])));
    }
    throw new Error(`Unsupported normalization method: ${this.normalizationMethod}`);
  }
}

function columnMeans(X) {
  const nCols = X[0].length;
  return Array.from(
    { length: nCols },
    (_, col) => numpySum(X.map((row) => Number(row[col]))) / X.length,
  );
}

function columnStd(X, mean) {
  return Array.from({ length: mean.length }, (_, col) => Math.sqrt(numpySum(
    X.map((row) => (Number(row[col]) - mean[col]) ** 2),
  ) / X.length));
}

function applyStandard(X, mean, std) {
  return X.map((row) => row.map((value, col) => Math.min(
    100,
    Math.max(-100, (Number(value) - mean[col]) / std[col]),
  )));
}

function percentile(sorted, p) {
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * p;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function fitNormalizer(X, method) {
  if (method === "none") return {};
  const columns = X[0].map((_, col) => X.map((row) => row[col]).sort((a, b) => a - b));
  if (method === "robust") {
    return {
      median: columns.map((values) => percentile(values, 0.5)),
      iqr: columns.map((values) => percentile(values, 0.75) - percentile(values, 0.25)),
    };
  }
  if (method === "power") {
    const unsortedColumns = X[0].map((_, col) => X.map((row) => row[col]));
    const lambda = unsortedColumns.map((values) => (
      isConstantFeature(values) ? 1 : fitYeoJohnsonLambda(values)
    ));
    const transformed = X.map((row) => row.map((value, col) => yeoJohnson(value, lambda[col])));
    const mean = columnMeans(transformed);
    const variance = columnVariance(transformed, mean);
    const scale = variance.map((value, col) => (
      isConstantVariance(value, mean[col], X.length) ? 1 : Math.sqrt(value)
    ));
    return {
      lambda,
      mean,
      scale,
      min: unsortedColumns.map((values) => Math.min(...values)),
      max: unsortedColumns.map((values) => Math.max(...values)),
    };
  }
  if (method === "quantile" || method === "quantile_rtdl") return { sorted: columns };
  throw new Error(`Unsupported normalization method: ${method}`);
}

function fitClip(X, threshold) {
  let mean = columnMeans(X);
  let std = columnSampleStd(X, mean).map((value) => Math.max(value, 1e-6));
  const cleanColumns = X[0].map((_, col) => X
    .map((row) => row[col])
    .filter((value) => value >= mean[col] - threshold * std[col] && value <= mean[col] + threshold * std[col]));
  mean = cleanColumns.map((values) => numpySum(values) / values.length);
  std = cleanColumns.map((values, col) => Math.max(sampleStd(values, mean[col]), 1e-6));
  return {
    lower: mean.map((value, col) => value - threshold * std[col]),
    upper: mean.map((value, col) => value + threshold * std[col]),
  };
}

function yeoJohnson(x, lambda) {
  if (x >= 0) {
    return Math.abs(lambda) < Number.EPSILON
      ? Math.log1p(x)
      : (Math.pow(x + 1, lambda) - 1) / lambda;
  }
  return Math.abs(lambda - 2) <= Number.EPSILON
    ? -Math.log1p(-x)
    : -(Math.pow(1 - x, 2 - lambda) - 1) / (2 - lambda);
}

function columnVariance(X, mean) {
  return Array.from({ length: mean.length }, (_, col) => numpySum(
    X.map((row) => (Number(row[col]) - mean[col]) ** 2),
  ) / X.length);
}

function columnSampleStd(X, mean) {
  const denominator = X.length > 1 ? X.length - 1 : X.length;
  return Array.from({ length: mean.length }, (_, col) => Math.sqrt(numpySum(
    X.map((row) => (Number(row[col]) - mean[col]) ** 2),
  ) / denominator));
}

function sampleStd(values, mean) {
  const denominator = values.length > 1 ? values.length - 1 : values.length;
  const variance = numpySum(values.map((value) => (value - mean) ** 2)) / denominator;
  return Math.sqrt(variance);
}

function isConstantVariance(variance, mean, nSamples) {
  const upperBound = nSamples * Number.EPSILON * variance
    + (nSamples * mean * Number.EPSILON) ** 2;
  return variance <= upperBound;
}

function isConstantFeature(values) {
  const mean = numpySum(values) / values.length;
  const variance = numpySum(values.map((value) => (value - mean) ** 2)) / values.length;
  return isConstantVariance(variance, mean, values.length);
}

function fitYeoJohnsonLambda(values) {
  const logJacobian = numpySum(values.map(
    (value) => Math.sign(value) * Math.log1p(Math.abs(value)),
  ));
  const objective = (lambda) => {
    const transformed = values.map((value) => yeoJohnson(value, lambda));
    const mean = numpySum(transformed) / transformed.length;
    const variance = numpySum(transformed.map((value) => (value - mean) ** 2)) / transformed.length;
    if (variance < 2.2250738585072014e-308) return Infinity;
    const logLikelihood = -values.length / 2 * Math.log(variance) + (lambda - 1) * logJacobian;
    return -logLikelihood;
  };
  return brent(objective, -2, 2);
}

function bracketMinimum(objective, xa, xb) {
  const gold = 1.618034;
  const growLimit = 110;
  const verySmall = 1e-21;
  let fa = objective(xa);
  let fb = objective(xb);
  if (fa < fb) {
    [xa, xb] = [xb, xa];
    [fa, fb] = [fb, fa];
  }
  let xc = xb + gold * (xb - xa);
  let fc = objective(xc);
  let iterations = 0;
  while (fc < fb) {
    const tmp1 = (xb - xa) * (fb - fc);
    const tmp2 = (xb - xc) * (fb - fa);
    const delta = tmp2 - tmp1;
    const denominator = Math.abs(delta) < verySmall ? 2 * verySmall : 2 * delta;
    let w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denominator;
    const wLimit = xb + growLimit * (xc - xb);
    if (iterations++ > 1000) throw new Error("Unable to bracket Yeo-Johnson lambda");
    let fw;
    if ((w - xc) * (xb - w) > 0) {
      fw = objective(w);
      if (fw < fc) return { xa: xb, xb: w, xc, fa: fb, fb: fw, fc };
      if (fw > fb) return { xa, xb, xc: w, fa, fb, fc: fw };
      w = xc + gold * (xc - xb);
      fw = objective(w);
    } else if ((w - wLimit) * (wLimit - xc) >= 0) {
      w = wLimit;
      fw = objective(w);
    } else if ((w - wLimit) * (xc - w) > 0) {
      fw = objective(w);
      if (fw < fc) {
        xb = xc;
        xc = w;
        w = xc + gold * (xc - xb);
        fb = fc;
        fc = fw;
        fw = objective(w);
      }
    } else {
      w = xc + gold * (xc - xb);
      fw = objective(w);
    }
    xa = xb;
    xb = xc;
    xc = w;
    fa = fb;
    fb = fc;
    fc = fw;
  }
  return { xa, xb, xc, fa, fb, fc };
}

function brent(objective, xa, xb) {
  const bracket = bracketMinimum(objective, xa, xb);
  let a = Math.min(bracket.xa, bracket.xc);
  let b = Math.max(bracket.xa, bracket.xc);
  let x = bracket.xb;
  let w = x;
  let v = x;
  let fx = bracket.fb;
  let fw = fx;
  let fv = fx;
  let deltaX = 0;
  let step = 0;
  for (let iteration = 0; iteration < 500; iteration++) {
    const tolerance1 = 1.48e-8 * Math.abs(x) + 1e-11;
    const tolerance2 = 2 * tolerance1;
    const midpoint = 0.5 * (a + b);
    if (Math.abs(x - midpoint) < tolerance2 - 0.5 * (b - a)) return x;
    if (Math.abs(deltaX) <= tolerance1) {
      deltaX = x >= midpoint ? a - x : b - x;
      step = 0.3819660 * deltaX;
    } else {
      const temp1 = (x - w) * (fx - fv);
      let temp2 = (x - v) * (fx - fw);
      let parabola = (x - v) * temp2 - (x - w) * temp1;
      temp2 = 2 * (temp2 - temp1);
      if (temp2 > 0) parabola = -parabola;
      temp2 = Math.abs(temp2);
      const previousDelta = deltaX;
      deltaX = step;
      if (parabola > temp2 * (a - x)
        && parabola < temp2 * (b - x)
        && Math.abs(parabola) < Math.abs(0.5 * temp2 * previousDelta)) {
        step = parabola / temp2;
        const candidate = x + step;
        if (candidate - a < tolerance2 || b - candidate < tolerance2) {
          step = midpoint - x >= 0 ? tolerance1 : -tolerance1;
        }
      } else {
        deltaX = x >= midpoint ? a - x : b - x;
        step = 0.3819660 * deltaX;
      }
    }
    const candidate = x + (Math.abs(step) < tolerance1
      ? (step >= 0 ? tolerance1 : -tolerance1)
      : step);
    const candidateValue = objective(candidate);
    if (candidateValue > fx) {
      if (candidate < x) a = candidate;
      else b = candidate;
      if (candidateValue <= fw || w === x) {
        v = w;
        w = candidate;
        fv = fw;
        fw = candidateValue;
      } else if (candidateValue <= fv || v === x || v === w) {
        v = candidate;
        fv = candidateValue;
      }
    } else {
      if (candidate >= x) a = x;
      else b = x;
      v = w;
      w = x;
      x = candidate;
      fv = fw;
      fw = fx;
      fx = candidateValue;
    }
  }
  throw new Error("Yeo-Johnson lambda optimization did not converge");
}

function applyClip(X, clip) {
  return X.map((row) => row.map((value, col) => {
    const lowerAdjusted = -Math.log1p(Math.abs(value)) + clip.lower[col];
    const upperAdjusted = Math.log1p(Math.abs(value)) + clip.upper[col];
    return Math.min(upperAdjusted, Math.max(lowerAdjusted, value));
  }));
}

// NumPy's float64 reductions use an eight-lane pairwise sum for short axes.
function numpySum(values, start = 0, length = values.length) {
  if (length < 8) {
    let result = -0;
    for (let index = 0; index < length; index++) result += values[start + index];
    return result;
  }
  if (length <= 128) {
    const lanes = values.slice(start, start + 8);
    let index = 8;
    for (; index <= length - 8; index += 8) {
      for (let lane = 0; lane < 8; lane++) lanes[lane] += values[start + index + lane];
    }
    let result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
      + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (; index < length; index++) result += values[start + index];
    return result;
  }
  let half = Math.floor(length / 2);
  half -= half % 8;
  return numpySum(values, start, half) + numpySum(values, start + half, length - half);
}

function quantileToNormal(value, sorted) {
  let count = 0;
  while (count < sorted.length && sorted[count] <= value) count++;
  const p = Math.min(1 - 1e-7, Math.max(1e-7, count / Math.max(1, sorted.length)));
  return inverseNormal(p);
}

function inverseNormal(p) {
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
