export function product(shape) {
  return shape.reduce((acc, value) => acc * value, 1);
}

export function f16ToF32(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits & 0x7c00) >> 10;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

export async function verifySha256(buffer, expected, label = "binary artifact") {
  if (!/^[0-9a-f]{64}$/i.test(expected || "")) {
    throw new Error(`${label} has no valid SHA-256 digest`);
  }
  if (!globalThis.crypto?.subtle) {
    throw new Error(`Web Crypto is required to verify ${label}`);
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", buffer);
  const actual = Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
  if (actual !== expected.toLowerCase()) {
    throw new Error(`${label} SHA-256 mismatch: expected ${expected}, got ${actual}`);
  }
  return actual;
}

export function loadFlatTensors(manifest, buffer, { dequantize = true } = {}) {
  const view = new DataView(buffer);
  const tensors = new Map();
  for (const item of manifest.tensors) {
    const size = product(item.shape);
    if (!dequantize && item.qtype === "int8") {
      const scales = new Float32Array(item.out_ch);
      for (let i = 0; i < item.out_ch; i++) scales[i] = view.getFloat32(item.offset + i * 4, true);
      const offset = item.offset + item.out_ch * 4;
      tensors.set(item.name, {
        shape: item.shape,
        qtype: item.qtype,
        scales,
        data: new Int8Array(buffer, offset, size),
      });
      continue;
    }

    const data = new Float32Array(size);
    if (item.qtype === "fp32") {
      for (let i = 0; i < size; i++) data[i] = view.getFloat32(item.offset + i * 4, true);
    } else if (item.qtype === "int8") {
      const outCh = item.out_ch;
      const inner = size / outCh;
      const dataOffset = item.offset + outCh * 4;
      for (let row = 0; row < outCh; row++) {
        const scale = view.getFloat32(item.offset + row * 4, true);
        const rowOffset = row * inner;
        for (let col = 0; col < inner; col++) {
          data[rowOffset + col] = view.getInt8(dataOffset + rowOffset + col) * scale;
        }
      }
    } else if (item.qtype === "fp16") {
      for (let i = 0; i < size; i++) data[i] = f16ToF32(view.getUint16(item.offset + i * 2, true));
    } else {
      throw new Error(`Unsupported tensor encoding: ${item.qtype}`);
    }
    tensors.set(item.name, { shape: item.shape, qtype: "fp32", data });
  }
  return tensors;
}

export async function loadManifestAndWeights(baseUrl, { fetchImpl = fetch, dequantize = true } = {}) {
  const manifest = await (await fetchImpl(`${baseUrl.replace(/\/$/, "")}/manifest.json`)).json();
  const binaryName = manifest.binary || manifest.bin || "tabicl.bin";
  const buffer = await (await fetchImpl(`${baseUrl.replace(/\/$/, "")}/${binaryName}`)).arrayBuffer();
  await verifySha256(buffer, manifest.binary_sha256, binaryName);
  return { manifest, tensors: loadFlatTensors(manifest, buffer, { dequantize }) };
}

export function linear(input, rows, inDim, weight, outDim, bias = null) {
  const output = new Float32Array(rows * outDim);
  for (let row = 0; row < rows; row++) {
    const inputOffset = row * inDim;
    const outputOffset = row * outDim;
    for (let out = 0; out < outDim; out++) {
      let acc = bias ? bias[out] : 0;
      const weightOffset = out * inDim;
      for (let k = 0; k < inDim; k++) acc += input[inputOffset + k] * weight[weightOffset + k];
      output[outputOffset + out] = acc;
    }
  }
  return output;
}

export function layerNorm(input, rows, dim, gamma, beta, eps = 1e-5) {
  const output = new Float32Array(rows * dim);
  for (let row = 0; row < rows; row++) {
    const offset = row * dim;
    let mean = 0;
    for (let col = 0; col < dim; col++) mean += input[offset + col];
    mean /= dim;
    let variance = 0;
    for (let col = 0; col < dim; col++) {
      const centered = input[offset + col] - mean;
      variance += centered * centered;
    }
    const inv = 1 / Math.sqrt(variance / dim + eps);
    for (let col = 0; col < dim; col++) output[offset + col] = (input[offset + col] - mean) * inv * gamma[col] + beta[col];
  }
  return output;
}

export function softmax(values, temperature = 1) {
  const output = new Float32Array(values.length);
  let max = -Infinity;
  for (const value of values) max = Math.max(max, value / temperature);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    output[i] = Math.exp(values[i] / temperature - max);
    sum += output[i];
  }
  for (let i = 0; i < output.length; i++) output[i] /= sum || 1;
  return Array.from(output);
}

export function softmaxRows(matrix, nRows, nCols, temperature = 1) {
  const rows = [];
  for (let row = 0; row < nRows; row++) {
    rows.push(softmax(matrix.slice(row * nCols, (row + 1) * nCols), temperature));
  }
  return rows;
}

export function ensure2D(X, name = "X") {
  if (!Array.isArray(X) || X.length === 0 || !Array.isArray(X[0])) {
    throw new Error(`${name} must be a non-empty two-dimensional array`);
  }
  const width = X[0].length;
  if (width === 0) throw new Error(`${name} must have at least one column`);
  for (const row of X) {
    if (!Array.isArray(row) || row.length !== width) throw new Error(`${name} must be rectangular`);
  }
  return { rows: X.length, cols: width };
}

export function flatten2D(X) {
  const { rows, cols } = ensure2D(X);
  const data = new Float32Array(rows * cols);
  for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) data[row * cols + col] = Number(X[row][col]);
  return { data, rows, cols };
}
