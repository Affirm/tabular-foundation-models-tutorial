/* nanotabicl.js — client-side forward pass for the converted TabICL v2 core.
 *
 * Checkpoint-compatible adaptation of the nanoTabICL forward graph, verified
 * against the official `tabicl` 2.0.2 checkpoint. It is not a literal port of
 * the standalone nanoTabICL weights or preprocessing. Loads the int8 / fp16
 * browser artifact. Batch size 1. No dependencies.
 *
 * Preprocessing (must match training): z-score by TRAIN mean/std, then repeated
 * feature grouping with the checkpoint's offsets 2^i = {1,2,4} (circular).
 * The standalone nanoTabICL source instead uses 2^i - 1 = {0,1,3}. RoPE is used
 * only in the row stage; QASSMax only in column tfm1 and the ICL blocks.
 */
import { toNanoClassifierManifest } from "./core.js";
import { verifySha256 } from "./tensor.js";

/* ---------------- weight store ---------------- */
function dequantize(manifest, buffer) {
  // Use DataView throughout: tensor offsets are not guaranteed 4-byte aligned,
  // so typed-array-on-buffer views (which require alignment) can't be used.
  const dv = new DataView(buffer);
  const W = new Map();
  for (const m of manifest.tensors) {
    const shape = m.shape, numel = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(numel);
    if (m.qtype === "int8") {
      const oc = m.out_ch, inner = numel / oc, dataOff = m.offset + 4 * oc;
      for (let r = 0; r < oc; r++) {
        const s = dv.getFloat32(m.offset + 4 * r, true), base = r * inner;
        for (let c = 0; c < inner; c++) data[base + c] = dv.getInt8(dataOff + base + c) * s;
      }
    } else {
      for (let i = 0; i < numel; i++) data[i] = f16(dv.getUint16(m.offset + 2 * i, true));
    }
    W.set(m.name, { data, shape });
  }
  return W;
}

function f16(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/* ---------------- primitive ops (row-major Float32Array) ---------------- */
// y[n,dout] = x[n,din] · W[dout,din]^T + b
function linear(x, n, din, W, dout, b) {
  const y = new Float32Array(n * dout);
  for (let i = 0; i < n; i++) {
    const xo = i * din, yo = i * dout;
    for (let o = 0; o < dout; o++) {
      let acc = b ? b[o] : 0; const wo = o * din;
      for (let k = 0; k < din; k++) acc += x[xo + k] * W[wo + k];
      y[yo + o] = acc;
    }
  }
  return y;
}

function layernorm(x, n, d, g, b, eps = 1e-5) {
  const y = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    const o = i * d; let mean = 0;
    for (let k = 0; k < d; k++) mean += x[o + k];
    mean /= d;
    let v = 0;
    for (let k = 0; k < d; k++) { const t = x[o + k] - mean; v += t * t; }
    v /= d; const inv = 1 / Math.sqrt(v + eps);
    for (let k = 0; k < d; k++) y[o + k] = (x[o + k] - mean) * inv * g[k] + b[k];
  }
  return y;
}

function erf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
function geluInplace(x) {
  for (let i = 0; i < x.length; i++) { const v = x[i]; x[i] = 0.5 * v * (1 + erf(v * 0.7071067811865476)); }
  return x;
}
function addInto(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; return a; }

// mlp: Linear(d,h) -> GELU -> Linear(h,d)
function mlp(x, n, d, W, pfx) {
  const w0 = W.get(pfx + ".mlp.0.weight"), b0 = W.get(pfx + ".mlp.0.bias");
  const w2 = W.get(pfx + ".mlp.2.weight"), b2 = W.get(pfx + ".mlp.2.bias");
  const h = w0.shape[0];
  let t = linear(x, n, d, w0.data, h, b0.data);
  geluInplace(t);
  return linear(t, n, h, w2.data, w2.shape[0], b2.data);
}

/* ---------------- RoPE ---------------- */
function ropeTables(seqLen, half, theta) {
  const cos = new Float32Array(seqLen * half), sin = new Float32Array(seqLen * half);
  for (let p = 0; p < seqLen; p++) for (let j = 0; j < half; j++) {
    const invf = Math.pow(theta, -j / half), a = p * invf;
    cos[p * half + j] = Math.cos(a); sin[p * half + j] = Math.sin(a);
  }
  return { cos, sin };
}
// rotate one head's sequence in place: xh[L,hd]
function applyRope(xh, L, hd, tab) {
  const half = hd >> 1;
  for (let p = 0; p < L; p++) {
    const o = p * hd, co = p * half;
    for (let j = 0; j < half; j++) {
      const c = tab.cos[co + j], s = tab.sin[co + j];
      const x1 = xh[o + j], x2 = xh[o + half + j];
      xh[o + j] = x1 * c - x2 * s;
      xh[o + half + j] = x1 * s + x2 * c;
    }
  }
}

/* ---------------- QASSMax ---------------- */
// scales Q[Lq,E] in place, using head layout (H, hd) and context length n
function qassmax(Q, Lq, E, H, hd, n, W, pfx) {
  // base_mlp(log n) -> vector length E
  const bw0 = W.get(pfx + ".ssmax_layer.base_mlp.0.weight"), bb0 = W.get(pfx + ".ssmax_layer.base_mlp.0.bias");
  const bw2 = W.get(pfx + ".ssmax_layer.base_mlp.2.weight"), bb2 = W.get(pfx + ".ssmax_layer.base_mlp.2.bias");
  const logn = new Float32Array([Math.log(Math.max(1, n))]);
  let bh = linear(logn, 1, 1, bw0.data, bw0.shape[0], bb0.data); geluInplace(bh);
  const base = linear(bh, 1, bw0.shape[0], bw2.data, E, bb2.data); // [E]

  // query_mlp per token: hd -> hd, applied per head slice
  const qw0 = W.get(pfx + ".ssmax_layer.query_mlp.0.weight"), qb0 = W.get(pfx + ".ssmax_layer.query_mlp.0.bias");
  const qw2 = W.get(pfx + ".ssmax_layer.query_mlp.2.weight"), qb2 = W.get(pfx + ".ssmax_layer.query_mlp.2.bias");
  const qh = qw0.shape[0];
  for (let h = 0; h < H; h++) {
    // gather this head's Q rows [Lq,hd]
    const sub = new Float32Array(Lq * hd);
    for (let p = 0; p < Lq; p++) for (let d = 0; d < hd; d++) sub[p * hd + d] = Q[p * E + h * hd + d];
    let t = linear(sub, Lq, hd, qw0.data, qh, qb0.data); geluInplace(t);
    const mod = linear(t, Lq, qh, qw2.data, hd, qb2.data); // [Lq,hd]
    for (let p = 0; p < Lq; p++) for (let d = 0; d < hd; d++) {
      const idx = p * E + h * hd + d;
      Q[idx] = base[h * hd + d] * (1 + Math.tanh(mod[p * hd + d])) * Q[idx];
    }
  }
}

/* ---------------- one TransformerBlock ---------------- */
// q:[Lq,E]; kv:[Lkv,E] or null(=self). opts: {qMax, kvMax, ssmax, rope(theta or null), nHeads}
function block(W, pfx, q, Lq, kv, Lkv, E, nHeads, opts) {
  const g = W.get(pfx + ".ln_attn.weight").data, b = W.get(pfx + ".ln_attn.bias").data;
  const xFull = q;                       // residual source (pre-norm keeps raw)
  let qn = layernorm(q, Lq, E, g, b);
  let kvn = kv ? layernorm(kv, Lkv, E, g, b) : qn;
  let LkvE = kv ? Lkv : Lq;
  // slice kv to kvMax (context = train rows)
  if (opts.kvMax != null && opts.kvMax < LkvE) { kvn = kvn.subarray(0, opts.kvMax * E); LkvE = opts.kvMax; }
  // slice q to qMax (keep only first qMax query tokens, e.g. CLS)
  let x = xFull, LqE = Lq;
  if (opts.qMax != null && opts.qMax < Lq) { x = xFull.subarray(0, opts.qMax * E); qn = qn.subarray(0, opts.qMax * E); LqE = opts.qMax; }

  const ipw = W.get(pfx + ".in_proj_weight").data, ipb = W.get(pfx + ".in_proj_bias").data;
  const Wq = ipw.subarray(0, E * E), Wk = ipw.subarray(E * E, 2 * E * E), Wv = ipw.subarray(2 * E * E, 3 * E * E);
  const bq = ipb.subarray(0, E), bk = ipb.subarray(E, 2 * E), bv = ipb.subarray(2 * E, 3 * E);
  const Q = linear(qn, LqE, E, Wq, E, bq);
  const K = linear(kvn, LkvE, E, Wk, E, bk);
  const V = linear(kvn, LkvE, E, Wv, E, bv);

  const hd = E / nHeads;
  if (opts.ssmax) qassmax(Q, LqE, E, nHeads, hd, LkvE, W, pfx);

  let ropeTab = null;
  if (opts.rope) {
    const half = hd >> 1;
    ropeTab = ropeTables(Math.max(LqE, LkvE), half, opts.rope);
  }

  const O = new Float32Array(LqE * E);
  const scale = 1 / Math.sqrt(hd);
  const qh = new Float32Array(LqE * hd), kh = new Float32Array(LkvE * hd), vh = new Float32Array(LkvE * hd);
  for (let h = 0; h < nHeads; h++) {
    for (let p = 0; p < LqE; p++) for (let d = 0; d < hd; d++) qh[p * hd + d] = Q[p * E + h * hd + d];
    for (let p = 0; p < LkvE; p++) for (let d = 0; d < hd; d++) { kh[p * hd + d] = K[p * E + h * hd + d]; vh[p * hd + d] = V[p * E + h * hd + d]; }
    if (ropeTab) { applyRope(qh.subarray(0, LqE * hd), LqE, hd, ropeTab); applyRope(kh.subarray(0, LkvE * hd), LkvE, hd, ropeTab); }
    const scores = new Float32Array(LkvE);
    for (let p = 0; p < LqE; p++) {
      let mx = -Infinity;
      for (let j = 0; j < LkvE; j++) {
        let acc = 0; const qo = p * hd, ko = j * hd;
        for (let d = 0; d < hd; d++) acc += qh[qo + d] * kh[ko + d];
        acc *= scale; scores[j] = acc; if (acc > mx) mx = acc;
      }
      let sum = 0;
      for (let j = 0; j < LkvE; j++) { const e = Math.exp(scores[j] - mx); scores[j] = e; sum += e; }
      const inv = 1 / sum, oo = p * E + h * hd;
      for (let d = 0; d < hd; d++) {
        let acc = 0;
        for (let j = 0; j < LkvE; j++) acc += scores[j] * vh[j * hd + d];
        O[oo + d] = acc * inv;
      }
    }
  }
  const opw = W.get(pfx + ".out_proj.weight").data, opb = W.get(pfx + ".out_proj.bias").data;
  const attnOut = linear(O, LqE, E, opw, E, opb);

  const xs = x.slice(0, LqE * E);        // copy residual (may be subarray)
  addInto(xs, attnOut);
  const gm = W.get(pfx + ".ln_mlp.weight").data, bm = W.get(pfx + ".ln_mlp.bias").data;
  const mo = mlp(layernorm(xs, LqE, E, gm, bm), LqE, E, W, pfx);
  addInto(xs, mo);
  return { data: xs, n: LqE };
}

/* ---------------- cached-KV fast path (for a fixed context) ----------------
 * The context (train rows) is fixed while a user drags a query or we sweep a
 * grid, so we project its K/V once and reuse it. Only valid where the block has
 * no RoPE (column tfm2 + all ICL blocks) — which is exactly the query path.
 */
function computeKV(W, pfx, kv, Lkv, E, nHeads, kvMax) {
  const g = W.get(pfx + ".ln_attn.weight").data, b = W.get(pfx + ".ln_attn.bias").data;
  let kvn = layernorm(kv, Lkv, E, g, b), L = Lkv;
  if (kvMax != null && kvMax < L) { kvn = kvn.subarray(0, kvMax * E); L = kvMax; }
  const ipw = W.get(pfx + ".in_proj_weight").data, ipb = W.get(pfx + ".in_proj_bias").data;
  const Wk = ipw.subarray(E * E, 2 * E * E), Wv = ipw.subarray(2 * E * E, 3 * E * E);
  const bk = ipb.subarray(E, 2 * E), bv = ipb.subarray(2 * E, 3 * E);
  const K = linear(kvn, L, E, Wk, E, bk), V = linear(kvn, L, E, Wv, E, bv);
  const hd = E / nHeads;
  const kh = [], vh = [];
  for (let h = 0; h < nHeads; h++) {
    const ka = new Float32Array(L * hd), va = new Float32Array(L * hd);
    for (let p = 0; p < L; p++) for (let d = 0; d < hd; d++) { ka[p * hd + d] = K[p * E + h * hd + d]; va[p * hd + d] = V[p * E + h * hd + d]; }
    kh.push(ka); vh.push(va);
  }
  return { kh, vh, kvLen: L };
}

// q:[Lq,E] attends precomputed per-head K/V (length kvLen). No RoPE.
function blockCachedKV(W, pfx, q, Lq, E, nHeads, cache, opts) {
  const g = W.get(pfx + ".ln_attn.weight").data, b = W.get(pfx + ".ln_attn.bias").data;
  const xFull = q;
  let qn = layernorm(q, Lq, E, g, b), LqE = Lq, x = xFull;
  if (opts.qMax != null && opts.qMax < Lq) { qn = qn.subarray(0, opts.qMax * E); x = xFull.subarray(0, opts.qMax * E); LqE = opts.qMax; }
  const ipw = W.get(pfx + ".in_proj_weight").data, ipb = W.get(pfx + ".in_proj_bias").data;
  const Wq = ipw.subarray(0, E * E), bq = ipb.subarray(0, E);
  const Q = linear(qn, LqE, E, Wq, E, bq);
  const hd = E / nHeads, kvLen = cache.kvLen;
  if (opts.ssmax) qassmax(Q, LqE, E, nHeads, hd, kvLen, W, pfx);
  const O = new Float32Array(LqE * E), scale = 1 / Math.sqrt(hd), scores = new Float32Array(kvLen);
  const qh = new Float32Array(LqE * hd);
  for (let h = 0; h < nHeads; h++) {
    const ka = cache.kh[h], va = cache.vh[h];
    for (let p = 0; p < LqE; p++) for (let d = 0; d < hd; d++) qh[p * hd + d] = Q[p * E + h * hd + d];
    for (let p = 0; p < LqE; p++) {
      let mx = -Infinity; const qo = p * hd;
      for (let j = 0; j < kvLen; j++) { let acc = 0; const ko = j * hd; for (let d = 0; d < hd; d++) acc += qh[qo + d] * ka[ko + d]; acc *= scale; scores[j] = acc; if (acc > mx) mx = acc; }
      let sum = 0; for (let j = 0; j < kvLen; j++) { const e = Math.exp(scores[j] - mx); scores[j] = e; sum += e; }
      const inv = 1 / sum, oo = p * E + h * hd;
      for (let d = 0; d < hd; d++) { let acc = 0; for (let j = 0; j < kvLen; j++) acc += scores[j] * va[j * hd + d]; O[oo + d] = acc * inv; }
    }
  }
  const opw = W.get(pfx + ".out_proj.weight").data, opb = W.get(pfx + ".out_proj.bias").data;
  const attnOut = linear(O, LqE, E, opw, E, opb);
  const xs = x.slice(0, LqE * E); addInto(xs, attnOut);
  const gm = W.get(pfx + ".ln_mlp.weight").data, bm = W.get(pfx + ".ln_mlp.bias").data;
  addInto(xs, mlp(layernorm(xs, LqE, E, gm, bm), LqE, E, W, pfx));
  return { data: xs, n: LqE };
}

/* ---------------- InducedTransformerBlock (column stage) ---------------- */
function inducedBlock(W, pfx, seq, rows, E, nHeads, nTrain) {
  const ind = W.get(pfx + ".inducing_vectors"); // [1,128,E]
  const nInd = ind.shape[1];
  // tfm1: q=inducing, kv=seq (kvMax = nTrain), ssmax on
  const kv = block(W, pfx + ".tfm1", ind.data.slice(0, nInd * E), nInd, seq, rows, E, nHeads,
    { kvMax: nTrain, ssmax: true, rope: null });
  // tfm2: q=seq, kv=inducing(kv.data), no ssmax
  const out = block(W, pfx + ".tfm2", seq, rows, kv.data, kv.n, E, nHeads, { ssmax: false, rope: null });
  return out.data; // [rows,E]
}

/* ---------------- row stage (per-row, standalone; shared train/query) ----------------
 * cols: array (nGroups) of Float32Array[N*E]. Returns Float32Array[N*iclDim].
 */
function rowStage(W, C, cols, N, E, trace = null) {
  const nCls = C.n_cls_cols, iclDim = E * nCls, nGroups = cols.length, nTok = nCls + nGroups;
  const clsT = W.get("row_cls_tokens").data;
  const rowLnG = W.get("row_ln.weight").data, rowLnB = W.get("row_ln.bias").data;
  const rowVec = new Float32Array(N * iclDim);
  for (let i = 0; i < N; i++) {
    let tok = new Float32Array(nTok * E);
    for (let t = 0; t < nCls; t++) for (let e = 0; e < E; e++) tok[t * E + e] = clsT[t * E + e];
    for (let c = 0; c < nGroups; c++) { const o = i * E; for (let e = 0; e < E; e++) tok[(nCls + c) * E + e] = cols[c][o + e]; }
    let L = nTok;
    for (let bi = 0; bi < C.row_num_blocks; bi++) {
      const last = bi === C.row_num_blocks - 1;
      const r = block(W, `row_blocks.${bi}`, tok, L, null, L, E, C.row_nhead, { rope: C.rope_theta, qMax: last ? nCls : null });
      tok = r.data; L = r.n;
      if (trace && i === 0) trace(bi, tok, L, E);
    }
    rowVec.set(layernorm(tok, nCls, E, rowLnG, rowLnB), i * iclDim);
  }
  return rowVec;
}

// standardize + group + x_embed (+ optional y for train). Returns array(nGroups) of Float32Array[N*E].
function embedColumns(W, C, Xall, N, F, ytrain, nTrain) {
  const E = C.embed_dim, gs = C.feature_group_size, offs = C.group_offsets, maxC = C.max_classes;
  const xw = W.get("x_embed.weight").data, xb = W.get("x_embed.bias").data;
  const cols = []; for (let c = 0; c < F; c++) cols.push(new Float32Array(N * E));
  const grp = new Float32Array(gs);
  for (let i = 0; i < N; i++) for (let c = 0; c < F; c++) {
    for (let k = 0; k < gs; k++) grp[k] = Xall[i][(c + offs[k]) % F];
    const dst = cols[c], o = i * E;
    for (let e = 0; e < E; e++) { let acc = xb[e]; const wo = e * gs; for (let k = 0; k < gs; k++) acc += grp[k] * xw[wo + k]; dst[o + e] = acc; }
  }
  if (ytrain) {
    const yInW = W.get("y_embed_in.weight").data, yInB = W.get("y_embed_in.bias").data;
    for (let c = 0; c < F; c++) { const dst = cols[c]; for (let i = 0; i < nTrain; i++) { const cls = ytrain[i], o = i * E; for (let e = 0; e < E; e++) dst[o + e] += yInW[e * maxC + cls] + yInB[e]; } }
  }
  return cols;
}

function standardizeStats(Xtrain, F, nTrain) {
  const mean = new Float32Array(F), std = new Float32Array(F);
  for (let f = 0; f < F; f++) { let s = 0; for (let i = 0; i < nTrain; i++) s += Xtrain[i][f]; mean[f] = s / nTrain; }
  for (let f = 0; f < F; f++) { let v = 0; for (let i = 0; i < nTrain; i++) { const t = Xtrain[i][f] - mean[f]; v += t * t; } std[f] = Math.sqrt(v / nTrain) + 1e-8; }
  return { mean, std };
}
function zrow(src, mean, std, F) { const r = new Float32Array(F); for (let f = 0; f < F; f++) r[f] = (src[f] - mean[f]) / std[f]; return r; }

/* ---------------- inspection helpers (read-only) ---------------- */
function summary(data) {
  let min = Infinity, max = -Infinity, sum = 0, sum2 = 0;
  for (let i = 0; i < data.length; i++) {
    const value = data[i];
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
    sum2 += value * value;
  }
  const n = Math.max(1, data.length);
  return {
    min,
    max,
    mean: sum / n,
    rms: Math.sqrt(sum2 / n),
  };
}

function sampleMatrix(data, rows, dim, shown = 24) {
  const width = Math.min(shown, dim);
  const values = [];
  for (let row = 0; row < rows; row++) {
    const sampled = [];
    for (let column = 0; column < width; column++) {
      const source = width === 1
        ? 0
        : Math.round(column * (dim - 1) / (width - 1));
      sampled.push(data[row * dim + source]);
    }
    values.push(sampled);
  }
  return { rows, dim, shown: width, values, stats: summary(data) };
}

function sampleColumns(cols, E, shown = 24) {
  const flat = new Float32Array(cols.length * E);
  for (let column = 0; column < cols.length; column++) {
    flat.set(cols[column].subarray(0, E), column * E);
  }
  return sampleMatrix(flat, cols.length, E, shown);
}

// Average and per-head attention for the first query token against cached K.
// This duplicates only the score calculation and never changes model outputs.
function cachedAttentionWeights(W, pfx, q, E, nHeads, cache, ssmax) {
  const g = W.get(pfx + ".ln_attn.weight").data;
  const b = W.get(pfx + ".ln_attn.bias").data;
  const qn = layernorm(q, 1, E, g, b);
  const ipw = W.get(pfx + ".in_proj_weight").data;
  const ipb = W.get(pfx + ".in_proj_bias").data;
  const Wq = ipw.subarray(0, E * E);
  const bq = ipb.subarray(0, E);
  const Q = linear(qn, 1, E, Wq, E, bq);
  const hd = E / nHeads;
  const scale = 1 / Math.sqrt(hd);
  const rawLogits = [];
  for (let head = 0; head < nHeads; head++) {
    const K = cache.kh[head];
    const scores = [];
    for (let row = 0; row < cache.kvLen; row++) {
      let score = 0;
      for (let d = 0; d < hd; d++) {
        score += Q[head * hd + d] * K[row * hd + d];
      }
      scores.push(score * scale);
    }
    rawLogits.push(scores);
  }
  if (ssmax) qassmax(Q, 1, E, nHeads, hd, cache.kvLen, W, pfx);

  const heads = [];
  const scaledLogits = [];
  const average = new Array(cache.kvLen).fill(0);
  for (let head = 0; head < nHeads; head++) {
    const K = cache.kh[head];
    const scores = new Array(cache.kvLen);
    let max = -Infinity;
    for (let row = 0; row < cache.kvLen; row++) {
      let score = 0;
      for (let d = 0; d < hd; d++) {
        score += Q[head * hd + d] * K[row * hd + d];
      }
      score *= scale;
      scores[row] = score;
      if (score > max) max = score;
    }
    scaledLogits.push([...scores]);
    let total = 0;
    for (let row = 0; row < scores.length; row++) {
      scores[row] = Math.exp(scores[row] - max);
      total += scores[row];
    }
    for (let row = 0; row < scores.length; row++) {
      scores[row] /= total;
      average[row] += scores[row] / nHeads;
    }
    heads.push(scores);
  }
  return { average, heads, rawLogits, scaledLogits };
}

/* ---------------- full model ---------------- */
export function buildModel(manifest, buffer) {
  // Backward-compatible bridge for the Svelte explainer only. It performs one
  // selected-view core inspection, not browser classifier ensemble inference.
  let runtimeManifest = manifest;
  if (manifest.schema === "tabicl-browser-js/flat-tensors-v1") {
    const adapted = toNanoClassifierManifest(manifest);
    runtimeManifest = {
      ...adapted,
      config: { ...adapted.config, standardize_input: true },
    };
  }
  const W = dequantize(runtimeManifest, buffer);
  const C = runtimeManifest.config;
  return {
    config: C,

    /* One-shot forward (reference path). Xtrain:number[][]; ytrain:int[]; Xtest:number[][]. */
    predict(Xtrain, ytrain, Xtest) {
      const cache = this.prepareContext(Xtrain, ytrain);
      return this.predictQueries(cache, Xtest);
    },

    /* Precompute everything that depends only on the fixed context (train rows).
     * Lets a dragged query / grid sweep reuse the train pipeline cheaply. */
    prepareContext(Xtrain, ytrain) {
      const F = Xtrain[0].length, nTrain = Xtrain.length;
      const E = C.embed_dim, nCls = C.n_cls_cols, iclDim = E * nCls, maxC = C.max_classes;
      const { mean, std } = standardizeStats(Xtrain, F, nTrain);
      const Xz = Xtrain.map((r) => zrow(r, mean, std, F));

      // column stage on train, caching per (block,col) the inducing-derived K/V used by tfm2.
      let cols = embedColumns(W, C, Xz, nTrain, F, ytrain, nTrain);
      const colKV = []; // colKV[bi][c] = {kh,vh,kvLen}
      for (let bi = 0; bi < C.col_num_blocks; bi++) {
        const pfx = `col_blocks.${bi}`, ind = W.get(pfx + ".inducing_vectors"), nInd = ind.shape[1];
        const perCol = [];
        for (let c = 0; c < F; c++) {
          const indOut = block(W, pfx + ".tfm1", ind.data.slice(0, nInd * E), nInd, cols[c], nTrain, E, C.col_nhead, { kvMax: nTrain, ssmax: true, rope: null });
          perCol.push(computeKV(W, pfx + ".tfm2", indOut.data, indOut.n, E, C.col_nhead, null));
          cols[c] = block(W, pfx + ".tfm2", cols[c], nTrain, indOut.data, indOut.n, E, C.col_nhead, { ssmax: false, rope: null }).data;
        }
        colKV.push(perCol);
      }

      // row stage on train -> row vectors; add y_embed_icl.
      const rowVec = rowStage(W, C, cols, nTrain, E);
      const yIcW = W.get("y_embed_icl.weight").data, yIcB = W.get("y_embed_icl.bias").data;
      for (let i = 0; i < nTrain; i++) { const cls = ytrain[i], o = i * iclDim; for (let e = 0; e < iclDim; e++) rowVec[o + e] += yIcW[e * maxC + cls] + yIcB[e]; }

      // ICL stage on train, caching per-block train K/V; evolve train (self-attention).
      const iclKV = [];
      let rv = rowVec;
      for (let bi = 0; bi < C.icl_num_blocks - 1; bi++) {
        const pfx = `icl_blocks.${bi}`;
        iclKV.push(computeKV(W, pfx, rv, nTrain, iclDim, C.icl_nhead, nTrain));
        rv = block(W, pfx, rv, nTrain, null, nTrain, iclDim, C.icl_nhead, { ssmax: true, kvMax: nTrain }).data;
      }
      // last block's train K/V (query attends these)
      iclKV.push(computeKV(W, `icl_blocks.${C.icl_num_blocks - 1}`, rv, nTrain, iclDim, C.icl_nhead, nTrain));

      return { mean, std, F, nTrain, colKV, iclKV };
    },

    /* Fast query pass against a prepared context. Xtest:number[][]. Returns logits[][]. */
    predictQueries(cache, Xtest) {
      const F = cache.F, nTest = Xtest.length, E = C.embed_dim, nCls = C.n_cls_cols, iclDim = E * nCls, maxC = C.max_classes;
      const Xz = Xtest.map((r) => zrow(r, cache.mean, cache.std, F));

      // column stage: query rows attend cached inducing K/V (tfm2), no ssmax/rope.
      let cols = embedColumns(W, C, Xz, nTest, F, null, 0);
      for (let bi = 0; bi < C.col_num_blocks; bi++)
        for (let c = 0; c < F; c++)
          cols[c] = blockCachedKV(W, `col_blocks.${bi}.tfm2`, cols[c], nTest, E, C.col_nhead, cache.colKV[bi][c], { ssmax: false }).data;

      // row stage (standalone, no y).
      let rv = rowStage(W, C, cols, nTest, E);

      // ICL: query tokens attend cached train K/V each block (independent across queries).
      for (let bi = 0; bi < C.icl_num_blocks; bi++)
        rv = blockCachedKV(W, `icl_blocks.${bi}`, rv, nTest, iclDim, C.icl_nhead, cache.iclKV[bi], { ssmax: true }).data;

      // out_ln -> out_mlp.
      const olG = W.get("out_ln.weight").data, olB = W.get("out_ln.bias").data;
      const normed = layernorm(rv, nTest, iclDim, olG, olB);
      const m0w = W.get("out_mlp.0.weight"), m0b = W.get("out_mlp.0.bias");
      const m2w = W.get("out_mlp.2.weight"), m2b = W.get("out_mlp.2.bias");
      let hlt = linear(normed, nTest, iclDim, m0w.data, m0w.shape[0], m0b.data); geluInplace(hlt);
      const logits = linear(hlt, nTest, m0w.shape[0], m2w.data, m2w.shape[0], m2b.data);
      const out = [];
      for (let i = 0; i < nTest; i++) { const row = []; for (let k = 0; k < maxC; k++) row.push(logits[i * maxC + k]); out.push(row); }
      return out;
    },

    /* Inspect one real query without changing the prediction path.
     * Returns compact activation samples and actual ICL attention probabilities.
     */
    inspectQuery(cache, query, nClasses = 3) {
      const F = cache.F;
      const E = C.embed_dim;
      const nCls = C.n_cls_cols;
      const iclDim = E * nCls;
      const standardized = zrow(query, cache.mean, cache.std, F);
      const grouped = [];
      for (let column = 0; column < F; column++) {
        const values = [];
        for (let k = 0; k < C.feature_group_size; k++) {
          values.push(standardized[(column + C.group_offsets[k]) % F]);
        }
        grouped.push(values);
      }

      let cols = embedColumns(W, C, [standardized], 1, F, null, 0);
      const embedding = sampleColumns(cols, E);
      const columnBlocks = [];
      for (let bi = 0; bi < C.col_num_blocks; bi++) {
        for (let column = 0; column < F; column++) {
          cols[column] = blockCachedKV(
            W,
            `col_blocks.${bi}.tfm2`,
            cols[column],
            1,
            E,
            C.col_nhead,
            cache.colKV[bi][column],
            { ssmax: false }
          ).data;
        }
        columnBlocks.push({
          block: bi + 1,
          activation: sampleColumns(cols, E),
        });
      }

      const rowBlocks = [];
      let rv = rowStage(W, C, cols, 1, E, (bi, tokens, count, dim) => {
        rowBlocks.push({
          block: bi + 1,
          tokens: sampleMatrix(tokens, count, dim),
        });
      });
      const rowVector = sampleMatrix(rv, 1, iclDim, 32);

      const iclBlocks = [];
      for (let bi = 0; bi < C.icl_num_blocks; bi++) {
        const pfx = `icl_blocks.${bi}`;
        const attention = cachedAttentionWeights(
          W,
          pfx,
          rv,
          iclDim,
          C.icl_nhead,
          cache.iclKV[bi],
          true
        );
        rv = blockCachedKV(
          W,
          pfx,
          rv,
          1,
          iclDim,
          C.icl_nhead,
          cache.iclKV[bi],
          { ssmax: true }
        ).data;
        iclBlocks.push({
          block: bi + 1,
          activation: sampleMatrix(rv, 1, iclDim, 32),
          attention,
        });
      }

      const olG = W.get("out_ln.weight").data;
      const olB = W.get("out_ln.bias").data;
      const normed = layernorm(rv, 1, iclDim, olG, olB);
      const m0w = W.get("out_mlp.0.weight");
      const m0b = W.get("out_mlp.0.bias");
      const m2w = W.get("out_mlp.2.weight");
      const m2b = W.get("out_mlp.2.bias");
      let hidden = linear(
        normed,
        1,
        iclDim,
        m0w.data,
        m0w.shape[0],
        m0b.data
      );
      geluInplace(hidden);
      const allLogits = linear(
        hidden,
        1,
        m0w.shape[0],
        m2w.data,
        m2w.shape[0],
        m2b.data
      );
      const logits = Array.from(allLogits.subarray(0, nClasses));
      const probabilities = softmaxRows([logits], nClasses)[0];

      return {
        config: {
          embedDim: E,
          iclDim,
          featureGroupSize: C.feature_group_size,
          groupOffsets: C.group_offsets,
          columnBlocks: C.col_num_blocks,
          rowBlocks: C.row_num_blocks,
          iclBlocks: C.icl_num_blocks,
          heads: C.icl_nhead,
          inducing: C.n_inducing,
          clsTokens: C.n_cls_cols,
        },
        input: {
          raw: [...query],
          mean: Array.from(cache.mean),
          std: Array.from(cache.std),
          standardized: Array.from(standardized),
          grouped,
        },
        embedding,
        columnBlocks,
        rowBlocks,
        rowVector,
        iclBlocks,
        output: {
          normalized: sampleMatrix(normed, 1, iclDim, 32),
          hidden: sampleMatrix(hidden, 1, m0w.shape[0], 32),
          logits,
          probabilities,
        },
      };
    },

    /* convenience: softmax over the first `nClasses` logits of predict() */
    predictProba(Xtrain, ytrain, Xtest, nClasses = 2) {
      return softmaxRows(this.predict(Xtrain, ytrain, Xtest), nClasses);
    },
    probaFromLogits(logits, nClasses = 2) { return softmaxRows(logits, nClasses); },
  };
}

function softmaxRows(logits, nClasses) {
  return logits.map((row) => {
    const z = row.slice(0, nClasses), mx = Math.max(...z);
    const ex = z.map((v) => Math.exp(v - mx)), s = ex.reduce((a, b) => a + b, 0);
    return ex.map((v) => v / s);
  });
}

export async function loadModel(baseUrl) {
  const manifest = await (await fetch(baseUrl + "/manifest.json")).json();
  const binaryName = manifest.binary || "tabicl.bin";
  const buffer = await (await fetch(baseUrl + "/" + binaryName)).arrayBuffer();
  await verifySha256(buffer, manifest.binary_sha256, binaryName);
  return buildModel(manifest, buffer);
}
