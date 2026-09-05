/* playground.js: compare kNN with the real model in your browser.
 * Drives a three-feature classification playground with a 2D prediction slice,
 * backed by either a local k-nearest neighbors rule or the converted TabICL v2
 * (int8) running in a Web Worker.
 * Pure vanilla, no build step.
 */
import { predictKnn } from "./knn.js";
import {
  classIndexFromProbability,
  contourSegments2d,
  createProjection,
  isosurfaceIntersections,
  predictionColorFromProbability,
  rankContextAttention,
  sliceInterpolation,
} from "./playground3d.js";

const $ = (s) => document.querySelector(s);

const COL0 = [46, 166, 214];    // class 0  cyan  #2ea6d6
const COL1 = [242, 114, 75];    // class 1  coral #f2724b
const DOM = 2.4;                 // data domain [-DOM, DOM]
const GRIDN = 16;               // 16 x 16 prediction samples
const GRID_SIZE = GRIDN ** 2;
const DEFAULT_YAW = -Math.PI / 4;
const DEFAULT_PITCH = 0.48;

/* ---------- seeded RNG + datasets ---------- */
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function gen(name, n, seed = 7) {
  const r = mulberry32(seed), X = [], y = [];
  const jit = (s) => (r() - 0.5) * s;
  for (let i = 0; i < n; i++) {
    let c = i % 2; let x0, x1;
    if (name === "interaction") {
      x0 = (r() * 2 - 1) * 2.15;
      x1 = (r() * 2 - 1) * 2.15;
      const score = Math.sin(1.35 * x0) + 0.58 * x0 * x1
        - 0.35 * Math.cos(1.45 * x1) + jit(0.28);
      c = Number(score > 0);
    } else if (name === "moons") {
      const t = r() * Math.PI;
      if (c === 0) { x0 = 1.4 * Math.cos(t) - 0.7; x1 = 1.4 * Math.sin(t) - 0.35; }
      else { x0 = 1.4 * Math.cos(t) + 0.0 + 0.7; x1 = -1.4 * Math.sin(t) + 0.35; }
      x0 += jit(0.35); x1 += jit(0.35);
    } else if (name === "circles") {
      const angle = r() * 2 * Math.PI;
      const radius = c === 0 ? 0.72 : 1.72;
      x0 = radius * Math.cos(angle) + jit(0.24);
      x1 = radius * Math.sin(angle) + jit(0.24);
    } else if (name === "xor") {
      const qx = r() < 0.5 ? -1 : 1, qy = r() < 0.5 ? -1 : 1;
      x0 = qx * (0.8 + r() * 0.75) + jit(0.22);
      x1 = qy * (0.8 + r() * 0.75) + jit(0.22);
      c = Number(qx * qy > 0);
    } else { // blobs
      const cx = c === 0 ? -0.9 : 0.9, cy = c === 0 ? -0.6 : 0.6;
      x0 = cx + jit(1.1); x1 = cy + jit(1.1);
    }
    X.push([x0, x1]); y.push(c);
  }
  return { X, y };
}

/* ---------- state ---------- */
const state = {
  ds: "interaction", n: 20, activeClass: 1, model: "tfm", k: 5, showLinks: true,
  view: "flat", viewYaw: DEFAULT_YAW, viewPitch: DEFAULT_PITCH,
  X: [], y: [], query: [0.4, -0.3],
  loaded: false, loading: false, prepared: false, prepTag: 0,
  qInFlight: false, qQueued: false, queryTag: 0, qProba: null, qClass: null, neighbors: [],
  inspectionView: 0, viewCount: 8, selectedView: null, selectedViewProbability: null,
  attention: null, attentionBlock: null, attentionHeads: null,
  grid: null, surface: null, gridTag: 0, gridBusy: false,
  classicalReady: false, classicalTag: 0, classicalQueryTag: 0,
  classicalProbabilities: {}, classicalGrids: {},
};

let worker = null, canvas = null, ctx = null, off = null, offctx = null, SZ = 440, dpr = 1;
let knnCanvas = null, knnCtx = null, knnOff = null, knnOffctx = null, KNNSZ = 320;
let classicalWorker = null;
const classicalViews = {};
let drawFrame = null;

/* ---------- coordinate mapping ---------- */
const toPx = (v) => ((v + DOM) / (2 * DOM)) * SZ;
const toData = (px) => (px / SZ) * 2 * DOM - DOM;
const gridIndex = (x, y) => x * GRIDN + y;
const gridValue = (index) => -DOM + ((index + 0.5) / GRIDN) * DOM * 2;

/* ---------- rendering ---------- */
function classColor(p) { return classIndexFromProbability(p) === 0 ? COL0 : COL1; }
function predictionColor(p) { return predictionColorFromProbability(p); }
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function knnPredict(query) {
  return predictKnn(state.X, state.y, query, state.k);
}

function draw() {
  drawClassicalComparison("linear");
  drawClassicalComparison("tree");
  drawClassicalComparison("knn");
  if (state.view === "3d") draw3d();
  else drawFlat();
}

function scheduleDraw() {
  if (drawFrame !== null) return;
  drawFrame = requestAnimationFrame(() => {
    drawFrame = null;
    draw();
  });
}

function drawFlat() {
  ctx.clearRect(0, 0, SZ, SZ);
  // background
  ctx.fillStyle = "#242c37"; ctx.fillRect(0, 0, SZ, SZ);

  // Two-dimensional prediction field.
  if (state.grid) {
    const img = offctx.createImageData(GRIDN, GRIDN);
    const sliceValues = new Float32Array(GRIDN ** 2).fill(-1);
    for (let a = 0; a < GRIDN; a++) for (let b = 0; b < GRIDN; b++) {
      const p = state.grid[gridIndex(a, b)];
      const i = (b * GRIDN + a) * 4;
      sliceValues[b * GRIDN + a] = p;
      if (p < 0) { img.data[i + 3] = 0; continue; }   // not yet computed
      const c = predictionColor(p);
      img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 220;
    }
    offctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.save(); ctx.translate(0, SZ); ctx.scale(1, -1);   // flip y so data-y points up
    ctx.drawImage(off, 0, 0, GRIDN, GRIDN, 0, 0, SZ, SZ);
    ctx.restore();
    drawSliceBoundary(sliceValues);
  }

  // grid axes (integer data coordinates)
  ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 1;
  for (let g = -2; g <= 2; g++) {
    const p = toPx(g);
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, SZ); ctx.stroke();          // vertical
    ctx.beginPath(); ctx.moveTo(0, SZ - p); ctx.lineTo(SZ, SZ - p); ctx.stroke(); // horizontal
  }

  // kNN evidence links are drawn below points and above the boundary.
  if (state.model === "knn" && state.showLinks && state.neighbors.length) {
    const qx = toPx(state.query[0]), qy = SZ - toPx(state.query[1]);
    ctx.save(); ctx.lineWidth = 1.25;
    for (const neighbor of state.neighbors) {
      const point = state.X[neighbor.index];
      if (!point) continue;
      const sliceDistance = Math.abs(point[2] - state.query[2]);
      ctx.strokeStyle = `rgba(255,255,255,${0.2 + 0.52 * Math.max(0, 1 - sliceDistance / (DOM * 2))})`;
      ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(toPx(point[0]), SZ - toPx(point[1])); ctx.stroke();
    }
    ctx.restore();
  }

  // context points (with glow)
  for (let i = 0; i < state.X.length; i++) {
    const px = toPx(state.X[i][0]), py = SZ - toPx(state.X[i][1]);
    const c = state.y[i] === 0 ? COL0 : COL1;
    ctx.save();
    ctx.beginPath(); ctx.arc(px, py, 5.5, 0, 2 * Math.PI);
    ctx.shadowColor = `rgba(${c[0]},${c[1]},${c[2]},0.9)`; ctx.shadowBlur = 12;
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.fill();
    ctx.lineWidth = 1.1; ctx.strokeStyle = "rgba(255,255,255,0.7)"; ctx.stroke();
    ctx.restore();
  }

  // query marker
  const qx = toPx(state.query[0]), qy = SZ - toPx(state.query[1]);
  const p1 = state.qProba ? state.qProba[1] : null;
  ctx.save();
  ctx.beginPath(); ctx.arc(qx, qy, 14, 0, 2 * Math.PI);
  if (p1 == null) { ctx.fillStyle = "rgba(124,108,240,0.18)"; }
  else {
    const c = classColor(p1);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.shadowColor = `rgba(${c[0]},${c[1]},${c[2]},0.9)`; ctx.shadowBlur = 18;
  }
  ctx.fill();
  ctx.restore();
  ctx.lineWidth = 3; ctx.strokeStyle = "#7c6cf0"; ctx.stroke();
  drawStar(qx, qy, 6.5, "#242c37");
}

function drawKnnComparison() {
  if (!knnCtx || !knnOffctx) return;
  const size = KNNSZ;
  const toKnnPx = (value) => ((value + DOM) / (2 * DOM)) * size;
  knnCtx.clearRect(0, 0, size, size);
  knnCtx.fillStyle = "#242c37";
  knnCtx.fillRect(0, 0, size, size);

  const image = knnOffctx.createImageData(GRIDN, GRIDN);
  const values = new Float32Array(GRIDN ** 2).fill(-1);
  for (let a = 0; a < GRIDN; a++) for (let b = 0; b < GRIDN; b++) {
    const result = knnPredict([gridValue(a), gridValue(b), state.query[2]]);
    const p = result?.proba?.[1] ?? -1;
    values[b * GRIDN + a] = p;
    const index = (b * GRIDN + a) * 4;
    if (p < 0) continue;
    const color = predictionColor(p);
    image.data[index] = color[0];
    image.data[index + 1] = color[1];
    image.data[index + 2] = color[2];
    image.data[index + 3] = 220;
  }
  knnOffctx.putImageData(image, 0, 0);
  knnCtx.imageSmoothingEnabled = true;
  knnCtx.save();
  knnCtx.translate(0, size);
  knnCtx.scale(1, -1);
  knnCtx.drawImage(knnOff, 0, 0, GRIDN, GRIDN, 0, 0, size, size);
  knnCtx.restore();
  drawSliceBoundary(values, knnCtx, size);

  knnCtx.strokeStyle = "rgba(255,255,255,.06)";
  knnCtx.lineWidth = 1;
  for (let grid = -2; grid <= 2; grid++) {
    const position = toKnnPx(grid);
    knnCtx.beginPath();
    knnCtx.moveTo(position, 0);
    knnCtx.lineTo(position, size);
    knnCtx.stroke();
    knnCtx.beginPath();
    knnCtx.moveTo(0, size - position);
    knnCtx.lineTo(size, size - position);
    knnCtx.stroke();
  }

  const queryResult = knnPredict(state.query);
  const queryX = toKnnPx(state.query[0]);
  const queryY = size - toKnnPx(state.query[1]);
  if (state.showLinks && queryResult?.neighbors?.length) {
    knnCtx.save();
    knnCtx.strokeStyle = "rgba(255,255,255,.48)";
    knnCtx.lineWidth = 1.1;
    queryResult.neighbors.forEach((neighbor) => {
      const point = state.X[neighbor.index];
      if (!point) return;
      knnCtx.beginPath();
      knnCtx.moveTo(queryX, queryY);
      knnCtx.lineTo(toKnnPx(point[0]), size - toKnnPx(point[1]));
      knnCtx.stroke();
    });
    knnCtx.restore();
  }

  state.X.forEach((point, index) => {
    const color = state.y[index] === 0 ? COL0 : COL1;
    const sliceDistance = Math.abs(point[2] - state.query[2]);
    const alpha = 0.18 + 0.82 * Math.max(0, 1 - sliceDistance / (DOM * 1.2));
    knnCtx.save();
    knnCtx.globalAlpha = alpha;
    knnCtx.beginPath();
    knnCtx.arc(toKnnPx(point[0]), size - toKnnPx(point[1]), 4.5, 0, 2 * Math.PI);
    knnCtx.fillStyle = `rgb(${color.join(",")})`;
    knnCtx.fill();
    knnCtx.strokeStyle = "rgba(255,255,255,.72)";
    knnCtx.stroke();
    knnCtx.restore();
  });

  const p1 = queryResult?.proba?.[1];
  const queryColor = Number.isFinite(p1) ? classColor(p1) : [124, 108, 240];
  knnCtx.beginPath();
  knnCtx.arc(queryX, queryY, 12, 0, 2 * Math.PI);
  knnCtx.fillStyle = `rgb(${queryColor.join(",")})`;
  knnCtx.fill();
  knnCtx.lineWidth = 3;
  knnCtx.strokeStyle = "#7c6cf0";
  knnCtx.stroke();
  drawStar(queryX, queryY, 6, "#242c37", knnCtx);

  knnCtx.fillStyle = "rgba(226,232,240,.72)";
  knnCtx.font = `${Math.max(9, size * 0.026)}px ui-monospace, monospace`;
  knnCtx.fillText(`k = ${state.k}`, 12, size - 13);

  const readout = $("#pg-knn-readout");
  if (!readout || !queryResult) return;
  const cls = queryResult.classIndex;
  const color = cls === 0 ? COL0 : COL1;
  const votes = queryResult.neighbors.reduce(
    (sum, neighbor) => sum + Number(state.y[neighbor.index] === cls),
    0
  );
  readout.style.setProperty("--prediction-color", `rgb(${color.join(",")})`);
  readout.innerHTML =
    `<span class="pg-dot" style="background:rgb(${color.join(",")})"></span>` +
    `<span class="readout__label">kNN prediction</span><strong>class ${cls}</strong>` +
    `<span class="readout__detail">${votes}/${queryResult.neighbors.length} neighbor votes · p(1) ${(p1 * 100).toFixed(0)}%</span>`;
}

function drawClassicalComparison(kind) {
  const view = classicalViews[kind];
  if (!view?.ctx || !view.offctx) return;
  const { ctx: targetCtx, off: targetOff, offctx: targetOffctx, size } = view;
  const toViewPx = (value) => ((value + DOM) / (2 * DOM)) * size;
  targetCtx.clearRect(0, 0, size, size);
  targetCtx.fillStyle = "#242c37";
  targetCtx.fillRect(0, 0, size, size);

  const grid = state.classicalGrids[kind];
  if (grid) {
    const image = targetOffctx.createImageData(GRIDN, GRIDN);
    const values = new Float32Array(GRIDN ** 2).fill(-1);
    for (let a = 0; a < GRIDN; a++) for (let b = 0; b < GRIDN; b++) {
      const p = grid[gridIndex(a, b)];
      values[b * GRIDN + a] = p;
      const index = (b * GRIDN + a) * 4;
      const color = predictionColor(p);
      image.data[index] = color[0];
      image.data[index + 1] = color[1];
      image.data[index + 2] = color[2];
      image.data[index + 3] = 220;
    }
    targetOffctx.putImageData(image, 0, 0);
    targetCtx.imageSmoothingEnabled = true;
    targetCtx.save();
    targetCtx.translate(0, size);
    targetCtx.scale(1, -1);
    targetCtx.drawImage(targetOff, 0, 0, GRIDN, GRIDN, 0, 0, size, size);
    targetCtx.restore();
    drawSliceBoundary(values, targetCtx, size);
  }

  targetCtx.strokeStyle = "rgba(255,255,255,.06)";
  targetCtx.lineWidth = 1;
  for (let line = -2; line <= 2; line += 1) {
    const position = toViewPx(line);
    targetCtx.beginPath();
    targetCtx.moveTo(position, 0);
    targetCtx.lineTo(position, size);
    targetCtx.stroke();
    targetCtx.beginPath();
    targetCtx.moveTo(0, size - position);
    targetCtx.lineTo(size, size - position);
    targetCtx.stroke();
  }

  const queryX = toViewPx(state.query[0]);
  const queryY = size - toViewPx(state.query[1]);
  const neighbors = kind === "knn" ? knnPredict(state.query)?.neighbors || [] : [];
  if (kind === "knn" && state.showLinks) {
    targetCtx.save();
    targetCtx.strokeStyle = "rgba(255,255,255,.48)";
    neighbors.forEach((neighbor) => {
      const point = state.X[neighbor.index];
      if (!point) return;
      targetCtx.beginPath();
      targetCtx.moveTo(queryX, queryY);
      targetCtx.lineTo(toViewPx(point[0]), size - toViewPx(point[1]));
      targetCtx.stroke();
    });
    targetCtx.restore();
  }

  state.X.forEach((point, index) => {
    const color = state.y[index] === 0 ? COL0 : COL1;
    targetCtx.save();
    targetCtx.beginPath();
    targetCtx.arc(toViewPx(point[0]), size - toViewPx(point[1]), 4.5, 0, 2 * Math.PI);
    targetCtx.fillStyle = `rgb(${color.join(",")})`;
    targetCtx.fill();
    targetCtx.strokeStyle = "rgba(255,255,255,.72)";
    targetCtx.stroke();
    targetCtx.restore();
  });

  const p1 = state.classicalProbabilities[kind];
  const queryColor = Number.isFinite(p1) ? classColor(p1) : [124, 108, 240];
  targetCtx.beginPath();
  targetCtx.arc(queryX, queryY, 12, 0, 2 * Math.PI);
  targetCtx.fillStyle = `rgb(${queryColor.join(",")})`;
  targetCtx.fill();
  targetCtx.lineWidth = 3;
  targetCtx.strokeStyle = "#7c6cf0";
  targetCtx.stroke();
  drawStar(queryX, queryY, 6, "#242c37", targetCtx);

  if (kind === "knn") {
    targetCtx.fillStyle = "rgba(226,232,240,.72)";
    targetCtx.font = `${Math.max(9, size * 0.026)}px ui-monospace, monospace`;
    targetCtx.fillText(`k = ${state.k}`, 12, size - 13);
  }

  const readout = $(`#pg-${kind}-readout`);
  if (!readout || !Number.isFinite(p1)) {
    if (readout) readout.innerHTML = "";
    return;
  }
  const cls = p1 >= 0.5 ? 1 : 0;
  const color = cls === 0 ? COL0 : COL1;
  const modelLabel = kind === "linear" ? "linear" : kind === "tree" ? "tree" : "kNN";
  const detail = kind === "knn"
    ? `${neighbors.filter((neighbor) => state.y[neighbor.index] === cls).length}/${neighbors.length} neighbor votes`
    : `p(1) ${(p1 * 100).toFixed(0)}%`;
  readout.style.setProperty("--prediction-color", `rgb(${color.join(",")})`);
  readout.innerHTML =
    `<span class="pg-dot" style="background:rgb(${color.join(",")})"></span>` +
    `<span class="readout__label">${modelLabel}</span><strong>class ${cls}</strong>` +
    `<span class="readout__detail">${detail}</span>`;
}

function drawSliceBoundary(values, targetCtx = ctx, size = SZ) {
  const segments = contourSegments2d(values, GRIDN);
  if (!segments.length) return;
  const toScreen = ([x, y]) => ({
    x: ((x + 0.5) / GRIDN) * size,
    y: size - ((y + 0.5) / GRIDN) * size,
  });
  targetCtx.save();
  targetCtx.beginPath();
  for (const segment of segments) {
    const start = toScreen(segment[0]);
    const end = toScreen(segment[1]);
    targetCtx.moveTo(start.x, start.y);
    targetCtx.lineTo(end.x, end.y);
  }
  targetCtx.lineCap = "round";
  targetCtx.lineJoin = "round";
  targetCtx.strokeStyle = "rgba(169,156,248,.34)";
  targetCtx.lineWidth = Math.max(6, size * 0.015);
  targetCtx.stroke();
  targetCtx.strokeStyle = "rgba(238,235,255,.94)";
  targetCtx.lineWidth = Math.max(1.5, size * 0.004);
  targetCtx.stroke();
  targetCtx.restore();
}

function polygon(points, fill, stroke = null, lineWidth = 1) {
  if (!points.length) return;
  ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); }
}

function drawSphere(point, color, radius = 6) {
  const glow = ctx.createRadialGradient(point.x - radius * 0.35, point.y - radius * 0.4, 1, point.x, point.y, radius * 1.5);
  glow.addColorStop(0, "rgba(255,255,255,.96)");
  glow.addColorStop(0.22, `rgb(${color.join(",")})`);
  glow.addColorStop(0.72, `rgb(${color.map((channel) => Math.round(channel * 0.72)).join(",")})`);
  glow.addColorStop(1, `rgba(${color.join(",")},0)`);
  ctx.save();
  ctx.beginPath(); ctx.arc(point.x, point.y, radius * 1.5, 0, 2 * Math.PI);
  ctx.fillStyle = glow; ctx.shadowColor = `rgba(${color.join(",")},.72)`; ctx.shadowBlur = radius * 2;
  ctx.fill(); ctx.restore();
  ctx.beginPath(); ctx.arc(point.x, point.y, radius, 0, 2 * Math.PI);
  ctx.strokeStyle = "rgba(255,255,255,.78)"; ctx.lineWidth = 1; ctx.stroke();
}

function drawProbabilityVolume(projection) {
  if (!state.grid) return;
  const samples = [];
  for (let x = 0; x < GRIDN; x++) for (let y = 0; y < GRIDN; y++) for (let z = 0; z < GRIDN; z++) {
    const probability = state.grid[gridIndex(x, y, z)];
    if (!Number.isFinite(probability) || probability < 0) continue;
    const position = projection.project([gridValue(x), gridValue(y), gridValue(z)]);
    samples.push({ probability, position });
  }
  samples.sort((left, right) => left.position.depth - right.position.depth);

  const radius = Math.max(7, SZ * 0.028);
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  for (const { probability, position } of samples) {
    const confidence = Math.abs(probability - 0.5) * 2;
    const color = predictionColor(probability);
    ctx.beginPath();
    ctx.arc(position.x, position.y, radius * (1.08 - confidence * 0.22), 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${color.join(",")},${0.018 + confidence * 0.07})`;
    ctx.fill();
  }

  const surface = state.surface || [];
  const surfaceStride = Math.max(1, Math.ceil(surface.length / 480));
  const boundary = [];
  for (let index = 0; index < surface.length; index += surfaceStride) {
    boundary.push(projection.project(surface[index]));
  }
  boundary.sort((left, right) => left.depth - right.depth);
  for (const point of boundary) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, Math.max(4, SZ * 0.013), 0, Math.PI * 2);
    ctx.fillStyle = "rgba(169,156,248,.13)";
    ctx.fill();
  }
  ctx.restore();
}

function drawRankBadge(point, rank) {
  const radius = 9;
  const x = point.x + 9;
  const y = point.y - 9;
  ctx.save();
  ctx.beginPath(); ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = "#f8fafc"; ctx.fill();
  ctx.lineWidth = 1.5; ctx.strokeStyle = "#7c6cf0"; ctx.stroke();
  ctx.fillStyle = "#172033"; ctx.font = "700 10px ui-monospace, monospace";
  ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(rank), x, y + 0.5);
  ctx.restore();
}

function draw3d() {
  const projection = createProjection(SZ, DOM, state.viewYaw, state.viewPitch);
  ctx.clearRect(0, 0, SZ, SZ);

  const background = ctx.createRadialGradient(SZ * 0.48, SZ * 0.42, SZ * 0.05, SZ * 0.5, SZ * 0.52, SZ * 0.72);
  background.addColorStop(0, "#303c4d"); background.addColorStop(0.58, "#202936"); background.addColorStop(1, "#151c25");
  ctx.fillStyle = background; ctx.fillRect(0, 0, SZ, SZ);

  const base = [[-DOM, -DOM, -DOM], [DOM, -DOM, -DOM], [DOM, DOM, -DOM], [-DOM, DOM, -DOM]];
  const top = base.map(([x, y]) => [x, y, DOM]);
  const baseProjected = base.map((point) => projection.project(point));
  const topProjected = top.map((point) => projection.project(point));
  polygon(baseProjected, "rgba(21,30,42,.82)", "rgba(203,213,225,.3)", 1.2);

  // The base grid anchors x and y while the enclosing frame makes z a real feature axis.
  ctx.save(); ctx.strokeStyle = "rgba(226,232,240,.13)"; ctx.lineWidth = 1;
  for (let g = -2; g <= 2; g++) {
    const verticalA = projection.project([g, -DOM, -DOM]);
    const verticalB = projection.project([g, DOM, -DOM]);
    ctx.beginPath(); ctx.moveTo(verticalA.x, verticalA.y); ctx.lineTo(verticalB.x, verticalB.y); ctx.stroke();
    const horizontalA = projection.project([-DOM, g, -DOM]);
    const horizontalB = projection.project([DOM, g, -DOM]);
    ctx.beginPath(); ctx.moveTo(horizontalA.x, horizontalA.y); ctx.lineTo(horizontalB.x, horizontalB.y); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(203,213,225,.26)";
  for (let edge = 0; edge < 4; edge++) {
    const next = (edge + 1) % 4;
    ctx.beginPath(); ctx.moveTo(topProjected[edge].x, topProjected[edge].y);
    ctx.lineTo(topProjected[next].x, topProjected[next].y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(baseProjected[edge].x, baseProjected[edge].y);
    ctx.lineTo(topProjected[edge].x, topProjected[edge].y); ctx.stroke();
  }
  ctx.restore();

  drawProbabilityVolume(projection);

  const context = state.X.map((point, index) => ({
    index, point, position: projection.project(point),
  })).sort((left, right) => left.position.depth - right.position.depth);

  // Neighbor segments connect the query to the exact rows used by 3D kNN.
  const p1 = state.qProba ? state.qProba[1] : null;
  const queryPosition = projection.project(state.query);
  if (state.model === "knn" && state.showLinks && state.neighbors.length) {
    ctx.save(); ctx.strokeStyle = "rgba(255,255,255,.6)"; ctx.lineWidth = 1.25;
    for (const neighbor of state.neighbors) {
      const point = state.X[neighbor.index];
      if (!point) continue;
      const target = projection.project(point);
      ctx.beginPath(); ctx.moveTo(queryPosition.x, queryPosition.y);
      ctx.lineTo(target.x, target.y); ctx.stroke();
    }
    ctx.restore();
  }

  const rankedAttention = state.model === "tfm"
    ? rankContextAttention(state.attention, 5)
    : [];
  if (rankedAttention.length) {
    const strongest = rankedAttention[0].weight || 1;
    ctx.save();
    ctx.lineCap = "round";
    for (const item of rankedAttention.slice().reverse()) {
      const point = state.X[item.index];
      if (!point) continue;
      const relative = item.weight / strongest;
      const target = projection.project(point);
      ctx.beginPath(); ctx.moveTo(queryPosition.x, queryPosition.y);
      ctx.lineTo(target.x, target.y);
      ctx.lineWidth = 1.5 + relative * 4;
      ctx.strokeStyle = `rgba(169,156,248,${0.3 + relative * 0.55})`;
      ctx.stroke();
    }
    ctx.restore();
  }

  for (const item of context) {
    const color = state.y[item.index] === 0 ? COL0 : COL1;
    drawSphere(item.position, color, 4.5);
  }

  for (let rank = rankedAttention.length - 1; rank >= 0; rank--) {
    const item = rankedAttention[rank];
    const point = state.X[item.index];
    if (point) drawRankBadge(projection.project(point), rank + 1);
  }

  const queryColor = p1 == null ? [124, 108, 240] : classColor(p1);
  drawSphere(queryPosition, queryColor, 13);
  ctx.lineWidth = 3; ctx.strokeStyle = "#a99cf8";
  ctx.beginPath(); ctx.arc(queryPosition.x, queryPosition.y, 14, 0, 2 * Math.PI); ctx.stroke();
  drawStar(queryPosition.x, queryPosition.y, 6.5, "#17202c");

  const xAxis = projection.project([DOM, -DOM, -DOM]);
  const yAxis = projection.project([-DOM, DOM, -DOM]);
  const zAxis = projection.project([-DOM, -DOM, DOM]);
  ctx.save(); ctx.fillStyle = "rgba(226,232,240,.8)"; ctx.font = `${Math.max(10, SZ * 0.024)}px ui-monospace, monospace`;
  ctx.fillText("x", xAxis.x + 5, xAxis.y + 4);
  ctx.fillText("y", yAxis.x - 10, yAxis.y + 5);
  ctx.fillText("z", zAxis.x - 4, zAxis.y - 7); ctx.restore();
}

function drawStar(cx, cy, R, fill, targetCtx = ctx) {
  targetCtx.beginPath();
  for (let i = 0; i < 10; i++) { const ang = -Math.PI / 2 + i * Math.PI / 5, r = i % 2 ? R * 0.45 : R; const x = cx + r * Math.cos(ang), yv = cy + r * Math.sin(ang); i ? targetCtx.lineTo(x, yv) : targetCtx.moveTo(x, yv); }
  targetCtx.closePath(); targetCtx.fillStyle = fill; targetCtx.fill();
}

/* ---------- worker orchestration ---------- */
function comparisonGridPoints() {
  const points = [];
  for (let a = 0; a < GRIDN; a++) for (let b = 0; b < GRIDN; b++) {
    points.push([gridValue(a), gridValue(b)]);
  }
  return points;
}

function ensureClassicalWorker() {
  if (classicalWorker) return;
  classicalWorker = new Worker(new URL("./classical-worker.js", import.meta.url), { type: "module" });
  classicalWorker.onmessage = (event) => {
    const message = event.data;
    if (message.type === "prepared") {
      if (message.tag !== state.classicalTag) return;
      state.classicalReady = true;
      requestClassicalQuery();
      classicalWorker.postMessage({
        type: "grid",
        tag: state.classicalTag,
        X: comparisonGridPoints(),
      });
      return;
    }
    if (message.type === "query") {
      if (message.tag !== state.classicalQueryTag) return;
      state.classicalProbabilities = message.probabilities;
      draw();
      return;
    }
    if (message.type === "grid") {
      if (message.tag !== state.classicalTag) return;
      state.classicalGrids = message.grids;
      draw();
      return;
    }
    if (message.type === "error") {
      console.error(message.message);
    }
  };
}

function prepareClassicalModels() {
  state.classicalTag += 1;
  state.classicalReady = false;
  state.classicalProbabilities = {};
  state.classicalGrids = {};
  if (!state.X.length) {
    draw();
    return;
  }
  ensureClassicalWorker();
  classicalWorker.postMessage({
    type: "prepare",
    tag: state.classicalTag,
    X: state.X,
    y: state.y,
    k: state.k,
  });
}

function requestClassicalQuery() {
  if (!state.classicalReady || !classicalWorker) return;
  state.classicalQueryTag += 1;
  classicalWorker.postMessage({
    type: "query",
    tag: state.classicalQueryTag,
    X: state.query,
  });
}

function ensureWorker() {
  if (worker) return;
  worker = new Worker(new URL("./tabicl/worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") {
      const pct = m.total ? Math.round((m.loaded / m.total) * 100) : null;
      setBar(pct == null ? Math.min(95, (m.loaded / 29e6) * 100) : pct);
      if (state.model === "tfm") setStatus(`downloading · ${(m.loaded / 1e6).toFixed(1)} mb`);
    } else if (m.type === "loaded") {
      state.loaded = true; state.loading = false; setBar(100);
      updateModelUI();
      reprepare();
    } else if (m.type === "prepared") {
      if (m.tag !== state.prepTag) return;   // stale
      state.prepared = true;
      state.viewCount = m.viewCount || 8;
      updateTraceViewOptions();
      updateModelUI();
      if (state.model === "tfm") { requestQuery(); requestLandscape(); }
    } else if (m.type === "result") {
      if (m.tag !== state.queryTag) return;
      state.qInFlight = false;
      if (state.qQueued) { state.qQueued = false; requestQuery(); return; }
      if (state.model === "tfm") {
        state.qProba = m.proba[0]; state.qClass = state.qProba[1] >= 0.5 ? 1 : 0; state.neighbors = [];
        updateReadout(); draw();
      }
    } else if (m.type === "inspection") {
      if (m.tag !== state.queryTag) return;
      state.qInFlight = false;
      if (state.qQueued) { state.qQueued = false; requestQuery(); return; }
      if (state.model === "tfm") {
        state.qProba = m.proba;
        state.qClass = state.qProba[1] >= 0.5 ? 1 : 0;
        state.neighbors = [];
        state.selectedView = m.selectedView;
        state.selectedViewProbability = m.selectedViewProbability;
        state.attention = m.attention;
        state.attentionBlock = m.block;
        state.attentionHeads = m.heads;
        updateReadout(); updateCompanion(); updateModelUI(); draw();
      }
    } else if (m.type === "gridChunk") {
      if (m.tag !== state.gridTag || !state.grid) return;
      for (let k = 0; k < m.proba.length; k++) state.grid[m.start + k] = m.proba[k][1];
      setBar(Math.round(((m.start + m.proba.length) / GRID_SIZE) * 100));
      draw();
    } else if (m.type === "gridDone") {
      if (m.tag !== state.gridTag) return;
      state.gridBusy = false;
      state.surface = null;
      setStatus("TFM field ready · move ★");
      updateModelUI();
      draw();
    } else if (m.type === "error") {
      state.loading = false; state.gridBusy = false; updateModelUI();
      if (state.model === "tfm") setStatus("error: " + m.message);
      console.error(m.message);
    }
  };
}

function loadModel() {
  if (state.loaded || state.loading) return;
  ensureWorker();
  state.loading = true;
  setBar(1);
  updateModelUI();
  setStatus("downloading model · ~28 mb");
  worker.postMessage({ type: "load" });
}

function reprepare() {
  if (worker) worker.postMessage({ type: "cancelGrid" });
  state.prepared = false;
  state.grid = null; state.surface = null; state.gridTag++; state.qProba = null; state.qClass = null; state.neighbors = [];
  state.selectedView = null; state.selectedViewProbability = null;
  state.attention = null; state.attentionBlock = null; state.attentionHeads = null;
  state.prepTag++; state.queryTag++; state.qInFlight = false; state.qQueued = false;
  state.gridBusy = false;                                        // cancel any in-flight sweep
  prepareClassicalModels();
  updateReadout(); updateCompanion(); updateModelUI();
  if (!state.X.length) {
    updateModelUI(); updateCompanion(); draw(); return;
  }
  if (!state.loaded) {
    updateModelUI();
    if (state.model === "knn") { requestQuery(); requestLandscape(); } else draw();
    return;
  }
  worker.postMessage({ type: "prepare", X: state.X, y: state.y, tag: state.prepTag });
  if (state.model === "knn") { requestQuery(); requestLandscape(); } else draw();
}

function requestQuery() {
  requestClassicalQuery();
  if (state.model === "knn") {
    const result = knnPredict(state.query);
    state.qProba = result ? result.proba : null;
    state.qClass = result ? result.classIndex : null;
    state.neighbors = result ? result.neighbors : [];
    state.attention = null;
    updateReadout(); updateCompanion(); updateModelUI(); draw();
    return;
  }
  if (!state.prepared) return;
  if (state.qInFlight) { state.qQueued = true; return; }
  state.qInFlight = true;
  state.attention = null;
  state.queryTag++;
  updateCompanion(); updateModelUI(); draw();
  worker.postMessage({
    type: "inspect",
    X: state.query,
    tag: state.queryTag,
    viewIndex: state.inspectionView,
  });
}

function requestLandscape() {
  if (state.grid || state.gridBusy || !state.X.length || (state.model === "tfm" && !state.prepared)) return;
  state.gridBusy = true;
  if (state.model === "tfm") { $("#pg-bar-wrap").hidden = false; setBar(0); }
  setStatus(state.model === "tfm" ? "mapping TFM field…" : "mapping kNN field…");
  state.grid = new Float32Array(GRID_SIZE).fill(-1);
  state.surface = null;
  updateModelUI();
  state.gridTag++;
  const pts = [];
  for (let a = 0; a < GRIDN; a++) for (let b = 0; b < GRIDN; b++)
    pts.push([gridValue(a), gridValue(b)]);
  if (state.model === "knn") {
    for (let i = 0; i < pts.length; i++) state.grid[i] = knnPredict(pts[i]).proba[1];
    state.gridBusy = false;
    state.surface = null;
    setStatus(`kNN field ready · k = ${state.k} · move ★`); draw();
    return;
  }
  worker.postMessage({ type: "grid", X: pts, tag: state.gridTag, chunk: 16, viewIndex: 0 });
}

/* ---------- UI helpers ---------- */
function setStatus(s) { $("#pg-status").textContent = s; }
function setBar(pct) {
  const value = Math.round(Math.max(0, Math.min(100, pct)));
  const bar = $("#pg-bar");
  const wrap = $("#pg-bar-wrap");
  const label = $("#pg-bar-label");
  if (bar) bar.style.width = `${value}%`;
  if (wrap) wrap.setAttribute("aria-valuenow", String(value));
  if (label) label.value = value === 100 ? "100% downloaded" : `${value}%`;
}
function syncViewUI() {
  const viewButtons = document.querySelectorAll("[data-view]");
  if (!viewButtons.length) {
    state.view = "flat";
    $("#pg-reset-view").hidden = true;
    $("#pg-view-step").textContent = "side-by-side 2D";
    $("#pg-view-copy").textContent = "All four canvases use the same 2D context and query.";
    canvas.classList.add("is-flat");
    return;
  }
  const is3d = state.view === "3d";
  viewButtons.forEach((button) => {
    const active = button.dataset.view === state.view;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  $("#pg-reset-view").hidden = !is3d;
  $("#pg-view-step").textContent = is3d ? "02 · 3D volume" : "01 · 2D slice";
  $("#pg-view-copy").textContent = is3d
    ? "Drag ★ in x/y, Shift-drag for z, and drag empty space to orbit."
    : "Drag ★ across x and y; use the z control to move the slice.";
  canvas.classList.toggle("is-flat", !is3d);
  canvas.setAttribute("aria-label", is3d
    ? "Interactive three-feature space with labeled context points and a movable query. Drag the query for x and y, Shift-drag it vertically for z, or use Page Up and Page Down. The query color and readout show its predicted class."
    : "Interactive interpolated x and y prediction slice at the query's current z value. Color continuously represents the probability of class 1, and context points fade with z distance.");
}

function updateReadout() {
  const el = $("#pg-readout"); if (!state.qProba) { el.innerHTML = ""; return; }
  const p1 = state.qProba[1], cls = state.qClass ?? (p1 >= 0.5 ? 1 : 0);
  const c = cls === 0 ? COL0 : COL1;
  const coordinates = `<span class="readout__coords">query (${state.query.map((value) => value.toFixed(1)).join(", ")})</span>`;
  el.style.setProperty("--prediction-color", `rgb(${c.join(",")})`);
  if (state.model === "knn") {
    const votes = state.neighbors.reduce((sum, item) => sum + (state.y[item.index] === cls ? 1 : 0), 0);
    el.innerHTML = `<span class="pg-dot" style="background:rgb(${c.join(",")})"></span>` +
      `<span class="readout__label">prediction</span><strong>class ${cls}</strong>` +
      `<span class="readout__detail">${votes} of ${state.neighbors.length} neighbor votes</span>${coordinates}`;
    return;
  }
  el.innerHTML = `<span class="pg-dot" style="background:rgb(${c.join(",")})"></span>` +
    `<span class="readout__label">TFM</span><strong>class ${cls}</strong>` +
    `<span class="readout__detail">p(1) ${(p1 * 100).toFixed(0)}%</span>`;
}

function updateCompanion() {
  const title = $("#pg-companion-title");
  const summary = $("#pg-companion-summary");
  const contextTitle = $("#pg-context-title");
  const equations = $("#pg-equations");
  const knnRows = $("#pg-knn-comparison-rows");
  const rows = $("#pg-context-weights");
  const note = $("#pg-companion-note");
  if (!title || !summary || !contextTitle || !equations || !knnRows || !rows || !note) return;

  knnRows.innerHTML = "";
  rows.innerHTML = "";

  function renderKnnRows(result) {
    if (!result) return;
    for (const [rank, neighbor] of result.neighbors.slice(0, 5).entries()) {
      const li = document.createElement("li");
      li.innerHTML = `<b>${rank + 1}</b><span>row ${neighbor.index + 1}</span>` +
        `<span>class ${state.y[neighbor.index]}</span><code>d = ${Math.sqrt(neighbor.distance).toFixed(2)}</code>`;
      knnRows.append(li);
    }
  }

  if (state.model === "knn") {
    title.textContent = "Same query · kNN vs TFM";
    contextTitle.textContent = "Switch to TFM to compare";
    rows.setAttribute("aria-label", "Nearest context rows used in the current kNN vote");
    summary.innerHTML =
      `<article><span>kNN · p(class 1)</span><strong>${Number.isFinite(state.qProba?.[1]) ? `${(state.qProba[1] * 100).toFixed(1)}%` : "–"}</strong></article>` +
      `<article class="is-final"><span>TFM · p(class 1)</span><strong>select TFM</strong></article>`;
    equations.innerHTML =
      `<div><span>select</span><code>N<sub>k</sub>(x★) = k nearest rows by ‖x<sub>i</sub> − x★‖<sub>2</sub></code></div>` +
      `<div><span>vote</span><code>ŷ = argmax<sub>c</sub> Σ<sub>i∈Nₖ</sub> 1[y<sub>i</sub> = c]</code></div>`;
    renderKnnRows({ neighbors: state.neighbors });
    note.textContent = state.neighbors.length
      ? "The left column shows kNN evidence. Select TFM above to compute its prediction and attention for the same query."
      : "Add a labeled row to form a neighbor vote.";
    return;
  }

  const selectedViewNumber = (state.selectedView?.viewIndex ?? state.inspectionView) + 1;
  title.textContent = "Same query · kNN vs TFM";
  contextTitle.textContent = `Top attention rows · view ${selectedViewNumber}`;
  rows.setAttribute("aria-label", `Five highest final-block context attention weights for selected trace view ${selectedViewNumber} of ${state.viewCount}`);
  const p1 = state.qProba?.[1];
  const block = state.attentionBlock || 12;
  const knnComparison = knnPredict(state.query);
  renderKnnRows(knnComparison);
  summary.innerHTML =
    `<article><span>kNN · p(class 1)</span><strong>${Number.isFinite(knnComparison?.proba?.[1]) ? `${(knnComparison.proba[1] * 100).toFixed(1)}%` : "–"}</strong></article>` +
    `<article class="is-final"><span>TFM · p(class 1)</span><strong>${Number.isFinite(p1) ? `${(p1 * 100).toFixed(1)}%` : "–"}</strong></article>`;
  equations.innerHTML =
    `<div><span>route</span><code>α<sub>i,h</sub> = softmax<sub>i</sub>((QASS(q<sub>h</sub>) · k<sub>i,h</sub>) / √d<sub>h</sub>)</code></div>` +
    `<div><span>combine</span><code>a★ = W<sub>O</sub> concat<sub>h</sub>(Σ<sub>i</sub> α<sub>i,h</sub>v<sub>i,h</sub>); h★<sup>${block}</sup> = h★<sup>${block - 1}</sup> + a★ + MLP(LN(h★<sup>${block - 1}</sup> + a★))</code></div>`;

  const ranked = rankContextAttention(state.attention, 5);
  const strongest = ranked[0]?.weight || 1;
  for (const [rank, item] of ranked.entries()) {
    const li = document.createElement("li");
    li.style.setProperty("--attention-width", `${Math.max(3, item.weight / strongest * 100)}%`);
    li.innerHTML = `<b>${rank + 1}</b><span>row ${item.index + 1}</span>` +
      `<span>class ${state.y[item.index]}</span><code>α = ${(item.weight * 100).toFixed(1)}%</code><i aria-hidden="true"></i>`;
    rows.append(li);
  }
  if (ranked.length) {
    note.textContent = `Same context and query: kNN uses distance; TFM uses learned attention. TFM bars show routing in block ${block}, not causal importance.`;
  } else if (state.qInFlight) {
    note.textContent = `Updating selected view ${selectedViewNumber} of ${state.viewCount}; the final prediction remains the eight-view ensemble…`;
  } else if (!state.loaded) {
    note.textContent = state.loading
      ? "TFM is loading automatically. kNN remains available while you wait."
      : "TFM will load automatically when this playground approaches the viewport.";
  } else {
    note.textContent = "Preparing the labeled context for TFM inference…";
  }
}

function updateModelUI() {
  const isKnn = state.model === "knn";
  const knnControls = $("#pg-knn-controls");
  const traceControls = $("#pg-trace-controls");
  if (knnControls) knnControls.hidden = false;
  if (traceControls) traceControls.hidden = isKnn;
  const loadOverlay = $("#pg-load-overlay");
  if (loadOverlay) loadOverlay.hidden = Boolean(state.grid && !state.gridBusy);
  $("#pg-attention-key").hidden = state.view !== "3d" || isKnn || !state.attention?.length;
  $("#pg-bar-wrap").hidden = false;
  const comparisonMode = !document.querySelector("[data-view]");
  const action = comparisonMode
    ? "drag any ★ · all four canvases update together"
    : state.view === "3d"
      ? "drag ★ x/y · Shift + vertical for z · drag space to orbit"
      : "drag ★ x/y · move z slider for depth · select to add";
  $("#pg-hint").textContent = `${action} · ${isKnn ? "neighbor vote" : "frozen weights"}`;
  const fieldKind = "field";
  setStatus(!state.X.length ? "add a point to classify" : state.gridBusy ?
    (isKnn ? `mapping kNN ${fieldKind}…` : `mapping TFM ${fieldKind}…`) : state.grid ?
    (isKnn ? `kNN ${fieldKind} ready · k = ${state.k} · move ★` : `TFM ${fieldKind} ready · move ★`) : isKnn ?
      `kNN ready · k = ${state.k}` : (state.prepared ? "TFM ready · move ★" : state.loaded ?
        "preparing 8 ensemble views" : state.loading ? "downloading model…" : "waiting to load automatically…"));
}

function setModel(model) {
  if (state.model === model) return;
  if (worker) worker.postMessage({ type: "cancelGrid" });
  state.model = model; state.grid = null; state.surface = null; state.gridTag++; state.gridBusy = false;
  state.queryTag++; state.qInFlight = false; state.qQueued = false;
  state.qProba = null; state.qClass = null; state.neighbors = [];
  state.selectedView = null; state.selectedViewProbability = null;
  state.attention = null; state.attentionBlock = null; state.attentionHeads = null;
  document.querySelectorAll("[data-model]").forEach((button) => {
    const active = button.dataset.model === model;
    button.classList.toggle("is-active", active); button.setAttribute("aria-pressed", String(active));
  });
  updateModelUI(); updateReadout(); updateCompanion();
  if (state.model === "knn") { requestQuery(); requestLandscape(); }
  else { draw(); requestQuery(); requestLandscape(); }
}

function setView(view) {
  if (state.view === view) return;
  state.view = view;
  syncViewUI();
  updateModelUI(); updateCompanion(); draw(); requestLandscape();
}

function resetView() {
  state.viewYaw = DEFAULT_YAW; state.viewPitch = DEFAULT_PITCH; draw();
}

function updateTraceViewOptions() {
  const select = $("#pg-trace-view");
  if (!select) return;
  const count = state.viewCount || 8;
  if (select.options.length !== count) {
    select.replaceChildren(...Array.from({ length: count }, (_, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = String(index + 1);
      return option;
    }));
  }
  select.value = String(state.inspectionView);
}

function newDataset() {
  const seed = (Math.random() * 1e9) | 0;
  const d = gen(state.ds, state.n, seed);
  state.X = d.X; state.y = d.y;
  reprepare(); draw();
}

function setQueryZ(value) {
  state.query[2] = clamp(value, -DOM, DOM);
  const zInput = $("#pg-z");
  const zOutput = $("#pg-z-value");
  if (zInput) zInput.value = String(state.query[2]);
  if (zOutput) zOutput.value = `x₃ = ${state.query[2].toFixed(1)}`;
  if (state.model === "knn") requestQuery(); else { draw(); requestQuery(); }
}

/* ---------- pointer interaction ---------- */
function bindCanvas() {
  let interaction = null, moved = false, start = null, last = null, dragOffset = [0, 0], zDrag = null;
  const localXY = (ev) => { const r = canvas.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * SZ, (ev.clientY - r.top) / r.height * SZ]; };
  const addPoint = (point) => {
    if (point.some((value) => Math.abs(value) > DOM)) return;
    state.X.push(point); state.y.push(state.activeClass); reprepare();
  };
  canvas.addEventListener("pointerdown", (ev) => {
    const [mx, my] = localXY(ev);
    const projection = createProjection(SZ, DOM, state.viewYaw, state.viewPitch);
    const query = state.view === "3d" ? projection.project(state.query) : { x: toPx(state.query[0]), y: SZ - toPx(state.query[1]) };
    if (Math.hypot(mx - query.x, my - query.y) < 24) {
      interaction = state.view === "3d" && ev.shiftKey ? "query-z" : "query";
      if (interaction === "query-z") {
        zDrag = { startY: my, startZ: state.query[2] };
        canvas.classList.add("is-z-dragging");
      } else if (state.view === "3d") {
        dragOffset = [mx - query.x, my - query.y];
      } else dragOffset = [0, 0];
    } else if (state.view === "3d") {
      interaction = "orbit"; start = [mx, my]; last = [mx, my]; moved = false;
      canvas.classList.add("is-orbiting");
    } else {
      addPoint([toData(mx), toData(SZ - my)]);
    }
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!interaction) return;
    const [mx, my] = localXY(ev);
    if (interaction === "orbit") {
      if (Math.hypot(mx - start[0], my - start[1]) > 5) moved = true;
      if (moved) {
        state.viewYaw += (mx - last[0]) * 0.008;
        state.viewPitch = clamp(state.viewPitch - (my - last[1]) * 0.004, 0.24, 0.78);
        last = [mx, my]; scheduleDraw();
      }
      return;
    }
    if (interaction === "query-z") {
      setQueryZ(zDrag.startZ + ((zDrag.startY - my) / SZ) * DOM * 2);
      return;
    }
    const point = state.view === "3d"
      ? createProjection(SZ, DOM, state.viewYaw, state.viewPitch).unproject([mx - dragOffset[0], my - dragOffset[1]], state.query[2])
      : [toData(mx), toData(SZ - my)];
    state.query = [clamp(point[0], -DOM, DOM), clamp(point[1], -DOM, DOM)];
    if (state.model === "knn") requestQuery(); else { draw(); requestQuery(); }
    requestLandscape();
  });
  canvas.addEventListener("pointerup", (ev) => {
    if (interaction === "orbit" && !moved) {
      const point = createProjection(SZ, DOM, state.viewYaw, state.viewPitch).unproject(localXY(ev), state.query[2]);
      addPoint(point);
    }
    interaction = null; zDrag = null; canvas.classList.remove("is-orbiting", "is-z-dragging");
  });
  canvas.addEventListener("pointercancel", () => { interaction = null; zDrag = null; canvas.classList.remove("is-orbiting", "is-z-dragging"); });
  canvas.addEventListener("keydown", (ev) => {
    const delta = ev.shiftKey ? 0.25 : 0.1;
    const moves = { ArrowLeft: [-delta, 0], ArrowRight: [delta, 0], ArrowUp: [0, delta], ArrowDown: [0, -delta] };
    const move = moves[ev.key];
    if (!move) return;
    ev.preventDefault();
    state.query = [
      Math.max(-DOM, Math.min(DOM, state.query[0] + move[0])),
      Math.max(-DOM, Math.min(DOM, state.query[1] + move[1])),
    ];
    if (state.model === "knn") requestQuery(); else { draw(); requestQuery(); }
    requestLandscape();
  });
}

/* ---------- setup ---------- */
function bindKnnCanvas() {
  let dragging = false;

  function pointerPosition(event) {
    const rect = knnCanvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (KNNSZ / rect.width),
      y: (event.clientY - rect.top) * (KNNSZ / rect.height),
    };
  }

  knnCanvas.addEventListener("pointerdown", (event) => {
    const point = pointerPosition(event);
    const queryX = toPx(state.query[0]) * (KNNSZ / SZ);
    const queryY = (SZ - toPx(state.query[1])) * (KNNSZ / SZ);
    if (Math.hypot(point.x - queryX, point.y - queryY) > 24) return;
    dragging = true;
    knnCanvas.setPointerCapture(event.pointerId);
  });
  knnCanvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const point = pointerPosition(event);
    state.query[0] = clamp((point.x / KNNSZ) * 2 * DOM - DOM, -DOM, DOM);
    state.query[1] = clamp(((KNNSZ - point.y) / KNNSZ) * 2 * DOM - DOM, -DOM, DOM);
    draw();
    requestQuery();
    event.preventDefault();
  });
  const stop = (event) => {
    if (!dragging) return;
    dragging = false;
    if (knnCanvas.hasPointerCapture(event.pointerId)) {
      knnCanvas.releasePointerCapture(event.pointerId);
    }
  };
  knnCanvas.addEventListener("pointerup", stop);
  knnCanvas.addEventListener("pointercancel", stop);
}

function bindClassicalCanvases() {
  Object.values(classicalViews).forEach((view) => {
    let dragging = false;
    const pointerPosition = (event) => {
      const rect = view.canvas.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * (view.size / rect.width),
        y: (event.clientY - rect.top) * (view.size / rect.height),
      };
    };
    view.canvas.addEventListener("pointerdown", (event) => {
      const point = pointerPosition(event);
      const queryX = ((state.query[0] + DOM) / (2 * DOM)) * view.size;
      const queryY = view.size - ((state.query[1] + DOM) / (2 * DOM)) * view.size;
      if (Math.hypot(point.x - queryX, point.y - queryY) > 24) return;
      dragging = true;
      view.canvas.setPointerCapture(event.pointerId);
    });
    view.canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const point = pointerPosition(event);
      state.query[0] = clamp((point.x / view.size) * 2 * DOM - DOM, -DOM, DOM);
      state.query[1] = clamp(((view.size - point.y) / view.size) * 2 * DOM - DOM, -DOM, DOM);
      draw();
      requestQuery();
      event.preventDefault();
    });
    const stop = (event) => {
      if (!dragging) return;
      dragging = false;
      if (view.canvas.hasPointerCapture(event.pointerId)) {
        view.canvas.releasePointerCapture(event.pointerId);
      }
    };
    view.canvas.addEventListener("pointerup", stop);
    view.canvas.addEventListener("pointercancel", stop);
  });
}

function setupCanvas() {
  canvas = $("#pg-canvas");
  dpr = Math.min(2, window.devicePixelRatio || 1);
  // CSS sizes the element (aspect-ratio 1:1); we only set the backing store.
  const rect = canvas.getBoundingClientRect();
  SZ = Math.max(220, Math.round(rect.width || 440));
  canvas.width = Math.round(SZ * dpr); canvas.height = Math.round(SZ * dpr);
  ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  off = document.createElement("canvas"); off.width = GRIDN; off.height = GRIDN; offctx = off.getContext("2d");

  ["linear", "tree", "knn"].forEach((kind) => {
    const comparisonCanvas = $(`#pg-${kind}-canvas`);
    const comparisonRect = comparisonCanvas.getBoundingClientRect();
    const size = Math.max(220, Math.round(comparisonRect.width || 320));
    comparisonCanvas.width = Math.round(size * dpr);
    comparisonCanvas.height = Math.round(size * dpr);
    const comparisonCtx = comparisonCanvas.getContext("2d");
    comparisonCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const comparisonOff = document.createElement("canvas");
    comparisonOff.width = GRIDN;
    comparisonOff.height = GRIDN;
    classicalViews[kind] = {
      canvas: comparisonCanvas,
      ctx: comparisonCtx,
      size,
      off: comparisonOff,
      offctx: comparisonOff.getContext("2d"),
    };
  });
  knnCanvas = classicalViews.knn.canvas;
  knnCtx = classicalViews.knn.ctx;
  KNNSZ = classicalViews.knn.size;
  knnOff = classicalViews.knn.off;
  knnOffctx = classicalViews.knn.offctx;
}

function init() {
  if (!$("#pg-canvas") || !$("#pg-linear-canvas") || !$("#pg-tree-canvas") || !$("#pg-knn-canvas")) return;
  setupCanvas();
  bindCanvas();
  bindClassicalCanvases();
  const d = gen(state.ds, state.n, 7); state.X = d.X; state.y = d.y;
  prepareClassicalModels();
  syncViewUI(); updateModelUI(); updateCompanion(); draw();

  const playground = document.querySelector(".nano-try");
  if ("IntersectionObserver" in window && playground) {
    const observer = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      observer.disconnect();
      loadModel();
    }, { rootMargin: "500px 0px" });
    observer.observe(playground);
  } else {
    loadModel();
  }
  const clearButton = $("#pg-clear");
  const newButton = $("#pg-new");
  if (clearButton) clearButton.addEventListener("click", () => { state.X = []; state.y = []; reprepare(); draw(); });
  if (newButton) newButton.addEventListener("click", newDataset);
  document.querySelectorAll("[data-model]").forEach((b) =>
    b.addEventListener("click", () => setModel(b.dataset.model)));
  document.querySelectorAll("[data-view]").forEach((b) =>
    b.addEventListener("click", () => setView(b.dataset.view)));
  $("#pg-reset-view").addEventListener("click", resetView);
  document.querySelectorAll("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => { state.ds = b.dataset.preset; setActive("[data-preset]", b); newDataset(); }));
  const datasetSelect = $("#pg-dataset");
  if (datasetSelect) datasetSelect.addEventListener("change", () => {
    state.ds = datasetSelect.value;
    newDataset();
  });
  document.querySelectorAll("[data-cls]").forEach((b) =>
    b.addEventListener("click", () => {
      state.activeClass = +b.dataset.cls;
      document.querySelectorAll("[data-cls]").forEach((button) => {
        const selected = button === b;
        button.classList.toggle("is-active", selected);
        button.setAttribute("aria-pressed", String(selected));
      });
    }));
  const sizeSel = $("#pg-size");
  if (sizeSel) sizeSel.addEventListener("change", () => { state.n = +sizeSel.value; newDataset(); });
  const kInput = $("#pg-k");
  if (kInput) kInput.addEventListener("input", () => {
    state.k = +kInput.value; $("#pg-k-value").value = `k = ${state.k}`;
    if (classicalWorker && state.classicalReady) {
      classicalWorker.postMessage({ type: "setK", k: state.k });
      requestClassicalQuery();
      classicalWorker.postMessage({
        type: "grid",
        tag: state.classicalTag,
        X: comparisonGridPoints(),
      });
    }
    updateModelUI(); updateCompanion(); draw();
  });
  const linksInput = $("#pg-links");
  if (linksInput) linksInput.addEventListener("change", () => { state.showLinks = linksInput.checked; draw(); });
  const traceViewSelect = $("#pg-trace-view");
  if (traceViewSelect) traceViewSelect.addEventListener("change", () => {
    state.inspectionView = +traceViewSelect.value;
    state.selectedView = null;
    state.selectedViewProbability = null;
    state.attention = null;
    updateCompanion();
    updateModelUI();
    draw();
    requestQuery();
  });
  const zInput = $("#pg-z");
  if (zInput) zInput.addEventListener("input", () => {
    setQueryZ(+zInput.value);
  });
  window.addEventListener("resize", () => { setupCanvas(); draw(); });
}

function setActive(sel, btn) { document.querySelectorAll(sel).forEach((b) => b.classList.toggle("is-active", b === btn)); }

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
