/* site.js
 * Non-model interactivity for scroll reveals, the outline, the hero fallback,
 * the PFN episode, and the UCI table stress test. The real model lives in
 * playground.js and loads only in section 06.
 */
import {
  PFN_TOY_HYPOTHESES,
  pfnToyPosterior,
  pfnToyPredict,
} from "./pfn-toy.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const C0 = [46, 166, 214];
const C1 = [242, 114, 75];
const ACCENT = [124, 108, 240];

function centerOverflowingStep(button) {
  const scroller = button?.parentElement;
  if (!scroller || scroller.scrollWidth <= scroller.clientWidth + 1) return;
  const left = button.offsetLeft - (scroller.clientWidth - button.offsetWidth) / 2;
  scroller.scrollTo({
    left: Math.max(0, left),
    behavior: reduceMotion ? "auto" : "smooth",
  });
}

/* ---------- HiDPI canvas helper ---------- */
function fitCanvas(canvas, cssW, cssH) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/* ============================ scroll reveals ============================ */
function initReveals() {
  const els = $$(".reveal");
  if (reduceMotion || !("IntersectionObserver" in window)) {
    els.forEach((e) => e.classList.add("is-in"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) { e.target.classList.add("is-in"); io.unobserve(e.target); }
      }
    },
    { rootMargin: "0px 0px -12% 0px", threshold: 0.08 }
  );
  els.forEach((e) => io.observe(e));
}

/* ============================ floating section outline ============================ */
function initOutlineNav() {
  const menu = $("#outline-menu");
  const nav = $("#outline-nav");
  const toggle = $("#outline-toggle");
  if (!menu || !nav || !toggle) return;
  const links = $$("a[href^='#']", nav);
  const items = links
    .map((link) => ({
      link,
      section: document.querySelector(link.getAttribute("href")),
    }))
    .filter((item) => item.section);
  let scheduled = false;

  function setOpen(open) {
    menu.classList.toggle("is-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute(
      "aria-label",
      open ? "Close tutorial outline" : "Open tutorial outline"
    );
    const label = $(".sr-only", toggle);
    if (label) label.textContent = open ? "Close tutorial outline" : "Open tutorial outline";
  }

  toggle.addEventListener("click", () => {
    setOpen(!menu.classList.contains("is-open"));
  });
  links.forEach((link) => link.addEventListener("click", () => setOpen(false)));
  document.addEventListener("pointerdown", (event) => {
    if (!menu.contains(event.target)) setOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !menu.classList.contains("is-open")) return;
    setOpen(false);
    toggle.focus();
  });

  function update() {
    scheduled = false;
    const marker = Math.min(window.innerHeight * 0.34, 300);
    let current = null;
    for (const item of items) {
      if (item.section.getBoundingClientRect().top <= marker) current = item;
      else break;
    }
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) {
      current = items[items.length - 1];
    }
    items.forEach((item) => {
      if (item === current) item.link.setAttribute("aria-current", "true");
      else item.link.removeAttribute("aria-current");
    });
  }

  function requestUpdate() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(update);
  }

  window.addEventListener("scroll", requestUpdate, { passive: true });
  window.addEventListener("resize", requestUpdate);
  update();
}

function initStickyTitle() {
  const bar = $(".bar");
  const heroTitle = $(".hero__title");
  const menu = $("#outline-menu");
  const toggle = $("#outline-toggle");
  if (!bar || !heroTitle || !menu) return;
  let wasScrolled = false;

  function setMenuOpen(open) {
    menu.classList.toggle("is-open", open);
    toggle?.setAttribute("aria-expanded", String(open));
    toggle?.setAttribute(
      "aria-label",
      open ? "Close tutorial outline" : "Open tutorial outline"
    );
    const label = toggle ? $(".sr-only", toggle) : null;
    if (label) label.textContent = open ? "Close tutorial outline" : "Open tutorial outline";
  }

  function setScrolled(scrolled) {
    bar.classList.toggle("has-page-title", scrolled);
    menu.classList.toggle("is-visible", scrolled);
    if (scrolled && !wasScrolled) setMenuOpen(true);
    if (!scrolled) setMenuOpen(false);
    wasScrolled = scrolled;
  }

  function update() {
    setScrolled(heroTitle.getBoundingClientRect().bottom <= bar.offsetHeight);
  }

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(update, {
      rootMargin: `${-bar.offsetHeight}px 0px 0px`,
      threshold: 0,
    });
    observer.observe(heroTitle);
  } else {
    window.addEventListener("scroll", update, { passive: true });
  }
  update();
}

/* ============================ hero teaser field ============================ */
function initHero() {
  const canvas = $("#hero-canvas");
  if (!canvas) return;
  let ctx, W, H;
  const G = 60;                       // low-res field grid
  const off = document.createElement("canvas");
  off.width = G; off.height = G;
  const offctx = off.getContext("2d");

  // slowly drifting gaussian "bumps" that define a moving boundary
  const bumps = [
    { a: 1.0, r: 0.42, px: 0.30, py: 0.35, sx: 0.11, sy: 0.07, phx: 0.0, phy: 1.7 },
    { a: -1.0, r: 0.40, px: 0.70, py: 0.55, sx: 0.09, sy: 0.12, phx: 2.1, phy: 0.4 },
    { a: 0.8, r: 0.34, px: 0.55, py: 0.78, sx: 0.13, sy: 0.09, phx: 1.1, phy: 3.0 },
    { a: -0.7, r: 0.32, px: 0.22, py: 0.72, sx: 0.10, sy: 0.10, phx: 0.6, phy: 2.2 },
  ];

  function resize() {
    W = canvas.clientWidth || canvas.parentElement.clientWidth;
    H = canvas.clientHeight || 600;
    ctx = fitCanvas(canvas, W, H);
  }
  resize();
  window.addEventListener("resize", resize);

  function field(nx, ny, t) {
    let v = 0;
    for (const b of bumps) {
      const cx = b.px + b.sx * Math.sin(t * 0.00016 + b.phx);
      const cy = b.py + b.sy * Math.cos(t * 0.00014 + b.phy);
      const dx = nx - cx, dy = ny - cy;
      v += b.a * Math.exp(-(dx * dx + dy * dy) / (2 * b.r * b.r));
    }
    return v;
  }

  function render(t) {
    const img = offctx.createImageData(G, G);
    for (let j = 0; j < G; j++) {
      for (let i = 0; i < G; i++) {
        const v = field(i / (G - 1), j / (G - 1), t);
        const s = Math.max(-1, Math.min(1, v * 1.1));
        const mix = (s + 1) / 2;                       // 0 -> C0, 1 -> C1
        const r = C0[0] + (C1[0] - C0[0]) * mix;
        const g = C0[1] + (C1[1] - C0[1]) * mix;
        const b = C0[2] + (C1[2] - C0[2]) * mix;
        const edge = 1 - Math.min(1, Math.abs(s) * 2.4); // brighten near boundary
        const alpha = 24 + Math.abs(s) * 46 + edge * 42;
        const k = (j * G + i) * 4;
        img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = alpha;
      }
    }
    offctx.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, G, G, 0, 0, W, H);
  }

  // The 3D hero (hero3d.js) calls this once WebGL is ready, so the 2D field
  // stops animating and hides in favor of the richer scene.
  let raf = 0, stopped = false;
  window.__heroStop2D = () => {
    stopped = true;
    if (raf) cancelAnimationFrame(raf);
    canvas.style.display = "none";
  };

  if (reduceMotion) { render(0); return; }
  let last = 0;
  function loop(ts) {
    if (stopped) return;
    if (ts - last > 40) { render(ts); last = ts; }   // ~25fps
    raf = requestAnimationFrame(loop);
  }
  raf = requestAnimationFrame(loop);
}

/* ============================ tensor journey ============================ */
const STAGES = {
  input: {
    title: "Table values and labels",
    shape: "shape: n rows × d columns",
    body: "The estimator preprocesses a table into numeric feature values and standardizes them using context-row statistics. Labels are not treated as another feature column; learned target embeddings are added to the context rows.",
  },
  col: {
    title: "Feature grouping and column induced attention",
    shape: "shape: n × d × e",
    body: "After context-based standardization, the paper and current nanoTabICL group features with cyclic offsets 0, 1, and 3, then project each group to 128 dimensions. A first learned label embedding is added to context cells. Three column blocks each use one shared bank of 128 inducing vectors.",
  },
  row: {
    title: "Row aggregation",
    shape: "shape: n × h",
    body: "Within each row, four learned CLS tokens attend across feature-group embeddings. Rotary positional encoding preserves feature-position information, and the four CLS outputs are concatenated into one fixed-width row representation.",
  },
  icl: {
    title: "Dataset-level in-context learning",
    shape: "shape: n × h",
    body: "A second learned label embedding is added after row aggregation. Across 12 blocks, query rows attend to labeled context representations while model weights stay fixed.",
  },
  out: {
    title: "Task predictions",
    shape: "shape: q queries × output dimension",
    body: "A normalized query representation passes through a two-layer MLP. Classification produces class logits; regression produces quantile predictions. In both cases, inference uses frozen weights and no task-specific gradient step.",
  },
};

function initArch() {
  const flow = $("#arch-flow");
  const detail = $("#arch-detail");
  if (!flow || !detail) return;
  const tabs = $$(".arch__stage", flow);
  function show(stage) {
    const s = STAGES[stage];
    detail.innerHTML = `<h4>${s.title}</h4><p>${s.body}</p><span class="mono">${s.shape}</span>`;
    tabs.forEach((button) => {
      const selected = button.dataset.stage === stage;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
  }
  tabs.forEach((button, index) => {
    button.addEventListener("click", () => show(button.dataset.stage));
    button.addEventListener("keydown", (event) => {
      const keyMoves = {
        ArrowRight: (index + 1) % tabs.length,
        ArrowDown: (index + 1) % tabs.length,
        ArrowLeft: (index - 1 + tabs.length) % tabs.length,
        ArrowUp: (index - 1 + tabs.length) % tabs.length,
        Home: 0,
        End: tabs.length - 1,
      };
      if (!(event.key in keyMoves)) return;
      event.preventDefault();
      const next = tabs[keyMoves[event.key]];
      show(next.dataset.stage);
      next.focus();
    });
  });
  show("input");
}

/* ============================ adaptation comparison ============================ */
const ADAPTATION_VIEWS = {
  gbdt: {
    mode: "dataset-specific training",
    input: "Labeled table",
    process: "Optimize objective",
    output: "Dataset-specific tree ensemble",
    equation: "θ<sub>D</sub> = argmin<sub>θ</sub> L<sub>D</sub>(θ)",
    state: "Model parameters change for this dataset.",
  },
  tfm: {
    mode: "in-context inference",
    input: "Labeled context + unlabeled queries",
    process: "Fixed pretrained model",
    output: "Predictions for query rows",
    equation: "ŷ<sub>q</sub> = f<sub>θ*</sub>(X<sub>c</sub>, y<sub>c</sub>, X<sub>q</sub>)",
    state: "Model parameters remain fixed.",
  },
};

function initAdaptation() {
  const compare = $("#adaptation-compare");
  if (!compare) return;
  const buttons = $$("[data-adaptation]", compare);
  const fields = {
    mode: $("#adaptation-mode"),
    input: $("#adaptation-input"),
    process: $("#adaptation-process"),
    output: $("#adaptation-output"),
    equation: $("#adaptation-equation"),
    state: $("#adaptation-state"),
  };
  function show(name) {
    const view = ADAPTATION_VIEWS[name];
    fields.mode.textContent = view.mode;
    fields.input.textContent = view.input;
    fields.process.textContent = view.process;
    fields.output.textContent = view.output;
    fields.equation.innerHTML = view.equation;
    fields.state.textContent = view.state;
    buttons.forEach((button) => {
      const selected = button.dataset.adaptation === name;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  }
  buttons.forEach((button) =>
    button.addEventListener("click", () => show(button.dataset.adaptation)));
  show("gbdt");
}

/* ============================ legacy attention chart ============================ */
function initQass() {
  const canvas = $("#qass-chart");
  const slider = $("#qass-slider");
  const out = $("#qass-out");
  const readout = $("#qass-readout");
  if (!canvas || !slider) return;

  const Z = 2.2;          // query is clearly closest to one context row
  const NREF = 128;       // reference length where the two schemes agree
  const xMin = 8, xMax = 4096;

  // peak attention weight the query places on its nearest neighbor
  const softmax = (n) => Math.exp(Z) / (Math.exp(Z) + (n - 1));
  const qassmax = (n) => {
    const g = Math.log(n) / Math.log(NREF);            // logit gain ~ log n
    return Math.exp(Z * g) / (Math.exp(Z * g) + (n - 1));
  };

  const lx = (n) => (Math.log(n) - Math.log(xMin)) / (Math.log(xMax) - Math.log(xMin));

  let W, H, ctx;
  function resize() {
    W = canvas.clientWidth || 440; H = Math.round(W * (300 / 440));
    ctx = fitCanvas(canvas, W, H);
    draw();
  }

  function curve(fn, color, pad) {
    ctx.beginPath();
    for (let i = 0; i <= 120; i++) {
      const n = Math.exp(Math.log(xMin) + (i / 120) * (Math.log(xMax) - Math.log(xMin)));
      const x = pad.l + lx(n) * (W - pad.l - pad.r);
      const y = H - pad.b - fn(n) * (H - pad.t - pad.b);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 2.4; ctx.stroke();
  }

  function draw() {
    const pad = { l: 34, r: 12, t: 16, b: 26 };
    ctx.clearRect(0, 0, W, H);
    // axes
    ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
    ctx.fillStyle = "rgba(170,178,192,0.7)"; ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillText("peak weight", pad.l - 30, pad.t + 4);
    ctx.fillText("context rows", W - 92, H - 8);
    // gridlines at powers
    [16, 64, 256, 1024, 4096].forEach((n) => {
      const x = pad.l + lx(n) * (W - pad.l - pad.r);
      ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
      ctx.fillStyle = "rgba(107,115,130,0.9)"; ctx.fillText(String(n), x - 10, H - 10);
    });
    curve(softmax, "rgba(170,178,192,0.85)", pad);
    curve(qassmax, `rgb(${ACCENT.join(",")})`, pad);

    // marker at current n
    const n = +slider.value;
    const mx = pad.l + lx(n) * (W - pad.l - pad.r);
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(mx, pad.t); ctx.lineTo(mx, H - pad.b); ctx.stroke(); ctx.setLineDash([]);
    const dot = (fn, col) => {
      const y = H - pad.b - fn(n) * (H - pad.t - pad.b);
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(mx, y, 4, 0, 2 * Math.PI); ctx.fill();
    };
    dot(softmax, "rgba(220,226,236,0.95)");
    dot(qassmax, `rgb(${ACCENT.join(",")})`);
  }

  function update() {
    const n = +slider.value;
    out.textContent = n;
    const sm = softmax(n), qm = qassmax(n);
    readout.innerHTML = `schematic softmax ${(sm * 100).toFixed(1)}% &nbsp;·&nbsp; length-scaled ${(qm * 100).toFixed(1)}%`;
    draw();
  }
  slider.addEventListener("input", update);
  window.addEventListener("resize", resize);
  resize(); update();
}

/* ============================ synthetic prior (SCM) sampler ============================ */
function initScm() {
  const canvas = $("#scm-viz");
  const newScmButton = $("#scm-btn");
  const rowButton = $("#scm-row-btn");
  const taskButton = $("#scm-task-btn");
  const noiseInput = $("#scm-noise");
  const noiseValue = $("#scm-noise-value");
  const table = $("#scm-table");
  const empty = $("#scm-empty");
  const summary = $("#scm-summary");
  const trace = $("#scm-trace");
  const equations = {
    z: $("#scm-eq-z"),
    x1: $("#scm-eq-x1"),
    x2: $("#scm-eq-x2"),
    y: $("#scm-eq-y"),
  };
  const values = {
    e1: $("#scm-value-e1"),
    e2: $("#scm-value-e2"),
    ey: $("#scm-value-ey"),
    z: $("#scm-value-z"),
    x1: $("#scm-value-x1"),
    x2: $("#scm-value-x2"),
    y: $("#scm-value-y"),
  };
  if (!canvas || !newScmButton || !rowButton || !taskButton || !noiseInput ||
      !noiseValue || !table || !empty || !summary || !trace ||
      Object.values(equations).some((node) => !node) ||
      Object.values(values).some((node) => !node)) return;

  let W;
  let H;
  let ctx;
  let mechanism;
  let rows = [];
  const rnd = (a, b) => a + Math.random() * (b - a);
  const signed = (value) => `${value < 0 ? "−" : "+"} ${Math.abs(value).toFixed(2)}`;
  let spareNormal = null;

  function normal() {
    if (spareNormal !== null) {
      const value = spareNormal;
      spareNormal = null;
      return value;
    }
    const u = Math.max(Math.random(), Number.EPSILON);
    const v = Math.random();
    const radius = Math.sqrt(-2 * Math.log(u));
    spareNormal = radius * Math.sin(2 * Math.PI * v);
    return radius * Math.cos(2 * Math.PI * v);
  }

  function sampleMechanism() {
    return {
      family: ["linear", "wave", "interaction"][Math.floor(Math.random() * 3)],
      a: rnd(0.65, 1.35) * (Math.random() < 0.5 ? -1 : 1),
      b: rnd(0.35, 1.05) * (Math.random() < 0.5 ? -1 : 1),
      c: rnd(0.65, 1.3) * (Math.random() < 0.5 ? -1 : 1),
      d: rnd(0.25, 0.8) * (Math.random() < 0.5 ? -1 : 1),
      k: rnd(0.8, 1.4) * (Math.random() < 0.5 ? -1 : 1),
      threshold: rnd(-0.25, 0.25),
    };
  }

  function updateEquations() {
    const m = mechanism;
    if (m.family === "linear") {
      equations.z.textContent = `z = tanh(${m.a.toFixed(2)}ε₁ ${signed(m.b)}ε₂)`;
      equations.x1.textContent = "X₁ = z + σεₓ₁";
      equations.x2.textContent = `X₂ = ${m.c.toFixed(2)}z ${signed(m.d)}ε₂ + σεₓ₂`;
      equations.y.textContent = `y = 1[${m.k.toFixed(2)}z + 0.55X₂ + σεᵧ > ${m.threshold.toFixed(2)}]`;
    } else if (m.family === "wave") {
      equations.z.textContent = `z = sin(${m.a.toFixed(2)}ε₁) ${signed(m.b)}tanh(ε₂)`;
      equations.x1.textContent = "X₁ = z² + σεₓ₁";
      equations.x2.textContent = `X₂ = sin(${m.c.toFixed(2)}z) ${signed(m.d)}ε₂ + σεₓ₂`;
      equations.y.textContent = `y = 1[${m.k.toFixed(2)}zX₂ + 0.35X₁ + σεᵧ > ${m.threshold.toFixed(2)}]`;
    } else {
      equations.z.textContent = `z = tanh(${m.a.toFixed(2)}ε₁ε₂ ${signed(m.b)}ε₁)`;
      equations.x1.textContent = "X₁ = z + 0.40ε₁ + σεₓ₁";
      equations.x2.textContent = `X₂ = ${m.c.toFixed(2)}z² ${signed(m.d)}ε₂ + σεₓ₂`;
      equations.y.textContent = `y = 1[${m.k.toFixed(2)}z − 0.45X₁X₂ + σεᵧ > ${m.threshold.toFixed(2)}]`;
    }
  }

  function generateRow() {
    const sigma = Number(noiseInput.value);
    const e1 = normal();
    const e2 = normal();
    const ey = normal();
    const ex1 = normal();
    const ex2 = normal();
    const m = mechanism;
    let z;
    let x1;
    let x2;
    let score;
    if (m.family === "linear") {
      z = Math.tanh(m.a * e1 + m.b * e2);
      x1 = z + sigma * ex1;
      x2 = m.c * z + m.d * e2 + sigma * ex2;
      score = m.k * z + 0.55 * x2 + sigma * ey - m.threshold;
    } else if (m.family === "wave") {
      z = Math.sin(m.a * e1) + m.b * Math.tanh(e2);
      x1 = z ** 2 + sigma * ex1;
      x2 = Math.sin(m.c * z) + m.d * e2 + sigma * ex2;
      score = m.k * z * x2 + 0.35 * x1 + sigma * ey - m.threshold;
    } else {
      z = Math.tanh(m.a * e1 * e2 + m.b * e1);
      x1 = z + 0.4 * e1 + sigma * ex1;
      x2 = m.c * z ** 2 + m.d * e2 + sigma * ex2;
      score = m.k * z - 0.45 * x1 * x2 + sigma * ey - m.threshold;
    }
    return { e1, e2, ey, z, x1, x2, y: Number(score > 0) };
  }

  function showTrace(row, animate = true) {
    Object.entries(values).forEach(([key, output]) => {
      output.value = key === "y" ? String(row[key]) : row[key].toFixed(2);
    });
    if (animate && !reduceMotion) {
      trace.classList.remove("is-sampling");
      void trace.offsetWidth;
      trace.classList.add("is-sampling");
    }
  }

  function renderRows() {
    table.replaceChildren();
    rows.forEach((row, index) => {
      const tr = document.createElement("tr");
      [index + 1, row.x1.toFixed(3), row.x2.toFixed(3), row.y].forEach((value) => {
        const td = document.createElement("td");
        td.textContent = String(value);
        tr.append(td);
      });
      table.append(tr);
    });
    empty.hidden = rows.length > 0;
    const positives = rows.reduce((sum, row) => sum + row.y, 0);
    summary.textContent = rows.length
      ? `${rows.length} rows from one SCM · ${positives} label 1 · ${rows.length - positives} label 0`
      : "No rows yet. Generate one row to see its complete causal trace.";
  }

  function addRows(count, replace = false) {
    if (replace) rows = [];
    let latest;
    for (let i = 0; i < count; i += 1) {
      latest = generateRow();
      rows.push(latest);
    }
    if (rows.length > 40) rows = rows.slice(-40);
    showTrace(latest);
    renderRows();
  }

  function drawDetailedGraph() {
    ctx.clearRect(0, 0, W, H);
    const scale = W / 640;
    ctx.save();
    ctx.scale(scale, scale);

    ctx.fillStyle = "#0b1220";
    ctx.fillRect(0, 0, 640, 320);
    ctx.strokeStyle = "rgba(148,163,184,.055)";
    ctx.lineWidth = 1;
    for (let x = 16; x < 640; x += 24) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 320);
      ctx.stroke();
    }
    for (let y = 16; y < 320; y += 24) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(640, y);
      ctx.stroke();
    }

    function roundedRect(x, y, width, height, radius) {
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, radius);
    }

    const familyName = {
      linear: "LINEAR MIX",
      wave: "NONLINEAR WAVE",
      interaction: "FEATURE INTERACTION",
    }[mechanism.family];
    roundedRect(16, 12, 178, 24, 12);
    ctx.fillStyle = "rgba(154,124,255,.16)";
    ctx.fill();
    ctx.strokeStyle = "rgba(196,181,253,.4)";
    ctx.stroke();
    ctx.fillStyle = "#c4b5fd";
    ctx.font = "700 11px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`SAMPLED SCM · ${familyName}`, 28, 24);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText("functions stay fixed for this task", 624, 24);

    const panels = [
      { x: 16, width: 154, title: "1 · FRESH NOISE", note: "resampled every row", color: "#94a3b8" },
      { x: 202, width: 144, title: "2 · HIDDEN CAUSE", note: "not saved", color: "#c4b5fd" },
      { x: 378, width: 246, title: "3 · OBSERVED ROW", note: "written to the table", color: "#7dd3fc" },
    ];
    panels.forEach((panel) => {
      roundedRect(panel.x, 48, panel.width, 256, 10);
      ctx.fillStyle = "rgba(15,23,42,.86)";
      ctx.fill();
      ctx.strokeStyle = "rgba(148,163,184,.18)";
      ctx.stroke();
      ctx.fillStyle = panel.color;
      ctx.font = "700 10px 'JetBrains Mono', monospace";
      ctx.textAlign = "left";
      ctx.fillText(panel.title, panel.x + 12, 67);
      ctx.fillStyle = "#64748b";
      ctx.font = "600 9px 'JetBrains Mono', monospace";
      ctx.fillText(panel.note, panel.x + 12, 289);
    });

    const nodes = {
      e1: { x: 32, y: 88, width: 56, height: 44, label: "ε₁", detail: "N(0,1)", kind: "noise" },
      e2: { x: 98, y: 88, width: 56, height: 44, label: "ε₂", detail: "N(0,1)", kind: "noise" },
      ex: { x: 32, y: 156, width: 56, height: 44, label: "εₓ", detail: "N(0,1)", kind: "noise" },
      ey: { x: 98, y: 224, width: 56, height: 44, label: "εᵧ", detail: "N(0,1)", kind: "noise" },
      z: { x: 220, y: 121, width: 108, height: 66, label: "z", detail: "f(ε₁, ε₂)", kind: "hidden" },
      x1: { x: 398, y: 84, width: 92, height: 58, label: "X₁", detail: "g₁(z, εₓ)", kind: "feature" },
      x2: { x: 512, y: 84, width: 92, height: 58, label: "X₂", detail: "g₂(z, ε₂, εₓ)", kind: "feature" },
      y: { x: 455, y: 208, width: 96, height: 58, label: "y", detail: "h(z, X₂, εᵧ)", kind: "label" },
    };

    const edges = [
      ["e1", "z", "noise", -22], ["e2", "z", "noise", 18],
      ["ex", "x1", "noise", -40], ["ex", "x2", "noise", 24],
      ["e2", "x2", "noise", -54], ["ey", "y", "noise", 35],
      ["z", "x1", "causal", -20], ["z", "x2", "causal", 16],
      ["z", "y", "causal", 35], ["x2", "y", "target", 20],
    ];
    if (mechanism.family !== "linear") edges.push(["x1", "y", "target", -18]);

    function anchor(node, side) {
      return {
        x: side === "left" ? node.x : side === "right" ? node.x + node.width : node.x + node.width / 2,
        y: side === "top" ? node.y : side === "bottom" ? node.y + node.height : node.y + node.height / 2,
      };
    }

    function arrow(from, to, type, bend) {
      const a = nodes[from];
      const b = nodes[to];
      const vertical = from === "x1" || from === "x2";
      const start = vertical ? anchor(a, "bottom") : anchor(a, "right");
      const end = vertical ? anchor(b, "top") : anchor(b, "left");
      const color = type === "noise"
        ? "rgba(148,163,184,.46)"
        : type === "target" ? "rgba(255,132,108,.72)" : "rgba(154,124,255,.72)";
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = type === "noise" ? 1.2 : 1.8;
      ctx.setLineDash(type === "noise" ? [4, 4] : []);
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      let c1x;
      let c1y;
      let c2x;
      let c2y;
      if (vertical) {
        c1x = start.x + bend;
        c1y = start.y + 28;
        c2x = end.x + bend;
        c2y = end.y - 28;
      } else {
        const middle = (start.x + end.x) / 2;
        c1x = middle;
        c1y = start.y + bend;
        c2x = middle;
        c2y = end.y + bend;
      }
      ctx.bezierCurveTo(c1x, c1y, c2x, c2y, end.x, end.y);
      ctx.stroke();
      ctx.setLineDash([]);
      const angle = Math.atan2(end.y - c2y, end.x - c2x);
      ctx.beginPath();
      ctx.moveTo(end.x, end.y);
      ctx.lineTo(end.x - 7 * Math.cos(angle - 0.4), end.y - 7 * Math.sin(angle - 0.4));
      ctx.lineTo(end.x - 7 * Math.cos(angle + 0.4), end.y - 7 * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fill();
    }
    edges.forEach(([from, to, type, bend]) => arrow(from, to, type, bend));

    const colors = {
      noise: "#94a3b8",
      hidden: "#9a7cff",
      feature: "#35d9ff",
      label: "#ff846c",
    };
    Object.values(nodes).forEach((node) => {
      roundedRect(node.x, node.y, node.width, node.height, 8);
      ctx.fillStyle = node.kind === "noise" ? "#172033" : `${colors[node.kind]}22`;
      ctx.fill();
      ctx.strokeStyle = colors[node.kind];
      ctx.lineWidth = node.kind === "noise" ? 1 : 1.4;
      ctx.stroke();
      ctx.fillStyle = colors[node.kind];
      ctx.font = "700 14px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(node.label, node.x + node.width / 2, node.y + node.height / 2 - 8);
      ctx.fillStyle = node.kind === "noise" ? "#94a3b8" : "#cbd5e1";
      ctx.font = "600 8px 'JetBrains Mono', monospace";
      ctx.fillText(node.detail, node.x + node.width / 2, node.y + node.height / 2 + 10);
    });

    ctx.textAlign = "right";
    ctx.font = "600 8px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("┄ noise injection", 615, 280);
    ctx.fillStyle = "#c4b5fd";
    ctx.fillText("━ fixed dependency", 615, 293);
    ctx.restore();
  }

  function drawGraph() {
    ctx.clearRect(0, 0, W, H);
    const scale = W / 640;
    ctx.save();
    ctx.scale(scale, scale);
    ctx.fillStyle = "#0b1220";
    ctx.fillRect(0, 0, 640, 320);

    function roundedRect(x, y, width, height, radius) {
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, radius);
    }

    function text(value, x, y, size, color, align = "left", weight = 600) {
      ctx.fillStyle = color;
      ctx.font = `${weight} ${size}px 'JetBrains Mono', monospace`;
      ctx.textAlign = align;
      ctx.textBaseline = "middle";
      ctx.fillText(value, x, y);
    }

    function card(x, y, width, height, border) {
      roundedRect(x, y, width, height, 10);
      ctx.fillStyle = "#111c2e";
      ctx.fill();
      ctx.strokeStyle = border;
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }

    function arrow(x1, x2, y) {
      ctx.strokeStyle = "#64748b";
      ctx.fillStyle = "#64748b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, y);
      ctx.lineTo(x2, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, y);
      ctx.lineTo(x2 - 8, y - 5);
      ctx.lineTo(x2 - 8, y + 5);
      ctx.closePath();
      ctx.fill();
    }

    const familyName = {
      linear: "linear mix",
      wave: "nonlinear wave",
      interaction: "feature interaction",
    }[mechanism.family];

    text("HOW ONE SYNTHETIC ROW IS MADE", 20, 24, 11, "#94a3b8", "left", 700);
    text(`sampled SCM: ${familyName}`, 620, 24, 10, "#c4b5fd", "right", 700);

    card(20, 62, 150, 168, "#475569");
    text("NEW FOR EVERY ROW", 34, 81, 9, "#94a3b8", "left", 700);
    text("fresh random noise", 34, 101, 11, "#e2e8f0");
    [
      ["ε₁", 34, 122],
      ["ε₂", 92, 122],
      ["εₓ", 34, 164],
      ["εᵧ", 92, 164],
    ].forEach(([label, x, y]) => {
      roundedRect(x, y, 44, 31, 7);
      ctx.fillStyle = "#1e293b";
      ctx.fill();
      ctx.strokeStyle = "#64748b";
      ctx.stroke();
      text(label, x + 22, y + 16, 12, "#cbd5e1", "center", 700);
    });
    text("different values each time", 95, 214, 9, "#64748b", "center");

    arrow(180, 220, 146);

    card(230, 62, 180, 168, "#9a7cff");
    text("FIXED FOR THIS TASK", 244, 81, 9, "#c4b5fd", "left", 700);
    text("same SCM", 320, 111, 17, "#fff", "center", 700);
    roundedRect(254, 130, 132, 38, 8);
    ctx.fillStyle = "rgba(154,124,255,.13)";
    ctx.fill();
    text("ε  →  z  →  X, y", 320, 149, 12, "#c4b5fd", "center", 700);
    text("same graph + equations", 320, 188, 10, "#cbd5e1", "center");
    text("for every row in this table", 320, 207, 9, "#64748b", "center");

    arrow(420, 460, 146);

    card(470, 62, 150, 168, "#35d9ff");
    text("APPEND TO DATASET", 484, 81, 9, "#7dd3fc", "left", 700);
    text("one observed row", 484, 101, 11, "#e2e8f0");
    [
      ["X₁", "#35d9ff"],
      ["X₂", "#35d9ff"],
      ["y", "#ff846c"],
    ].forEach(([label, color], index) => {
      const x = 484 + index * 41;
      roundedRect(x, 132, 34, 45, 7);
      ctx.fillStyle = `${color}18`;
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.stroke();
      text(label, x + 17, 154, 11, color, "center", 700);
    });
    text("(X₁, X₂, y)", 545, 201, 10, "#cbd5e1", "center", 700);

    roundedRect(20, 252, 600, 43, 9);
    ctx.fillStyle = "rgba(30,41,59,.72)";
    ctx.fill();
    text("repeat with fresh noise", 38, 273, 10, "#94a3b8", "left", 700);
    text("row 1   row 2   row 3   ···   row n", 320, 273, 10, "#e2e8f0", "center", 600);
    text("new SCM = new task", 602, 273, 10, "#c4b5fd", "right", 700);
    ctx.restore();
  }

  function resize() {
    W = canvas.clientWidth || 640;
    H = Math.round(W * 0.5);
    ctx = fitCanvas(canvas, W, H);
    drawGraph();
  }

  function resetTrace() {
    Object.values(values).forEach((output) => { output.value = "–"; });
  }

  function newMechanism() {
    mechanism = sampleMechanism();
    rows = [];
    updateEquations();
    resetTrace();
    renderRows();
    resize();
  }

  newScmButton.addEventListener("click", newMechanism);
  rowButton.addEventListener("click", () => addRows(1));
  taskButton.addEventListener("click", () => addRows(20, true));
  noiseInput.addEventListener("input", () => {
    noiseValue.value = Number(noiseInput.value).toFixed(2);
  });
  trace.addEventListener("animationend", () => trace.classList.remove("is-sampling"));
  window.addEventListener("resize", resize);
  noiseValue.value = Number(noiseInput.value).toFixed(2);
  newMechanism();
}

/* ============================ PFN task-stream overview ============================ */
function initPfnOverview() {
  const overview = $("#pfn-overview");
  if (!overview) return;
  const tasks = $$("[data-pfn-overview-task]", overview);
  const taskLabel = $("#pfn-overview-task-label");
  const p0Bar = $("#pfn-overview-p0");
  const p1Bar = $("#pfn-overview-p1");
  const p0Label = $("#pfn-overview-p0-label");
  const p1Label = $("#pfn-overview-p1-label");
  const loss = $("#pfn-overview-loss");
  const contextLabels = $$(".pfn-episode__rows > span:not(.is-query) em", overview);
  if (!tasks.length || !taskLabel || !p0Bar || !p1Bar || !p0Label || !p1Label || !loss) return;

  const episodes = [
    { name: "D¹", p1: 0.63, truth: 1, context: [0, 1, 0, 1, 1] },
    { name: "D²", p1: 0.41, truth: 0, context: [1, 0, 1, 0, 0] },
    { name: "D³", p1: 0.78, truth: 1, context: [0, 0, 1, 1, 0] },
  ];
  function show(index, animate = true) {
    const episode = episodes[index];
    overview.dataset.task = String(index);
    tasks.forEach((button, taskIndex) => {
      const selected = taskIndex === index;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    contextLabels.forEach((label, row) => { label.textContent = String(episode.context[row]); });
    taskLabel.textContent = episode.name;
    const p0 = 1 - episode.p1;
    p0Bar.style.width = `${Math.round(p0 * 100)}%`;
    p1Bar.style.width = `${Math.round(episode.p1 * 100)}%`;
    p0Label.value = `${Math.round(p0 * 100)}%`;
    p1Label.value = `${Math.round(episode.p1 * 100)}%`;
    const trueProbability = episode.truth === 1 ? episode.p1 : p0;
    loss.innerHTML = `y<sub>q</sub> = ${episode.truth}; NLL = −log q<sub>θ</sub>(y<sub>q</sub> = ${episode.truth} | x<sub>q</sub>, D<sub>c</sub>) = −log ${trueProbability.toFixed(2)} = ${(-Math.log(trueProbability)).toFixed(2)}`;
    if (animate && !reduceMotion) {
      overview.classList.remove("is-changing");
      void overview.offsetWidth;
      overview.classList.add("is-changing");
    }
  }

  tasks.forEach((button, index) => button.addEventListener("click", () => show(index)));
  overview.addEventListener("animationend", () => overview.classList.remove("is-changing"));
  show(0, false);
}

/* ============================ PFN posterior playground ============================ */
function initPfnPlayground() {
  const root = $("#pfn-playground");
  if (!root) return;

  const axis = $("#pfn-sim-axis");
  const newTask = $("#pfn-sim-new");
  const episodeLabel = $("#pfn-sim-episode");
  const survivorsLabel = $("#pfn-sim-survivors");
  const equation = $("#pfn-sim-equation");
  const p0Bar = $("#pfn-sim-p0");
  const p1Bar = $("#pfn-sim-p1");
  const p0Label = $("#pfn-sim-p0-label");
  const p1Label = $("#pfn-sim-p1-label");
  const rulesEl = $("#pfn-sim-rules");
  const guessButtons = $$("[data-pfn-sim-guess]", root);
  const check = $("#pfn-sim-check");
  const result = $("#pfn-sim-result");
  const resultLabel = $("#pfn-sim-result-label");
  const resultTitle = $("#pfn-sim-result-title");
  const lossEl = $("#pfn-sim-loss");
  const explanation = $("#pfn-sim-explanation");
  const question = $("#pfn-sim-question");
  if (!axis || !newTask || !episodeLabel || !survivorsLabel || !equation ||
      !p0Bar || !p1Bar || !p0Label || !p1Label || !rulesEl ||
      !guessButtons.length || !check || !result || !resultLabel ||
      !resultTitle || !lossEl || !explanation || !question) return;

  const hypotheses = PFN_TOY_HYPOTHESES;
  let episode = 0;
  let trueRule;
  let context = [];
  let queryX = 1;
  let guess = null;
  let revealed = false;

  function shuffled(values) {
    const copy = [...values];
    for (let i = copy.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  }

  function ruleText(rule) {
    return `y=1 if x ${rule.direction === "up" ? "≥" : "≤"} ${rule.tau}`;
  }

  function plainRuleText(rule) {
    return `purchase when pages viewed is ${rule.direction === "up" ? "at least" : "at most"} ${rule.tau}`;
  }

  function renderAxis() {
    axis.replaceChildren();
    const rows = [
      ...context.map((point, index) => ({
        name: `C${index + 1}`,
        x: point.x,
        y: point.y,
        query: false,
      })),
      {
        name: "Q",
        x: queryX,
        y: revealed ? pfnToyPredict(trueRule, queryX) : null,
        query: true,
      },
    ];

    rows.forEach((item) => {
      const row = document.createElement("div");
      row.className = `pfn-sim__table-row${item.query ? " is-query" : ""}`;
      row.setAttribute("role", "row");

      const name = document.createElement("span");
      name.setAttribute("role", "cell");
      name.innerHTML = item.query ? "<b>Q</b><small>predict this row</small>" : `<b>${item.name}</b><small>context</small>`;

      const xValue = document.createElement("span");
      xValue.setAttribute("role", "cell");
      xValue.innerHTML = `<b>${item.x}</b><small>pages viewed</small>`;

      const yValue = document.createElement("span");
      yValue.setAttribute("role", "cell");
      if (item.y === null) {
        yValue.className = "is-hidden";
        yValue.innerHTML = "<b>?</b><small>hidden label</small>";
      } else {
        yValue.className = `is-label-${item.y}`;
        yValue.innerHTML = `<b>${item.y}</b><small>${item.y ? "Yes, purchased" : "No purchase"}</small>`;
      }
      row.append(name, xValue, yValue);
      axis.append(row);
    });
  }

  function renderRules(survivors) {
    const survivingSet = new Set(survivors);
    rulesEl.replaceChildren();
    hypotheses.forEach((rule) => {
      const chip = document.createElement("span");
      chip.textContent = ruleText(rule);
      if (!survivingSet.has(rule)) chip.className = "is-rejected";
      rulesEl.append(chip);
    });
  }

  function renderResult(p1) {
    result.hidden = !revealed;
    if (!revealed) return;
    const truth = pfnToyPredict(trueRule, queryX);
    const correct = guess === truth;
    const truthProbability = truth === 1 ? p1 : 1 - p1;
    const loss = -Math.log(truthProbability);
    result.classList.toggle("is-wrong", !correct);
    resultLabel.textContent = correct ? "you got it" : "different from your guess";
    resultTitle.textContent =
      `Actual answer: ${truth ? "Purchase" : "No purchase"} · ` +
      `your answer: ${guess ? "Purchase" : "No purchase"}`;
    lossEl.textContent =
      `Probability on the true answer: ${Math.round(truthProbability * 100)}% · training loss: ${loss.toFixed(2)}`;
    explanation.textContent =
      `This table's hidden pattern was “${plainRuleText(trueRule)}.” During PFN pretraining, ` +
      "this error updates the shared model; the six sessions are then discarded.";
  }

  function render() {
    const { survivors, votesForOne, p1 } =
      pfnToyPosterior(hypotheses, context, queryX);
    const p0 = 1 - p1;
    renderAxis();
    renderRules(survivors);
    episodeLabel.textContent = `episode ${String(episode).padStart(2, "0")}`;
    question.textContent = `For query row Q, X = ${queryX} pages. What is Y?`;
    survivorsLabel.textContent =
      `${survivors.length} possible patterns still fit the five examples`;
    equation.textContent =
      `At ${queryX} pages, ${votesForOne} of ${survivors.length} patterns predict purchase`;
    p0Bar.style.width = `${p0 * 100}%`;
    p1Bar.style.width = `${p1 * 100}%`;
    p0Label.value = `${Math.round(p0 * 100)}%`;
    p1Label.value = `${Math.round(p1 * 100)}%`;
    guessButtons.forEach((button) => {
      const selected = Number(button.dataset.pfnSimGuess) === guess;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
      button.disabled = revealed;
    });
    check.disabled = guess === null || revealed;
    renderResult(p1);
  }

  function generateTask() {
    episode += 1;
    trueRule = hypotheses[Math.floor(Math.random() * hypotheses.length)];
    const positions = shuffled([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    context = positions.slice(0, 5)
      .map((x) => ({ x, y: pfnToyPredict(trueRule, x) }))
      .sort((a, b) => a.x - b.x);
    queryX = positions[5];
    guess = null;
    revealed = false;
    result.hidden = true;
    render();
  }

  guessButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (revealed) return;
      guess = Number(button.dataset.pfnSimGuess);
      render();
    });
  });
  check.addEventListener("click", () => {
    if (guess === null || revealed) return;
    revealed = true;
    render();
  });
  newTask.addEventListener("click", generateTask);
  generateTask();
}

/* ============================ architecture tensor scenes ============================ */
const TECH = {
  bg: "#07111f",
  panel: "#0d1b2d",
  line: "#29415c",
  text: "#e6f2ff",
  muted: "#9fb3c8",
  cyan: "#35d9ff",
  violet: "#9a7cff",
  coral: "#ff846c",
  gold: "#ffd166",
};

function createTensorScene(canvas, render) {
  const BASE_W = 960;
  const BASE_H = 500;
  let ctx;
  let width = BASE_W;
  let height = BASE_H;

  function text(value, x, y, size = 14, color = TECH.text, align = "left", weight = 600) {
    ctx.fillStyle = color;
    ctx.font = `${weight} ${size}px Inter, sans-serif`;
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    ctx.fillText(value, x, y);
  }

  function mono(value, x, y, size = 13, color = TECH.muted, align = "left", weight = 600) {
    ctx.fillStyle = color;
    ctx.font = `${weight} ${size}px 'JetBrains Mono', monospace`;
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    ctx.fillText(value, x, y);
  }

  function arrow(x1, y1, x2, y2, color = TECH.cyan, width0 = 1.6) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width0;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 8 * Math.cos(angle - 0.42), y2 - 8 * Math.sin(angle - 0.42));
    ctx.lineTo(x2 - 8 * Math.cos(angle + 0.42), y2 - 8 * Math.sin(angle + 0.42));
    ctx.closePath();
    ctx.fill();
  }

  function panel(x, y, w, h, color = TECH.line) {
    ctx.fillStyle = "rgba(13,27,45,.88)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 8);
    ctx.fill();
    ctx.stroke();
  }

  function cube(x, y, w, h, depth, label, dims, color = TECH.cyan) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.7;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.12;
    ctx.fillRect(x, y, w, h);
    ctx.globalAlpha = 0.2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + depth, y - depth);
    ctx.lineTo(x + w + depth, y - depth);
    ctx.lineTo(x + w, y);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 0.28;
    ctx.beginPath();
    ctx.moveTo(x + w, y);
    ctx.lineTo(x + w + depth, y - depth);
    ctx.lineTo(x + w + depth, y + h - depth);
    ctx.lineTo(x + w, y + h);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeRect(x, y, w, h);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + depth, y - depth);
    ctx.lineTo(x + w + depth, y - depth);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w + depth, y - depth);
    ctx.lineTo(x + w + depth, y + h - depth);
    ctx.lineTo(x + w, y + h);
    ctx.stroke();
    ctx.shadowBlur = 0;
    text(label, x + w / 2, y + h / 2 - 11, 15, TECH.text, "center", 700);
    mono(dims, x + w / 2, y + h / 2 + 14, 13, color, "center", 700);
    ctx.restore();
  }

  function title(kicker, heading, detail) {
    mono(kicker.toUpperCase(), 28, 27, 12, TECH.cyan, "left", 700);
    text(heading, 28, 52, 18, TECH.text, "left", 750);
    if (detail) mono(detail, 932, 29, 12, TECH.muted, "right", 600);
  }

  function table(x, y, X, labels, options = {}) {
    const cw = options.cw || 39;
    const ch = options.ch || 29;
    const showLabels = options.showLabels !== false;
    const cols = showLabels ? 5 : 4;
    const query = 9;
    mono("x₁", x + cw * 0.5, y - 13, 12, TECH.muted, "center");
    mono("x₂", x + cw * 1.5, y - 13, 12, TECH.muted, "center");
    mono("x₃", x + cw * 2.5, y - 13, 12, TECH.muted, "center");
    mono("x₄", x + cw * 3.5, y - 13, 12, TECH.muted, "center");
    if (showLabels) mono("y", x + cw * 4.5, y - 13, 12, TECH.gold, "center");
    for (let row = 0; row < 10; row++) {
      mono(row === query ? "Q" : `C${row + 1}`, x - 9, y + row * ch + ch / 2, 11,
        row === query ? TECH.coral : TECH.muted, "right", 700);
      for (let col = 0; col < cols; col++) {
        const px = x + col * cw;
        const py = y + row * ch;
        const labelCell = col === 4;
        const queryLabel = labelCell && row === query;
        const alpha = 0.2 + Math.abs(X?.[row]?.[col] || 0.2) * 0.34;
        ctx.fillStyle = queryLabel ? "rgba(255,132,108,.06)" :
          labelCell ? "rgba(255,209,102,.22)" : `rgba(53,217,255,${alpha})`;
        ctx.strokeStyle = queryLabel ? TECH.coral : labelCell ? TECH.gold : "rgba(53,217,255,.58)";
        ctx.setLineDash(queryLabel ? [4, 3] : []);
        ctx.fillRect(px, py, cw - 4, ch - 4);
        ctx.strokeRect(px, py, cw - 4, ch - 4);
        ctx.setLineDash([]);
        if (labelCell) {
          text(queryLabel ? "?" : String(labels?.[row] ?? row % 2), px + (cw - 4) / 2,
            py + (ch - 4) / 2, 13, queryLabel ? TECH.coral : TECH.gold, "center", 750);
        }
      }
    }
    return { width: cols * cw, height: 10 * ch };
  }

  function columnAttention(x, y, scale = 1) {
    const colGap = 48 * scale;
    const rowGap = 27 * scale;
    mono("SHARED BANK: 128 INDUCING TOKENS PER BLOCK", x + colGap * 1.5, y - 31, 9, TECH.violet, "center", 700);
    for (let col = 0; col < 4; col++) {
      const cx = x + col * colGap;
      const contextTop = y + 17;
      const contextBottom = y + 17 + 8 * rowGap;
      const queryY = y + 17 + 9 * rowGap;
      ctx.fillStyle = "rgba(154,124,255,.24)";
      ctx.strokeStyle = TECH.violet;
      ctx.fillRect(cx - 11, y - 16, 22, 13);
      ctx.strokeRect(cx - 11, y - 16, 22, 13);
      mono("I", cx, y - 9, 10, TECH.violet, "center", 700);
      for (let row = 0; row < 10; row++) {
        const cy = y + 17 + row * rowGap;
        ctx.fillStyle = row === 9 ? "rgba(255,132,108,.22)" : "rgba(53,217,255,.2)";
        ctx.strokeStyle = row === 9 ? TECH.coral : TECH.cyan;
        ctx.beginPath();
        ctx.arc(cx, cy, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        if (row === 9) text("Q", cx, cy, 7, TECH.text, "center", 750);
      }

      // Left bracket: all context cells contribute to the inducing summaries.
      ctx.save();
      ctx.strokeStyle = "rgba(53,217,255,.62)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - 13, contextTop);
      ctx.lineTo(cx - 13, contextBottom);
      for (let row = 0; row < 9; row++) {
        const cy = y + 17 + row * rowGap;
        ctx.moveTo(cx - 13, cy);
        ctx.lineTo(cx - 8, cy);
      }
      ctx.stroke();

      // Right rail: the inducing summaries feed every cell, including Q.
      ctx.strokeStyle = "rgba(154,124,255,.58)";
      ctx.beginPath();
      ctx.moveTo(cx + 13, y - 2);
      ctx.lineTo(cx + 13, queryY);
      for (let row = 0; row < 10; row++) {
        const cy = y + 17 + row * rowGap;
        ctx.moveTo(cx + 8, cy);
        ctx.lineTo(cx + 13, cy);
      }
      ctx.stroke();
      ctx.restore();
      arrow(cx - 13, contextTop, cx - 8, y - 2, "rgba(53,217,255,.72)", 1);
      arrow(cx + 13, y - 2, cx + 13, queryY - 8, "rgba(154,124,255,.72)", 1);
    }
  }

  function rowCompression(x, y, compact = false) {
    const rowGap = compact ? 27 : 34;
    const tokenW = compact ? 12 : 14;
    for (let row = 0; row < 10; row++) {
      const cy = y + row * rowGap;
      mono(row === 9 ? "Q" : `C${row + 1}`, x - 9, cy + 6, 11,
        row === 9 ? TECH.coral : TECH.muted, "right", 700);
      for (let col = 0; col < 8; col++) {
        const px = x + col * (tokenW + 4);
        const cls = col < 4;
        ctx.fillStyle = cls ? "rgba(154,124,255,.25)" : "rgba(53,217,255,.2)";
        ctx.strokeStyle = cls ? TECH.violet : TECH.cyan;
        ctx.fillRect(px, cy, tokenW, 13);
        ctx.strokeRect(px, cy, tokenW, 13);
      }
      const end = x + 8 * (tokenW + 4);
      arrow(end + 2, cy + 6.5, end + 24, cy + 6.5, TECH.violet, 1.1);
      ctx.fillStyle = row === 9 ? "rgba(255,132,108,.3)" : "rgba(154,124,255,.3)";
      ctx.strokeStyle = row === 9 ? TECH.coral : TECH.violet;
      ctx.fillRect(end + 28, cy - 3, 62, 19);
      ctx.strokeRect(end + 28, cy - 3, 62, 19);
      mono("512D", end + 59, cy + 6.5, 10, row === 9 ? TECH.coral : TECH.violet, "center", 700);
    }
  }

  function datasetAttention(x, y, labels = null, showOutput = true) {
    const queryY = y + 9 * 31;
    for (let row = 0; row < 10; row++) {
      const cy = y + row * 31;
      ctx.fillStyle = row === 9 ? "rgba(255,132,108,.28)" : "rgba(154,124,255,.24)";
      ctx.strokeStyle = row === 9 ? TECH.coral : TECH.violet;
      ctx.fillRect(x, cy, 72, 20);
      ctx.strokeRect(x, cy, 72, 20);
      const label = labels?.[row] ?? row % 2;
      mono(row === 9 ? "Q · y=?" : `C${row + 1} · y=${label}`, x + 36, cy + 10, 10,
        row === 9 ? TECH.coral : TECH.text, "center", 650);
      if (row < 9) {
        arrow(x + 76, cy + 10, x + 150, queryY + 10, "rgba(53,217,255,.52)", 1);
      }
    }
    ctx.fillStyle = "rgba(255,132,108,.18)";
    ctx.strokeStyle = TECH.coral;
    ctx.beginPath();
    ctx.arc(x + 166, queryY + 10, 15, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    text("Q", x + 166, queryY + 10, 13, TECH.coral, "center", 750);
    if (showOutput) {
      arrow(x + 183, queryY + 10, x + 222, queryY + 10, TECH.coral, 1.5);
      cube(x + 228, queryY - 18, 66, 55, 10, "head", "1×10", TECH.coral);
      arrow(x + 306, queryY + 10, x + 336, queryY + 10, TECH.gold, 1.5);
      ctx.fillStyle = "rgba(255,209,102,.22)";
      ctx.strokeStyle = TECH.gold;
      ctx.fillRect(x + 342, queryY - 17, 55, 23);
      ctx.fillRect(x + 342, queryY + 13, 34, 23);
      ctx.strokeRect(x + 342, queryY - 17, 55, 23);
      ctx.strokeRect(x + 342, queryY + 13, 34, 23);
      mono("2 classes", x + 370, queryY + 52, 11, TECH.gold, "center", 700);
    }
  }

  function background() {
    const gradient = ctx.createLinearGradient(0, 0, BASE_W, BASE_H);
    gradient.addColorStop(0, "#07111f");
    gradient.addColorStop(1, "#0a1628");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, BASE_W, BASE_H);
    ctx.strokeStyle = "rgba(88,127,164,.11)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= BASE_W; x += 32) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, BASE_H);
      ctx.stroke();
    }
    for (let y = 0; y <= BASE_H; y += 32) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(BASE_W, y);
      ctx.stroke();
    }
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.scale(width / BASE_W, height / BASE_H);
    background();
    render({ ctx, text, mono, arrow, panel, cube, title, table, columnAttention, rowCompression, datasetAttention });
    ctx.restore();
  }

  function resize() {
    width = Math.max(760, canvas.clientWidth || BASE_W);
    height = width * BASE_H / BASE_W;
    canvas.style.height = `${height}px`;
    ctx = fitCanvas(canvas, width, height);
    draw();
  }

  resize();
  const observer = "ResizeObserver" in window ? new ResizeObserver(resize) : null;
  if (observer) observer.observe(canvas);
  else window.addEventListener("resize", resize);
  return { draw, disconnect: () => observer?.disconnect() };
}

function createNanoDemoTask() {
  const X = Array.from({ length: 10 }, (_, row) =>
    Array.from({ length: 4 }, (_, col) =>
      +(Math.sin((row + 1) * (col + 2)) * 0.8).toFixed(2)));
  const y = X.map((row) => +(row[0] - 0.45 * row[1] + 0.25 * row[2] > 0));
  const queryScore = X[9][0] - 0.45 * X[9][1] + 0.25 * X[9][2];
  const logits = [-2 * queryScore, 2 * queryScore, -3.5, -3.7, -3.9, -4.1, -4.3, -4.5, -4.7, -4.9];
  const maxLogit = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maxLogit));
  const denominator = exponentials.reduce((sum, value) => sum + value, 0);
  const probabilities = exponentials.map((value) => value / denominator);
  const trueProbability = probabilities[y[9]];
  return { X, y, trueProbability, loss: -Math.log(trueProbability) };
}

const NANO_DEMO_TASK = createNanoDemoTask();

/* ============================ live training simulation ============================ */
function initTrainingLive() {
  const demo = $("#training-live");
  const canvas = $("#training-canvas");
  const steps = $$("[data-training-step]", demo || document);
  const detailLabel = $("#training-detail-label");
  const detailTitle = $("#training-detail-title");
  const detailCopy = $("#training-detail-copy");
  const detailExample = $("#training-detail-example");
  const cyclicDetails = $("#training-cyclic-details");
  const columnDetails = $("#training-column-details");
  const tokenLegend = $("#training-token-legend");
  if (!demo || !canvas || !steps.length) return;

  let active = 0;
  const { X, y, trueProbability, loss } = NANO_DEMO_TASK;
  let scene;
  scene = createTensorScene(canvas, (v) => {
    const { ctx, text, mono, arrow, panel, cube, title, table, columnAttention,
      rowCompression, datasetAttention } = v;
    if (active === 0) {
      title("Step 01 · prepare and embed", "Standardize from context, group features, then add context y", "fixed toy task · 9 context + 1 query");
      table(50, 132, X, y, { cw: 40, ch: 28 });
      mono("RAW INPUT", 130, 100, 12, TECH.cyan, "center", 700);
      arrow(270, 267, 326, 267, TECH.cyan, 2);
      mono("μ, σ from C1…C9", 299, 249, 9, TECH.muted, "center");
      cube(350, 171, 168, 188, 22, "grouped input", "10×4×3", TECH.cyan);
      mono("x₁ → [x′₁, x′₂, x′₄]", 445, 136, 13, TECH.cyan, "center", 700);
      mono("paper/current nano: 0, +1, +3 mod 4", 445, 390, 11, TECH.muted, "center");
      arrow(554, 267, 612, 267, TECH.violet, 2);
      cube(640, 139, 220, 252, 28, "cell embeddings", "10×4×128", TECH.violet);
      mono("+ y_embed_in for C1…C9 · no y for Q", 750, 425, 12, TECH.coral, "center", 700);
    } else if (active === 1) {
      title("Step 02 · forward pass", "3 column blocks → 3 row blocks → 12 dataset-ICL blocks", "shared θ");
      panel(20, 82, 262, 372, TECH.cyan);
      mono("1 · COLUMN ATTENTION", 38, 104, 12, TECH.cyan, "left", 700);
      mono("shared I bank: I ← context; all cells ← I", 38, 124, 9, TECH.muted);
      columnAttention(66, 177, 0.92);
      mono("10×4×128", 151, 446, 12, TECH.cyan, "center", 700);
      arrow(288, 267, 315, 267, TECH.cyan, 2);
      panel(322, 82, 307, 372, TECH.violet);
      mono("2 · ROW ATTENTION", 340, 104, 12, TECH.violet, "left", 700);
      mono("blocks 1–2: 8→8 · block 3: 4 CLS query 8", 340, 123, 9, TECH.muted);
      mono("LayerNorm + concat: 4 × 128D = 512D", 340, 140, 10, TECH.violet);
      rowCompression(365, 164, true);
      mono("10×8×128 → 10×512", 475, 446, 12, TECH.violet, "center", 700);
      arrow(635, 267, 662, 267, TECH.violet, 2);
      panel(669, 82, 271, 372, TECH.coral);
      mono("3 · DATASET ICL ATTENTION", 687, 104, 12, TECH.coral, "left", 700);
      mono("11 blocks: C,Q read C · final: Q reads C", 687, 124, 9, TECH.muted);
      datasetAttention(692, 146, y, false);
      arrow(875, 435, 898, 435, TECH.coral, 1.4);
      mono("LN→MLP→1×10", 934, 450, 9, TECH.gold, "right", 700);
    } else if (active === 2) {
      title("Step 03 · query loss", "The toy query contributes 10-way cross-entropy", "illustrative values · training only");
      panel(72, 115, 260, 240, TECH.coral);
      mono("HIDDEN TRUTH", 202, 144, 12, TECH.coral, "center", 700);
      text(`query y = ${y[9]}`, 202, 218, 30, TECH.text, "center", 800);
      mono("query label never enters forward pass", 202, 319, 11, TECH.muted, "center");
      arrow(350, 235, 405, 235, TECH.coral, 2);
      panel(426, 115, 250, 240, TECH.cyan);
      mono("ILLUSTRATIVE OUTPUT", 551, 144, 12, TECH.cyan, "center", 700);
      cube(492, 181, 118, 82, 15, "LN + MLP", "1×10 logits", TECH.cyan);
      mono(`softmax₁₀: P(y=${y[9]}) = ${trueProbability.toFixed(2)}`, 551, 319, 11, TECH.text, "center", 700);
      arrow(694, 235, 744, 235, TECH.violet, 2);
      panel(766, 115, 150, 240, TECH.violet);
      mono("CROSS-ENTROPY", 841, 144, 11, TECH.violet, "center", 700);
      text(loss.toFixed(2), 841, 218, 38, TECH.text, "center", 800);
      mono("∇θ", 841, 319, 18, TECH.violet, "center", 700);
      mono("labels visible: C1…C9 · label masked: Q", 480, 406, 13, TECH.muted, "center");
    } else {
      title("Step 04 · update weights", "Accumulate task losses, then update shared weights", "real recipe: graph-SCM batch · Muon");
      panel(81, 119, 276, 224, TECH.cyan);
      mono("SHARED θ · BEFORE", 219, 148, 12, TECH.cyan, "center", 700);
      for (let i = 0; i < 40; i++) {
        ctx.fillStyle = `rgba(53,217,255,${0.2 + (i % 5) * 0.12})`;
        ctx.fillRect(125 + (i % 8) * 25, 183 + Math.floor(i / 8) * 24, 14, 10);
      }
      arrow(382, 231, 558, 231, TECH.violet, 2.2);
      mono("this task adds ∇θL to the batch", 470, 201, 9, TECH.violet, "center", 700);
      mono("Muon updates after accumulation", 470, 218, 9, TECH.violet, "center", 700);
      panel(583, 119, 296, 224, TECH.violet);
      mono("SHARED θ · AFTER", 731, 148, 12, TECH.violet, "center", 700);
      for (let i = 0; i < 40; i++) {
        ctx.fillStyle = `rgba(154,124,255,${0.22 + ((i + 1) % 5) * 0.12})`;
        ctx.fillRect(633 + (i % 8) * 25, 183 + Math.floor(i / 8) * 24, 14, 10);
      }
      mono("sample discarded after contributing loss", 219, 380, 11, TECH.muted, "center", 700);
      mono("updated θ retained → next graph-SCM batch", 731, 380, 11, TECH.violet, "center", 700);
    }
  });

  const details = () => [
    ["step 1 · prepare and embed", "Standardize from context, group, project, then inject context labels", "Compute each feature’s mean and standard deviation from C1…C9 only. Under the paper/current nanoTabICL convention, collect offsets 0, +1, and +3 modulo the number of columns. Project each group to 128D, then add y_embed_in(yᵢ) to context cells only.", "10×4 → z-score from context → 10×4×3 → 10×4×128 + y_embed_in(C)"],
    ["step 2 · nanoTabICL forward", "Run 3 column, 3 row, and 12 dataset-ICL blocks", "Each column block uses a shared 128-vector inducing bank: inducing queries summarize context cells with QASSMax, then every cell reads those summaries. Row blocks use RoPE; the final block computes four CLS outputs and concatenates them to 512D. Context row tokens receive a second label embedding, y_embed_icl. Across the first 11 ICL blocks, context and query tokens read context K/V; the final block emits query representations only.", "Cᵢ = label-aware rowᵢ(512D) + y_embed_icl(yᵢ) · Q has no y · out: LayerNorm → MLP → 10 logits"],
    ["step 3 · supervised query loss", `Compare illustrative logits with hidden y = ${y[9]}`, "For this one-query toy, cross-entropy is the negative log probability assigned to the hidden query label after softmax over all ten output slots. The query features participate in attention; only its label is withheld. Real training sums or averages this objective over every held-out row in a batch of tasks.", `softmax₁₀(1×10 logits) · P(y=${y[9]})=${trueProbability.toFixed(2)} · CE=${loss.toFixed(2)}`],
    ["step 4 · shared update", "Accumulate task gradients, update θ with Muon, then sample new tasks", "In the released TabICLv2 recipe, many on-the-fly graph-SCM tasks contribute held-out-row cross-entropy to a batch gradient before a Muon optimizer step. Shared parameters persist; generated task samples are discarded. This fixed widget illustrates that process but performs no optimizer step or task resampling.", "θ ← Muon(θ, accumulated ∇θLbatch) · next batch samples new graph-SCM tasks"],
  ];

  function show(step) {
    active = step;
    steps.forEach((button, index) => {
      const selected = index === step;
      button.classList.toggle("is-active", selected);
      button.classList.toggle("is-complete", index < step);
      button.setAttribute("aria-pressed", String(selected));
    });
    centerOverflowingStep(steps[step]);
    const [label, heading, copy, example] = details()[step];
    detailLabel.textContent = label;
    detailTitle.textContent = heading;
    detailCopy.textContent = copy;
    detailExample.textContent = example;
    if (cyclicDetails) {
      cyclicDetails.hidden = step !== 0;
      if (step !== 0) cyclicDetails.open = false;
    }
    if (columnDetails) {
      columnDetails.hidden = step !== 1;
      if (step !== 1) columnDetails.open = false;
    }
    if (tokenLegend) tokenLegend.hidden = step !== 1;
    scene.draw();
  }

  steps.forEach((button, index) => button.addEventListener("click", () => show(index)));
  show(0);
  window.addEventListener("pagehide", () => {
    scene.disconnect();
  }, { once: true });
}

/* ============================ live inference architecture ============================ */
function initInferenceLive() {
  const demo = $("#inference-live");
  const canvas = $("#inference-canvas");
  const steps = $$("[data-inference-step]", demo || document);
  const detailLabel = $("#inference-detail-label");
  const detailTitle = $("#inference-detail-title");
  const detailCopy = $("#inference-detail-copy");
  const detailExample = $("#inference-detail-example");
  if (!demo || !canvas || !steps.length) return;

  let active = 0;
  const { X, y } = NANO_DEMO_TASK;
  let scene;
  scene = createTensorScene(canvas, (v) => {
    const { text, mono, arrow, panel, cube, title, table, columnAttention,
      rowCompression, datasetAttention } = v;
    if (active === 0) {
      title("Stage 01 · prepare + group", "Repeat training-time preprocessing with context statistics", "same toy table · weights frozen");
      table(48, 131, X, y, { cw: 39, ch: 28 });
      arrow(268, 269, 325, 269, TECH.cyan, 2);
      panel(348, 101, 238, 330, TECH.cyan);
      mono("POSITION j = x₁", 467, 131, 12, TECH.cyan, "center", 700);
      cube(391, 177, 122, 91, 18, "g₁", "1×3", TECH.cyan);
      mono("[x′₁, x′₂, x′₄]", 452, 302, 16, TECH.text, "center", 700);
      mono("0  +1  +3 modulo 4", 467, 337, 12, TECH.muted, "center");
      mono("repeat for 10×4 positions", 467, 393, 12, TECH.cyan, "center", 700);
      arrow(611, 269, 660, 269, TECH.violet, 2);
      cube(686, 146, 190, 232, 28, "cell embeddings", "10×4×128", TECH.violet);
      mono("10×4×3", 467, 419, 12, TECH.cyan, "center", 700);
      mono("+ y_embed_in for context only", 781, 407, 11, TECH.coral, "center", 700);
    } else if (active === 1) {
      title("Stage 02 · column attention", "Three induced blocks reuse context-derived summaries", "query never enters K/V");
      panel(56, 92, 574, 354, TECH.cyan);
      columnAttention(150, 155, 1.08);
      arrow(659, 268, 714, 268, TECH.violet, 2);
      cube(742, 163, 150, 204, 25, "column output", "10×4×128", TECH.violet);
    } else if (active === 2) {
      title("Stage 03 · row compression", "Three RoPE blocks process each row independently", "final block returns 4 CLS");
      panel(63, 92, 604, 354, TECH.violet);
      mono("4 CLS TOKENS", 270, 118, 11, TECH.violet, "center", 700);
      mono("FEATURES", 334, 118, 11, TECH.cyan, "center", 700);
      mono("ROW TOKEN", 427, 118, 11, TECH.violet, "center", 700);
      rowCompression(240, 145, true);
      arrow(696, 267, 748, 267, TECH.violet, 2);
      cube(776, 173, 116, 188, 23, "all rows", "10×512", TECH.violet);
    } else {
      title("Stage 04 · dataset ICL + output", "Twelve ICL blocks turn context into query logits", "no loss · no backward · no update");
      panel(49, 90, 474, 356, TECH.coral);
      mono("C += y_embed_icl · Q has no y · K/V = context only", 286, 116, 10, TECH.coral, "center", 700);
      datasetAttention(82, 139, y, false);
      arrow(550, 268, 603, 268, TECH.coral, 2);
      panel(628, 134, 268, 270, TECH.gold);
      mono("FROZEN OUTPUT HEAD", 762, 163, 12, TECH.gold, "center", 700);
      cube(691, 207, 122, 82, 17, "LN + MLP", "1×10 logits", TECH.coral);
      ctxFillBars(v.ctx, 838, 203);
      mono("2 active classes", 762, 355, 13, TECH.gold, "center", 700);
    }
  });

  function ctxFillBars(ctx, x, y0) {
    ctx.fillStyle = "rgba(255,209,102,.24)";
    ctx.strokeStyle = TECH.gold;
    ctx.fillRect(x, y0, 23, 48);
    ctx.fillRect(x + 31, y0 + 19, 23, 29);
    ctx.strokeRect(x, y0, 23, 48);
    ctx.strokeRect(x + 31, y0 + 19, 23, 29);
  }

  const details = [
    ["stage 1 · cells and cyclic grouping", "Standardize, group each feature position, then embed it", "Compute feature statistics from context rows. Under the paper/current nanoTabICL convention, position j gathers j, j+1, and j+3 modulo d; for d=4, x₁ maps to [x′₁, x′₂, x′₄]. A first y embedding is added to context cells only.", "10×4 + separate context y → z-score → 10×4×3 → 10×4×128"],
    ["stage 2 · column attention", "Attend vertically through context-derived inducing summaries", "In each of three blocks, one shared 128-vector bank is reused across feature columns. Inducing queries summarize context cells only; all cells, including the query, then read those summaries.", "10×4×128 → 10×4×128"],
    ["stage 3 · row compression", "Use three RoPE row blocks and retain four CLS outputs", "For every row, four learned CLS tokens precede four feature tokens. The first two blocks update all eight tokens; the last uses four CLS queries over all eight keys and values. LayerNorm and concatenation produce one 512D row token.", "10×8×128 → 10×4×128 → 10×512"],
    ["stage 4 · dataset ICL and output", "Use labeled context tokens to answer the query", "After a second context-only y embedding, the first eleven ICL blocks update context and query tokens from context K/V; the final block returns queries only. LayerNorm and a two-layer MLP emit ten logits, then inference keeps the active classes. There is no loss or weight update.", "10×512 → 12 ICL blocks → LayerNorm → MLP → 1×10 logits → 2 active classes"],
  ];

  function show(step) {
    active = step;
    steps.forEach((button, index) => {
      const selected = index === step;
      button.classList.toggle("is-active", selected);
      button.classList.toggle("is-complete", index < step);
      button.setAttribute("aria-pressed", String(selected));
    });
    centerOverflowingStep(steps[step]);
    const [label, heading, copy, example] = details[step];
    detailLabel.textContent = label;
    detailTitle.textContent = heading;
    detailCopy.textContent = copy;
    detailExample.textContent = example;
    scene.draw();
  }

  steps.forEach((button, index) => button.addEventListener("click", () => show(index)));
  show(0);
  window.addEventListener("pagehide", scene.disconnect, { once: true });
}

/* ============================ table stress test ============================ */
function initTableLab() {
  const lab = $("#table-lab");
  const head = $("#tablelab-head");
  const body = $("#tablelab-body");
  const type = $("#tablelab-type");
  const title = $("#tablelab-title");
  const copy = $("#tablelab-copy");
  const meter = $$("#tablelab-meter span");
  if (!lab || !head || !body) return;

  const columns = [
    {
      key: "productRelated",
      label: "ProductRelated",
      kind: "number",
      title: "Counts are discrete and strongly skewed",
      copy: "Most sessions view only a few product pages, while a small number view more than one hundred. The long tail makes scale important.",
      severity: 4,
    },
    {
      key: "bounceRates",
      label: "BounceRates",
      kind: "number",
      title: "Rates are bounded but concentrated near zero",
      copy: "Bounce rate lies between zero and one, yet most values cluster near zero. Its distribution differs sharply from page counts.",
      severity: 4,
    },
    {
      key: "pageValues",
      label: "PageValues",
      kind: "number",
      title: "Derived metrics can be zero-inflated",
      copy: "Page value is zero for many sessions and positive for a smaller group. A few large values carry much of the signal.",
      severity: 5,
    },
    {
      key: "visitorType",
      label: "VisitorType",
      kind: "category",
      title: "Categories have no numeric distance",
      copy: "Returning and new visitor labels are states, not measurements. They require categorical treatment rather than arbitrary numeric codes.",
      severity: 3,
    },
    {
      key: "revenue",
      label: "Revenue",
      kind: "target",
      title: "The target is imbalanced",
      copy: "Only 15.5% of the full dataset's sessions end in a purchase. Accuracy alone can therefore hide a model that ignores the minority class.",
      severity: 5,
    },
  ];

  // Authentic records from the UCI Online Shoppers Purchasing Intention dataset.
  // The compact pool keeps the original values and includes both target classes.
  const sourceRows = [
    { id: "shopper-01", productRelated: 1, bounceRates: 0.2, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-02", productRelated: 1, bounceRates: 0, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-03", productRelated: 5, bounceRates: 0, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-04", productRelated: 7, bounceRates: 0, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-05", productRelated: 37, bounceRates: 0.006766917, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-06", productRelated: 9, bounceRates: 0, pageValues: 0, visitorType: "New_Visitor", revenue: false },
    { id: "shopper-07", productRelated: 8, bounceRates: 0, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-08", productRelated: 9, bounceRates: 0.022222222, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-09", productRelated: 35, bounceRates: 0, pageValues: 0, visitorType: "New_Visitor", revenue: false },
    { id: "shopper-10", productRelated: 4, bounceRates: 0, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-11", productRelated: 8, bounceRates: 0, pageValues: 0, visitorType: "New_Visitor", revenue: false },
    { id: "shopper-12", productRelated: 43, bounceRates: 0, pageValues: 0, visitorType: "New_Visitor", revenue: false },
    { id: "shopper-13", productRelated: 30, bounceRates: 0.043851852, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-14", productRelated: 141, bounceRates: 0.006103286, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-15", productRelated: 11, bounceRates: 0.030769231, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-16", productRelated: 40, bounceRates: 0.004878049, pageValues: 0, visitorType: "Returning_Visitor", revenue: false },
    { id: "shopper-17", productRelated: 27, bounceRates: 0, pageValues: 22.9160357, visitorType: "Returning_Visitor", revenue: true },
    { id: "shopper-18", productRelated: 14, bounceRates: 0, pageValues: 22.88908649, visitorType: "New_Visitor", revenue: true },
    { id: "shopper-19", productRelated: 12, bounceRates: 0, pageValues: 49.62545455, visitorType: "New_Visitor", revenue: true },
    { id: "shopper-20", productRelated: 23, bounceRates: 0, pageValues: 56.35884615, visitorType: "Returning_Visitor", revenue: true },
  ];

  const sampleSize = 5;
  const markQuery = (rows) => rows.map((row, index) => ({
    ...row,
    query: index === rows.length - 1,
  }));
  const initialRows = () => markQuery([
    sourceRows[0],
    sourceRows[4],
    sourceRows[8],
    sourceRows[12],
    sourceRows[13],
  ]);
  const randomSample = () => {
    const currentKey = state.rows.map((row) => row.id).sort().join(":");
    let sampled;
    for (let attempt = 0; attempt < 8; attempt++) {
      const pool = [...sourceRows];
      for (let index = pool.length - 1; index > 0; index--) {
        const swap = Math.floor(Math.random() * (index + 1));
        [pool[index], pool[swap]] = [pool[swap], pool[index]];
      }
      sampled = pool.slice(0, sampleSize);
      if (sampled.map((row) => row.id).sort().join(":") !== currentKey) break;
    }
    return markQuery(sampled);
  };

  const state = {
    rows: initialRows(),
    columns: [...columns],
    missing: new Set(),
    small: false,
    selected: null,
  };

  const format = (key, value) => {
    if (key === "bounceRates" && typeof value === "number") {
      return `${(value * 100).toFixed(2)}%`;
    }
    if (key === "pageValues" && typeof value === "number") {
      return value.toFixed(2);
    }
    return String(value);
  };

  function explain(label, heading, paragraph, severity) {
    type.textContent = label;
    title.textContent = heading;
    copy.textContent = paragraph;
    meter.forEach((bar, index) => bar.classList.toggle("is-on", index < severity));
  }

  function render() {
    head.replaceChildren();
    body.replaceChildren();

    const headerRow = document.createElement("tr");
    state.columns.forEach((column) => {
      const th = document.createElement("th");
      th.dataset.kind = column.kind;
      th.classList.toggle("is-selected", state.selected === column.key);
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = column.label;
      button.setAttribute(
        "aria-pressed",
        String(state.selected === column.key)
      );
      button.addEventListener("click", () => {
        state.selected = column.key;
        explain(column.kind, column.title, column.copy, column.severity);
        render();
      });
      th.append(button);
      headerRow.append(th);
    });
    head.append(headerRow);

    const queryRow = state.rows.find((row) => row.query);
    const contextRow = state.rows.find((row) => !row.query);
    const visibleRows = state.small
      ? [contextRow, queryRow].filter(Boolean)
      : state.rows;
    visibleRows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.className = "is-new";
      state.columns.forEach((column) => {
        const td = document.createElement("td");
        const missing = state.missing.has(`${row.id}:${column.key}`);
        const hiddenTarget = row.query && column.key === "revenue";
        const value = row[column.key];
        td.textContent = hiddenTarget
          ? "?"
          : missing
            ? "NA"
            : format(column.key, value);
        td.classList.toggle("is-selected", state.selected === column.key);
        td.classList.toggle("is-missing", missing);
        if (hiddenTarget) {
          td.setAttribute("aria-label", "Revenue target hidden");
        }
        tr.append(td);
      });
      tr.classList.toggle("is-query", row.query);
      if (row.query) tr.setAttribute("aria-label", "Query row with hidden Revenue target");
      body.append(tr);
    });
  }

  function clearActions() {
    $$("[data-table-action]", lab).forEach((button) => {
      button.classList.remove("is-active");
      if (button.dataset.tableAction !== "reset") {
        button.setAttribute("aria-pressed", "false");
      }
    });
  }

  $$("[data-table-action]", lab).forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.tableAction;
      if (action === "reset") {
        state.rows = initialRows();
        state.columns = [...columns];
        state.missing.clear();
        state.small = false;
        state.selected = null;
        clearActions();
        explain(
          "mixed schema",
          "A grid does not imply one data type",
          "Page counts, bounded rates, derived values, visitor categories, and a binary target share one grid, but each carries a different kind of meaning.",
          3
        );
      } else if (action === "sample-rows") {
        state.rows = randomSample();
        state.missing.clear();
        state.small = false;
        ["missing", "include-purchase", "small"].forEach((name) => {
          const related = $(`[data-table-action="${name}"]`, lab);
          if (related) {
            related.classList.remove("is-active");
            related.setAttribute("aria-pressed", "false");
          }
        });
        explain(
          "row sampling",
          "A fresh context changes the evidence",
          "Five authentic UCI sessions were sampled without replacement. The final row is always the query, and its Revenue label remains hidden.",
          4
        );
      } else if (action === "shuffle-columns") {
        state.columns.sort(() => Math.random() - 0.5);
        button.classList.add("is-active");
        explain(
          "schema",
          "Column order changed; their meaning did not",
          "Moving a column changes its position but not its semantic role. A tabular model must track feature identity without treating the table like a sentence.",
          5
        );
      } else if (action === "missing") {
        const contextRows = state.rows.filter((row) => !row.query);
        state.missing = new Set([
          `${contextRows[0].id}:bounceRates`,
          `${contextRows[1].id}:visitorType`,
          `${contextRows[2].id}:pageValues`,
        ]);
        button.classList.add("is-active");
        explain(
          "missingness",
          "Masked values remove part of the context",
          "The underlying records stay unchanged, but three cells are hidden to show how incomplete evidence affects a tabular task.",
          5
        );
      } else if (action === "include-purchase") {
        const purchaseRow = sourceRows.find(
          (row) => row.revenue && !state.rows.some((shown) => shown.id === row.id)
        ) || sourceRows.find((row) => row.revenue);
        state.rows = markQuery([purchaseRow, ...state.rows.slice(1)]);
        state.missing.clear();
        state.small = false;
        button.classList.add("is-active");
        explain(
          "class imbalance",
          "Purchases are the minority class",
          "The inserted positive session is an authentic UCI record. In the full dataset, only 15.5% of sessions end in a purchase.",
          5
        );
      } else if (action === "small") {
        state.small = true;
        button.classList.add("is-active");
        explain(
          "small data",
          "With two rows, every example matters",
          "Small tables provide weak evidence and make validation noisy. This is where a useful pretrained prior can matter most.",
          5
        );
      }
      if (action !== "reset" && action !== "sample-rows") {
        button.setAttribute("aria-pressed", "true");
      }
      render();
    });
  });

  clearActions();
  explain(
    "mixed schema",
    "A grid does not imply one data type",
    "Page counts, bounded rates, derived values, visitor categories, and a binary target share one grid, but each carries a different kind of meaning.",
    3
  );
  render();
}

/* ============================ boosting walkthrough ============================ */
function initBoostingLab() {
  const lab = $("#boosting-lab");
  if (!lab) return;

  const steps = [
    {
      badge: "initial prediction",
      title: "Start from a constant log-odds score",
      copy: "Before fitting a tree, the model predicts the same value for every row. Here the initial score is a log-odds constant corresponding to a 15.5% purchase rate.",
      score: -1.70,
      probability: 0.155,
      equation: "σ(−1.70) = 0.155",
      treeCaption: "No tree fitted yet",
      treeDescription: "The initial ensemble contains only a constant prediction.",
      residualTitle: "Pseudo-residuals before tree 1",
      residualMean: 0.500,
      residuals: [
        ["A · y0", -0.155], ["B · y1", 0.845], ["C · y0", -0.155], ["D · y1", 0.845],
      ],
      residualNote: "For logistic loss, a teaching-friendly pseudo-residual is label minus current probability. Positive rows ask for a higher score; negative rows ask for a lower one.",
      path: "No split yet. The query receives the same initial score as every other session.",
      tree: null,
    },
    {
      badge: "round 1 added",
      title: "Fit a stump to the current errors",
      copy: "Tree 1 is fitted to pseudo-residuals from labeled sessions. Its PageValues split groups rows whose current predictions need different score corrections.",
      score: -2.00,
      probability: 0.119,
      equation: "−1.70 + 0.30(−1.00) = −2.00  →  σ = 0.119",
      treeCaption: "Tree 1 fits residual structure",
      treeDescription: "A stump splits on PageValues greater than 10. The query follows the no branch to a raw leaf value of minus 1.",
      residualTitle: "Errors after tree 1",
      residualMean: 0.430,
      residuals: [
        ["A · y0", -0.119], ["B · y1", 0.741], ["C · y0", -0.119], ["D · y1", 0.741],
      ],
      residualNote: "The displayed mean absolute pseudo-residual falls from 0.500 to 0.430. Individual errors need not all shrink on every round.",
      path: "PageValues = 0.00 follows the no branch. Raw leaf −1.00 × learning rate 0.30 adds −0.30 to the ensemble score.",
      tree: {
        split: "PageValues > 10?",
        leftBranch: "no", rightBranch: "yes",
        leftValue: "leaf −1.00", rightValue: "leaf +1.95",
        path: "left",
      },
    },
    {
      badge: "round 2 added",
      title: "Add another scaled correction",
      copy: "Tree 2 sees the errors left after round 1, not the original labels in isolation. This stump captures a remaining pattern associated with VisitorType.",
      score: -1.76,
      probability: 0.147,
      equation: "−2.00 + 0.30(+0.80) = −1.76  →  σ = 0.147",
      treeCaption: "Tree 2 fits remaining errors",
      treeDescription: "A stump tests whether VisitorType is New Visitor. The query follows the yes branch to a raw leaf value of plus 0.8.",
      residualTitle: "Errors after tree 2",
      residualMean: 0.419,
      residuals: [
        ["A · y0", -0.105], ["B · y1", 0.771], ["C · y0", -0.105], ["D · y1", 0.694],
      ],
      residualNote: "The second tree makes a smaller objective-improving adjustment. The learning rate tempers every raw leaf value.",
      path: "VisitorType = New_Visitor follows the yes branch. Raw leaf +0.80 × learning rate 0.30 adds +0.24 to the running score.",
      tree: {
        split: "VisitorType = New?",
        leftBranch: "no", rightBranch: "yes",
        leftValue: "leaf −0.30", rightValue: "leaf +0.80",
        path: "right",
      },
    },
    {
      badge: "round 3 added",
      title: "Repeat, then sum all tree outputs",
      copy: "Tree 3 fits the residual pattern that remains after two rounds. Prediction uses the initial score plus all three scaled leaf values.",
      score: -1.37,
      probability: 0.203,
      equation: "−1.76 + 0.30(+1.30) = −1.37  →  σ = 0.203",
      treeCaption: "Tree 3 refines the ensemble",
      treeDescription: "A stump splits on ProductRelated greater than 20. The query follows the yes branch to a raw leaf value of plus 1.3.",
      residualTitle: "Errors after tree 3",
      residualMean: 0.409,
      residuals: [
        ["A · y0", -0.080], ["B · y1", 0.690], ["C · y0", -0.170], ["D · y1", 0.694],
      ],
      residualNote: "After three illustrative rounds, the displayed mean absolute pseudo-residual is 0.409. A real fit would continue while validation performance warrants it.",
      path: "ProductRelated = 35 follows the yes branch. Raw leaf +1.30 × learning rate 0.30 adds +0.39, producing a 20.3% accumulated probability.",
      tree: {
        split: "ProductRelated > 20?",
        leftBranch: "no", rightBranch: "yes",
        leftValue: "leaf −0.25", rightValue: "leaf +1.30",
        path: "right",
      },
    },
  ];

  const badge = $("#boost-round-badge");
  const stepLabel = $("#boost-step-label");
  const stepTitle = $("#boost-step-title");
  const stepCopy = $("#boost-step-copy");
  const probability = $("#boost-probability");
  const probabilityBar = $("#boost-probability-bar");
  const equation = $("#boost-equation");
  const treeCaption = $("#boost-tree-caption");
  const treeSvg = $("#boost-tree-svg");
  const residualTitle = $("#boost-residual-title");
  const residualMean = $("#boost-residual-mean");
  const residualRows = $("#boost-residual-rows");
  const residualNote = $("#boost-residual-note");
  const pathCopy = $("#boost-path-copy");
  const status = $("#boost-status");
  const prev = $("#boost-prev");
  const next = $("#boost-next");
  const reset = $("#boost-reset");
  const progress = $$(".booststeps li", lab);
  let current = 0;

  function treeMarkup(step) {
    if (!step.tree) {
      return `
        <title id="boost-tree-title">Current shallow decision tree</title>
        <desc id="boost-tree-desc">${step.treeDescription}</desc>
        <rect class="tree-node is-path" x="150" y="80" width="220" height="70" rx="3"></rect>
        <text class="tree-text" x="260" y="109">constant score</text>
        <text class="tree-subtext" x="260" y="132">log-odds −1.70</text>
        <text class="tree-empty" x="260" y="202">trees will fit the remaining errors</text>
      `;
    }
    const t = step.tree;
    const leftPath = t.path === "left" ? " is-path" : "";
    const rightPath = t.path === "right" ? " is-path" : "";
    return `
      <title id="boost-tree-title">Current shallow decision tree</title>
      <desc id="boost-tree-desc">${step.treeDescription}</desc>
      <path class="tree-edge${leftPath}" d="M260 87 L140 165"></path>
      <path class="tree-edge${rightPath}" d="M260 87 L380 165"></path>
      <text class="tree-branch" x="184" y="131">${t.leftBranch}</text>
      <text class="tree-branch" x="336" y="131">${t.rightBranch}</text>
      <rect class="tree-node is-path" x="145" y="27" width="230" height="60" rx="3"></rect>
      <text class="tree-text" x="260" y="63">${t.split}</text>
      <rect class="tree-leaf${leftPath}" x="52" y="165" width="176" height="58" rx="3"></rect>
      <text class="tree-text" x="140" y="190">${t.leftValue}</text>
      <text class="tree-subtext" x="140" y="209">raw correction</text>
      <rect class="tree-leaf${rightPath}" x="292" y="165" width="176" height="58" rx="3"></rect>
      <text class="tree-text" x="380" y="190">${t.rightValue}</text>
      <text class="tree-subtext" x="380" y="209">raw correction</text>
    `;
  }

  function renderResiduals(values) {
    residualRows.replaceChildren();
    values.forEach(([id, value]) => {
      const row = document.createElement("div");
      row.className = "boostresidual__row";
      const label = document.createElement("span");
      label.className = "boostresidual__id";
      label.textContent = id;
      const line = document.createElement("span");
      line.className = "boostresidual__line";
      const bar = document.createElement("span");
      bar.className = `boostresidual__bar${value < 0 ? " is-negative" : ""}`;
      const magnitude = Math.min(Math.abs(value), 1) * 50;
      bar.style.left = value < 0 ? `${50 - magnitude}%` : "50%";
      bar.style.width = `${magnitude}%`;
      line.append(bar);
      const number = document.createElement("span");
      number.className = "boostresidual__value";
      number.textContent = `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
      row.append(label, line, number);
      residualRows.append(row);
    });
  }

  function render(announce = true) {
    const step = steps[current];
    badge.textContent = step.badge;
    stepLabel.textContent = `step ${current} of 3`;
    stepTitle.textContent = step.title;
    stepCopy.textContent = step.copy;
    probability.textContent = `${(step.probability * 100).toFixed(1)}%`;
    probabilityBar.style.width = `${step.probability * 100}%`;
    equation.textContent = step.equation;
    treeCaption.textContent = step.treeCaption;
    treeSvg.innerHTML = treeMarkup(step);
    residualTitle.textContent = step.residualTitle;
    residualMean.textContent = `mean |error| ${step.residualMean.toFixed(3)}`;
    residualNote.textContent = step.residualNote;
    pathCopy.textContent = step.path;
    renderResiduals(step.residuals);

    progress.forEach((item, index) => {
      item.classList.toggle("is-active", index === current);
      item.classList.toggle("is-done", index < current);
    });
    prev.disabled = current === 0;
    next.disabled = current === steps.length - 1;
    next.textContent = current === steps.length - 1 ? "All rounds added" : "Next boosting round";
    if (announce) {
      status.textContent = `${step.badge}. Query purchase probability ${(step.probability * 100).toFixed(1)} percent.`;
    }
  }

  prev.addEventListener("click", () => {
    current = Math.max(0, current - 1);
    render();
  });
  next.addEventListener("click", () => {
    current = Math.min(steps.length - 1, current + 1);
    render();
  });
  reset.addEventListener("click", () => {
    current = 0;
    render();
  });
  render(false);
}

/* ============================ boot ============================ */
function boot() {
  initReveals();
  initOutlineNav();
  initStickyTitle();
  initHero();
  initPfnOverview();
  initPfnPlayground();
  initScm();
  initTrainingLive();
  initInferenceLive();
  initTableLab();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
