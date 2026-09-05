/*
 * TabICL v2 as one exploded tensor sculpture.
 *
 * The aligned glass planes correspond to the implementation:
 * input table, feature grouping, column TF x3, row TF x3 with four CLS
 * tokens, 512D row tokens, ICL TF x12, and prediction.
 *
 * This is decorative only. No inference runs in the landing page.
 */
let THREE;
try {
  THREE = await import("three");
} catch (error) {
  console.warn("[hero3d] Three.js unavailable; using 2D fallback", error);
}

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const COLOR = {
  cyan: 0x42c3e7,
  violet: 0x8b7ff2,
  coral: 0xf48966,
  white: 0xeaf1fb,
  slate: 0x7e899c,
  dark: 0x10151f,
};

function supportsWebGL() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(
      window.WebGLRenderingContext &&
      (canvas.getContext("webgl") || canvas.getContext("experimental-webgl"))
    );
  } catch {
    return false;
  }
}

function init() {
  const hero = document.querySelector(".hero");
  const visual = document.querySelector("#hero-visual");
  if (!hero || !visual || !THREE || !supportsWebGL()) return;

  const canvas = document.createElement("canvas");
  canvas.className = "hero__canvas";
  canvas.id = "hero-3d";
  canvas.setAttribute("aria-hidden", "true");
  visual.appendChild(canvas);

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
  } catch (error) {
    console.warn("[hero3d] WebGL renderer unavailable", error);
    canvas.remove();
    return;
  }

  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;

  if (typeof window.__heroStop2D === "function") window.__heroStop2D();

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
  camera.position.set(0, 0.2, 23);
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.HemisphereLight(0xf4f7ff, 0x0d1119, 2.8));
  const key = new THREE.DirectionalLight(0xffffff, 4);
  key.position.set(-5, 8, 12);
  scene.add(key);
  const cyanLight = new THREE.RectAreaLight(COLOR.cyan, 6, 8, 10);
  cyanLight.position.set(-4, -2, 8);
  cyanLight.lookAt(3, 0, 0);
  scene.add(cyanLight);
  const coralLight = new THREE.RectAreaLight(COLOR.coral, 5, 8, 10);
  coralLight.position.set(11, 4, 7);
  coralLight.lookAt(4, 0, 0);
  scene.add(coralLight);

  const sculpture = new THREE.Group();
  sculpture.position.set(0, 0, 0);
  sculpture.scale.setScalar(1.12);
  sculpture.rotation.y = -Math.PI / 4;
  sculpture.rotation.x = -0.08;
  scene.add(sculpture);

  const glass = (color, opacity = 0.28) =>
    new THREE.MeshPhysicalMaterial({
      color,
      roughness: 0.12,
      metalness: 0.02,
      transmission: 0.72,
      transparent: true,
      opacity,
      thickness: 0.65,
      ior: 1.42,
      clearcoat: 1,
      clearcoatRoughness: 0.08,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

  const solid = (color, opacity = 1) =>
    new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.16,
      roughness: 0.32,
      metalness: 0.03,
      transparent: opacity < 1,
      opacity,
      depthWrite: opacity === 1,
    });

  const lineMaterial = (color, opacity = 0.5) =>
    new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthWrite: false,
    });

  function addEdges(mesh, color, opacity = 0.48) {
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      lineMaterial(color, opacity)
    );
    edges.position.copy(mesh.position);
    edges.rotation.copy(mesh.rotation);
    edges.scale.copy(mesh.scale);
    sculpture.add(edges);
    return edges;
  }

  function panel(z, color, opacity = 0.24, width = 6.9, height = 6.25) {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(width, height, 0.075),
      glass(color, opacity)
    );
    mesh.position.z = z;
    sculpture.add(mesh);
    addEdges(mesh, color, 0.43);
    return mesh;
  }

  function cell(x, y, z, color, width = 0.72, height = 0.62) {
    const shell = new THREE.Mesh(
      new THREE.BoxGeometry(width, height, 0.1),
      glass(color, 0.3)
    );
    shell.position.set(x, y, z);
    sculpture.add(shell);
    addEdges(shell, color, 0.35);
    return shell;
  }

  function box(x, y, z, width, height, depth, color, opacity = 1) {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(width, height, depth),
      solid(color, opacity)
    );
    mesh.position.set(x, y, z);
    sculpture.add(mesh);
    return mesh;
  }

  function node(x, y, z, color, radius = 0.11) {
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 18, 18),
      solid(color)
    );
    mesh.position.set(x, y, z);
    sculpture.add(mesh);
    return mesh;
  }

  const flowingEdges = [];
  function flowEdge(curve, color, radius = 0.018, opacity = 0.48) {
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 40, radius, 7, false),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.42,
        roughness: 0.28,
        transparent: true,
        opacity,
        depthWrite: false,
      })
    );
    sculpture.add(tube);

    const dashGeometry = new THREE.BufferGeometry().setFromPoints(
      curve.getPoints(48)
    );
    const dashMaterial = new THREE.LineDashedMaterial({
      color: COLOR.white,
      transparent: true,
      opacity: 0.5,
      dashSize: 0.13,
      gapSize: 0.12,
      depthWrite: false,
    });
    const dash = new THREE.Line(dashGeometry, dashMaterial);
    dash.computeLineDistances();
    sculpture.add(dash);
    flowingEdges.push({ tube, dash, curve });
    return { tube, dash };
  }

  function stageLabel(title, detail, x, y, z, color, width = 3.2) {
    const surface = document.createElement("canvas");
    surface.width = 640;
    surface.height = 128;
    const ctx = surface.getContext("2d");
    ctx.fillStyle = "rgba(8,11,17,.82)";
    ctx.beginPath();
    ctx.roundRect(3, 3, 634, 122, 16);
    ctx.fill();
    ctx.strokeStyle = `#${new THREE.Color(color).getHexString()}`;
    ctx.globalAlpha = 0.68;
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = `#${new THREE.Color(color).getHexString()}`;
    ctx.fillRect(22, 22, 6, 49);
    ctx.fillStyle = "#f2f5fa";
    ctx.font = "650 26px Inter, sans-serif";
    ctx.fillText(title, 45, 51);
    ctx.fillStyle = "#a5afbf";
    ctx.font = "500 18px 'JetBrains Mono', monospace";
    ctx.fillText(detail, 45, 88);

    const texture = new THREE.CanvasTexture(surface);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthTest: false,
        depthWrite: false,
      })
    );
    sprite.position.set(x, y, z);
    sprite.scale.set(width, width * (128 / 640), 1);
    sculpture.add(sprite);
    return sprite;
  }

  const rows = 6;
  const cols = 5;
  const xAt = (column) => (column - (cols - 1) / 2) * 1.02;
  const yAt = (row) => ((rows - 1) / 2 - row) * 0.82;

  // -----------------------------------------------------------------------
  // Front plane: raw table. Cyan feature columns, coral target column.
  // -----------------------------------------------------------------------
  const zInput = 3.85;
  const inputPanel = panel(zInput, COLOR.cyan, 0.2);
  const inputCells = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < cols; column++) {
      const target = column === cols - 1;
      const queryLabel = target && row >= rows - 2;
      const color = target ? COLOR.coral : COLOR.cyan;
      const shell = cell(xAt(column), yAt(row), zInput + 0.08, color);
      inputCells.push(shell);
      if (!queryLabel) {
        const value =
          0.1 + 0.3 * (0.5 + 0.5 * Math.sin(row * 1.9 + column * 1.3));
        box(
          xAt(column),
          yAt(row),
          zInput + 0.16,
          0.43,
          value,
          0.07,
          color,
          0.82
        );
      } else {
        node(xAt(column), yAt(row), zInput + 0.18, COLOR.white, 0.075);
      }
    }
  }
  stageLabel("INPUT TABLE", "context rows + query rows", -3.7, 3.28, zInput + 0.2, COLOR.cyan);

  // -----------------------------------------------------------------------
  // Feature grouping: paper/current nanoTabICL offsets 0, 1, and 3.
  // Only the query row is traced to keep the sculpture legible.
  // -----------------------------------------------------------------------
  const groupingZ = [3.05, 2.67, 2.29];
  const groupingPanels = groupingZ.map((z) => panel(z, COLOR.cyan, 0.18));
  const offsets = [0, 1, 3];
  offsets.forEach((offset, layer) => {
    const z = groupingZ[layer] + 0.07;
    for (let column = 0; column < cols; column++) {
      const source = xAt(column);
      const destination = xAt((column + offset) % cols);
      const y = yAt(rows - 1);
      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(source, y, zInput - 0.03),
        new THREE.Vector3((source + destination) / 2, y + 0.15, z + 0.3),
        new THREE.Vector3(destination, y, z)
      );
      flowEdge(curve, COLOR.cyan, 0.012, 0.3);
    }
  });
  stageLabel("FEATURE GROUPING", "context z-score · offsets 0, 1, 3 · 3 → 128 · +y", 3.65, 2.95, 2.65, COLOR.cyan, 3.65);

  // -----------------------------------------------------------------------
  // Column TF x3: aligned planes with vertical induced-attention structure.
  // -----------------------------------------------------------------------
  const columnZ = [1.63, 1.22, 0.81];
  const columnPanels = columnZ.map((z) => panel(z, COLOR.cyan, 0.2));
  columnZ.forEach((z, layer) => {
    for (let column = 0; column < cols; column++) {
      const x = xAt(column);
      const inducingY = 0;
      const inducing = node(x, inducingY, z + 0.13, COLOR.white, 0.055);
      if (layer === columnZ.length - 1) {
        for (let row = 0; row < rows; row++) {
          const start = new THREE.Vector3(x, yAt(row), z + 0.11);
          const end = inducing.position.clone();
          const curve = new THREE.QuadraticBezierCurve3(
            start,
            new THREE.Vector3(x + 0.08, (start.y + end.y) / 2, z + 0.34),
            end
          );
          flowEdge(curve, COLOR.cyan, 0.009, 0.25);
        }
      }
    }
  });
  stageLabel("COLUMN ENCODER ×3", "induced attention over rows", -3.72, 2.78, 1.2, COLOR.cyan, 3.7);

  // -----------------------------------------------------------------------
  // Row TF x3: horizontal attention structure with four visible CLS tokens.
  // -----------------------------------------------------------------------
  const rowZ = [0.05, -0.37, -0.79];
  const rowPanels = rowZ.map((z) => panel(z, COLOR.violet, 0.2));
  rowZ.forEach((z, layer) => {
    for (let row = 0; row < rows; row++) {
      const y = yAt(row);
      box(0, y, z + 0.11, 4.7, 0.075, 0.055, COLOR.violet, 0.48);
      if (layer === rowZ.length - 1) {
        for (let cls = 0; cls < 4; cls++) {
          node(-2.75 + cls * 0.22, y, z + 0.18, COLOR.coral, 0.055);
        }
      }
    }
  });
  stageLabel("ROW AGGREGATOR ×3", "+4 CLS · RoPE · merge → 512D", 3.65, 2.68, -0.38, COLOR.violet, 3.45);

  // -----------------------------------------------------------------------
  // Row-token plane: the table has collapsed to one 512D token per row.
  // -----------------------------------------------------------------------
  const zRowTokens = -1.48;
  const rowTokenPanel = panel(zRowTokens, COLOR.violet, 0.18, 3.35, 6.25);
  const rowTokens = [];
  for (let row = 0; row < rows; row++) {
    const token = box(
      0,
      yAt(row),
      zRowTokens + 0.13,
      1.95,
      0.25,
      0.12,
      row >= rows - 2 ? COLOR.coral : COLOR.violet,
      0.78
    );
    rowTokens.push(token);
  }

  // -----------------------------------------------------------------------
  // ICL TF x12: a deep aligned stack. Front layer shows context-to-query edges.
  // -----------------------------------------------------------------------
  const iclZ = Array.from({ length: 8 }, (_, index) => -2.15 - index * 0.31);
  const iclPanels = iclZ.map((z) => panel(z, COLOR.coral, 0.14, 3.9, 6.25));
  const zAttention = iclZ[0] + 0.13;
  const contextNodes = [];
  for (let row = 0; row < rows - 2; row++) {
    contextNodes.push(node(-0.72, yAt(row), zAttention, COLOR.violet, 0.1));
  }
  const queryNodes = [
    node(0.82, yAt(rows - 2), zAttention, COLOR.coral, 0.15),
    node(0.82, yAt(rows - 1), zAttention, COLOR.coral, 0.15),
  ];

  const attentionEdges = [];
  queryNodes.forEach((query, queryIndex) => {
    contextNodes.forEach((context, contextIndex) => {
      const start = context.position.clone();
      const end = query.position.clone();
      const curve = new THREE.QuadraticBezierCurve3(
        start,
        new THREE.Vector3(
          0,
          (start.y + end.y) / 2 + (contextIndex - 1.5) * 0.08,
          zAttention + 0.45 + queryIndex * 0.1
        ),
        end
      );
      attentionEdges.push(flowEdge(curve, COLOR.coral, 0.016, 0.52));
    });
  });
  stageLabel("IN-CONTEXT TF ×12", "+y · context KV → query Q", -3.72, 2.58, -3.18, COLOR.coral, 3.85);

  // -----------------------------------------------------------------------
  // Back plane: prediction head and class probabilities.
  // -----------------------------------------------------------------------
  const zOutput = -5.05;
  const outputPanel = panel(zOutput, COLOR.coral, 0.22, 3.35, 4.2);
  box(0, -1.45, zOutput + 0.14, 2.2, 0.1, 0.08, COLOR.white, 0.54);
  const probabilityBars = [
    box(-0.48, 0, zOutput + 0.16, 0.55, 1.2, 0.16, COLOR.cyan, 0.84),
    box(0.48, 0.25, zOutput + 0.16, 0.55, 1.7, 0.16, COLOR.coral, 0.84),
  ];
  stageLabel("PREDICTION HEAD", "LayerNorm + MLP → class logits", 3.2, -2.15, zOutput + 0.2, COLOR.coral, 3.35);

  const stagePlanes = [
    [inputPanel, ...groupingPanels],
    columnPanels,
    [...rowPanels, rowTokenPanel],
    iclPanels,
    [outputPanel],
  ];

  const pointer = { x: 0, y: 0 };
  const dragSurface = document.querySelector("#hero-drag-zone") || visual;
  const drag = {
    active: false,
    yaw: 0,
    pitch: 0,
    startX: 0,
    startY: 0,
    startYaw: 0,
    startPitch: 0,
  };
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  dragSurface.addEventListener("pointerdown", (event) => {
    drag.active = true;
    drag.startX = event.clientX;
    drag.startY = event.clientY;
    drag.startYaw = drag.yaw;
    drag.startPitch = drag.pitch;
    hero.classList.add("is-dragging");
    dragSurface.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  dragSurface.addEventListener("pointermove", (event) => {
    const bounds = visual.getBoundingClientRect();
    pointer.x = ((event.clientX - bounds.left) / bounds.width - 0.5) * 2;
    pointer.y = ((event.clientY - bounds.top) / bounds.height - 0.5) * 2;

    if (drag.active) {
      drag.yaw = clamp(
        drag.startYaw + (event.clientX - drag.startX) * 0.006,
        -1.15,
        1.15
      );
      drag.pitch = clamp(
        drag.startPitch + (event.clientY - drag.startY) * 0.004,
        -0.42,
        0.42
      );
    }
  });

  const endDrag = (event) => {
    if (!drag.active) return;
    drag.active = false;
    hero.classList.remove("is-dragging");
    if (dragSurface.hasPointerCapture(event.pointerId)) {
      dragSurface.releasePointerCapture(event.pointerId);
    }
  };
  dragSurface.addEventListener("pointerup", endDrag);
  dragSurface.addEventListener("pointercancel", endDrag);
  dragSurface.addEventListener("pointerleave", () => {
    if (!drag.active) {
      pointer.x = 0;
      pointer.y = 0;
    }
  });
  dragSurface.addEventListener("dblclick", () => {
    drag.yaw = 0;
    drag.pitch = 0;
  });

  function resize() {
    const width = visual.clientWidth;
    const height = visual.clientHeight || window.innerHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  resize();
  const resizeObserver = "ResizeObserver" in window
    ? new ResizeObserver(resize)
    : null;
  if (resizeObserver) resizeObserver.observe(visual);
  else window.addEventListener("resize", resize);

  const frameInterval = 1000 / 30;
  let frameId = 0;
  let running = false;
  let previous = performance.now();
  let lastRendered = 0;

  function animate(time) {
    if (!running) return;
    frameId = requestAnimationFrame(animate);
    const elapsed = time - lastRendered;
    if (elapsed < frameInterval) return;
    lastRendered = time - (elapsed % frameInterval);
    const delta = Math.min(0.05, (time - previous) / 1000);
    previous = time;
    const seconds = time * 0.001;

    const idleYaw = drag.active
      ? -Math.PI / 4
      : -Math.PI / 4 + Math.sin(seconds * 0.24) * 0.08;
    const idlePitch = drag.active
      ? -0.08
      : -0.08 + Math.sin(seconds * 0.19) * 0.025;
    sculpture.rotation.y +=
      (idleYaw + drag.yaw + pointer.x * 0.08 - sculpture.rotation.y) *
      (drag.active ? 0.16 : 0.035);
    sculpture.rotation.x +=
      (idlePitch + drag.pitch - pointer.y * 0.03 - sculpture.rotation.x) *
      (drag.active ? 0.16 : 0.035);

    const activeStage = Math.floor((seconds * 0.42) % stagePlanes.length);
    stagePlanes.forEach((planes, stageIndex) => {
      planes.forEach((plane, planeIndex) => {
        const base =
          stageIndex === 3 ? 0.14 : stageIndex === 0 ? 0.2 : 0.18;
        plane.material.opacity =
          stageIndex === activeStage
            ? base + 0.13 + 0.06 * Math.sin(seconds * 2 - planeIndex * 0.35)
            : base;
      });
    });

    flowingEdges.forEach(({ tube: edge, dash }, index) => {
      dash.material.dashOffset -= delta * (0.45 + (index % 3) * 0.08);
      const edgePulse = 0.5 + 0.5 * Math.sin(seconds * 1.6 - index * 0.22);
      dash.material.opacity = 0.24 + edgePulse * 0.42;
      edge.material.emissiveIntensity = 0.25 + edgePulse * 0.35;
    });

    queryNodes.forEach((query, index) => {
      query.scale.setScalar(1 + 0.16 * Math.sin(seconds * 1.8 - index));
    });

    const probability = 0.5 + 0.3 * Math.sin(seconds * 0.7);
    probabilityBars[0].scale.y = 0.7 + (1 - probability) * 0.65;
    probabilityBars[1].scale.y = 0.7 + probability * 0.65;

    renderer.render(scene, camera);
  }

  const startAnimation = () => {
    if (running || reduceMotion) return;
    running = true;
    previous = performance.now();
    lastRendered = previous;
    frameId = requestAnimationFrame(animate);
  };
  const stopAnimation = () => {
    if (!running) return;
    running = false;
    cancelAnimationFrame(frameId);
  };

  let heroVisible = true;
  const viewportObserver = "IntersectionObserver" in window
    ? new IntersectionObserver(([entry]) => {
        heroVisible = entry.isIntersecting;
        document.body.classList.toggle("hero-out-of-view", !heroVisible);
        if (heroVisible && !document.hidden) startAnimation();
        else stopAnimation();
      }, { threshold: 0 })
    : null;

  if (reduceMotion) {
    renderer.render(scene, camera);
  } else {
    viewportObserver?.observe(hero);
    if (!viewportObserver) startAnimation();
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stopAnimation();
      else if (heroVisible) startAnimation();
    });
  }

  window.addEventListener(
    "pagehide",
    () => {
      stopAnimation();
      document.body.classList.remove("hero-out-of-view");
      viewportObserver?.disconnect();
      resizeObserver?.disconnect();
      renderer.dispose();
    },
    { once: true }
  );
}

if (THREE) {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}
