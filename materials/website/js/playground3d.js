/* Pure projection helpers for the browser playground's prediction field. */

export function classIndexFromProbability(probability) {
  return Number.isFinite(probability) && probability >= 0.5 ? 1 : 0;
}

export function predictionColorFromProbability(probability) {
  const p = Number.isFinite(probability) ? Math.max(0, Math.min(1, probability)) : 0.5;
  const low = [46, 166, 214];
  const midpoint = [112, 125, 145];
  const high = [242, 114, 75];
  const start = p <= 0.5 ? low : midpoint;
  const end = p <= 0.5 ? midpoint : high;
  const mix = p <= 0.5 ? p * 2 : (p - 0.5) * 2;
  return start.map((channel, index) => Math.round(channel + (end[index] - channel) * mix));
}

export function rankContextAttention(weights, limit = 5) {
  if (!Array.isArray(weights) || limit <= 0) return [];
  return weights
    .map((weight, index) => ({
      index,
      weight: Number.isFinite(weight) && weight > 0 ? weight : 0,
    }))
    .sort((left, right) => right.weight - left.weight || left.index - right.index)
    .slice(0, Math.min(limit, weights.length));
}

export function sliceInterpolation(value, domain, gridSize) {
  const clamped = Math.max(-domain, Math.min(domain, value));
  const position = ((clamped + domain) / (domain * 2)) * gridSize - 0.5;
  const floor = Math.floor(position);
  const lower = Math.max(0, Math.min(gridSize - 1, floor));
  const upper = Math.max(0, Math.min(gridSize - 1, floor + 1));
  return { lower, upper, mix: lower === upper ? 0 : position - floor };
}

export function isosurfaceIntersections(values, gridSize, domain, threshold = 0.5) {
  if (!values || values.length !== gridSize ** 3 || gridSize < 2) return [];
  const index = (x, y, z) => (x * gridSize + y) * gridSize + z;
  const coordinate = (value) => -domain + ((value + 0.5) / gridSize) * domain * 2;
  const points = [];

  for (let x = 0; x < gridSize; x++) {
    for (let y = 0; y < gridSize; y++) {
      for (let z = 0; z < gridSize; z++) {
        const probability = values[index(x, y, z)];
        if (!Number.isFinite(probability) || probability < 0) continue;
        for (const [dx, dy, dz] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
          const nx = x + dx, ny = y + dy, nz = z + dz;
          if (nx >= gridSize || ny >= gridSize || nz >= gridSize) continue;
          const neighbor = values[index(nx, ny, nz)];
          if (!Number.isFinite(neighbor) || neighbor < 0) continue;
          if ((probability - threshold) * (neighbor - threshold) > 0 || probability === neighbor) continue;
          const mix = Math.max(0, Math.min(1, (threshold - probability) / (neighbor - probability)));
          points.push([
            coordinate(x + dx * mix),
            coordinate(y + dy * mix),
            coordinate(z + dz * mix),
          ]);
        }
      }
    }
  }
  return points;
}

export function contourSegments2d(values, gridSize, threshold = 0.5) {
  if (!values || values.length !== gridSize ** 2 || gridSize < 2) return [];
  const at = (x, y) => values[y * gridSize + x];
  const interpolate = (a, b) => a === b ? 0.5 : Math.max(0, Math.min(1, (threshold - a) / (b - a)));
  const segments = [];

  for (let x = 0; x < gridSize - 1; x++) {
    for (let y = 0; y < gridSize - 1; y++) {
      const valuesAtCorners = {
        tl: at(x, y),
        tr: at(x + 1, y),
        br: at(x + 1, y + 1),
        bl: at(x, y + 1),
      };
      if (Object.values(valuesAtCorners).some((value) => !Number.isFinite(value) || value < 0)) continue;
      const crossings = [];
      const addCrossing = (name, a, b, point) => {
        if ((a >= threshold) === (b >= threshold)) return;
        crossings.push({ name, point: point(interpolate(a, b)) });
      };
      addCrossing("top", valuesAtCorners.tl, valuesAtCorners.tr, (mix) => [x + mix, y]);
      addCrossing("right", valuesAtCorners.tr, valuesAtCorners.br, (mix) => [x + 1, y + mix]);
      addCrossing("bottom", valuesAtCorners.bl, valuesAtCorners.br, (mix) => [x + mix, y + 1]);
      addCrossing("left", valuesAtCorners.tl, valuesAtCorners.bl, (mix) => [x, y + mix]);

      const uniqueCrossings = crossings.filter((crossing, index) =>
        crossings.findIndex((candidate) =>
          Math.abs(candidate.point[0] - crossing.point[0]) < 1e-9 &&
          Math.abs(candidate.point[1] - crossing.point[1]) < 1e-9
        ) === index);
      if (uniqueCrossings.length === 2) {
        segments.push([uniqueCrossings[0].point, uniqueCrossings[1].point]);
      } else if (uniqueCrossings.length === 4) {
        const byName = Object.fromEntries(uniqueCrossings.map((crossing) => [crossing.name, crossing.point]));
        const determinant =
          (valuesAtCorners.tl - threshold) * (valuesAtCorners.br - threshold) -
          (valuesAtCorners.tr - threshold) * (valuesAtCorners.bl - threshold);
        const pairTopRight = determinant >= 0;
        const pairs = pairTopRight
          ? [["top", "right"], ["bottom", "left"]]
          : [["top", "left"], ["right", "bottom"]];
        for (const [first, second] of pairs) segments.push([byName[first], byName[second]]);
      }
    }
  }
  return segments;
}

export function createProjection(size, domain, yaw, pitch) {
  const scale = size * 0.28;
  const centerX = size * 0.5;
  const centerY = size * 0.56;
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);
  const pitchScale = Math.max(0.2, Math.sin(pitch));
  const liftScale = size * 0.24;

  return {
    project(point, z = point[2] || 0) {
      const nx = point[0] / domain;
      const ny = point[1] / domain;
      const nz = z / domain;
      const rotatedX = nx * cosYaw - ny * sinYaw;
      const rotatedY = nx * sinYaw + ny * cosYaw;
      return {
        x: centerX + rotatedX * scale,
        y: centerY + rotatedY * scale * pitchScale - nz * liftScale,
        depth: rotatedY + nz * 0.28,
      };
    },

    unproject(screenPoint, z = 0) {
      const nz = z / domain;
      const rotatedX = (screenPoint[0] - centerX) / scale;
      const rotatedY = (screenPoint[1] - centerY + nz * liftScale) / (scale * pitchScale);
      return [
        (rotatedX * cosYaw + rotatedY * sinYaw) * domain,
        (-rotatedX * sinYaw + rotatedY * cosYaw) * domain,
        z,
      ];
    },
  };
}
