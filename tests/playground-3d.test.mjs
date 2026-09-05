import assert from "node:assert/strict";
import test from "node:test";

import {
  classIndexFromProbability,
  contourSegments2d,
  createProjection,
  isosurfaceIntersections,
  predictionColorFromProbability,
  rankContextAttention,
  sliceInterpolation,
} from "../materials/website/js/playground3d.js";

test("prediction color directly encodes class 1 probability", () => {
  assert.deepEqual(predictionColorFromProbability(0), [46, 166, 214]);
  assert.deepEqual(predictionColorFromProbability(0.5), [112, 125, 145]);
  assert.deepEqual(predictionColorFromProbability(1), [242, 114, 75]);
  assert.deepEqual(predictionColorFromProbability(-1), [46, 166, 214]);
  assert.deepEqual(predictionColorFromProbability(2), [242, 114, 75]);
  assert.deepEqual(predictionColorFromProbability(Number.NaN), [112, 125, 145]);
});

test("predicted class uses the binary decision threshold", () => {
  assert.equal(classIndexFromProbability(0.49), 0);
  assert.equal(classIndexFromProbability(0.5), 1);
  assert.equal(classIndexFromProbability(1), 1);
  assert.equal(classIndexFromProbability(Number.NaN), 0);
});

test("context attention is ranked by weight with stable row order", () => {
  assert.deepEqual(rankContextAttention([0.1, 0.4, 0.4, 0.05], 3), [
    { index: 1, weight: 0.4 },
    { index: 2, weight: 0.4 },
    { index: 0, weight: 0.1 },
  ]);
});

test("context attention sanitizes invalid weights and respects limits", () => {
  assert.deepEqual(rankContextAttention([Number.NaN, -0.2, 0.3, Infinity], 5), [
    { index: 2, weight: 0.3 },
    { index: 0, weight: 0 },
    { index: 1, weight: 0 },
    { index: 3, weight: 0 },
  ]);
  assert.deepEqual(rankContextAttention([0.2], 0), []);
  assert.deepEqual(rankContextAttention(null), []);
});

test("perspective projection round-trips points on a fixed-z slice", () => {
  const projection = createProjection(560, 2.4, -Math.PI / 4, 0.48);
  const points = [[0, 0, 0], [-2.4, 2.4, -1], [1.25, -0.75, 1.8], [2.4, -2.4, 0.4]];

  for (const point of points) {
    const screen = projection.project(point);
    const restored = projection.unproject([screen.x, screen.y], point[2]);
    assert.ok(Math.abs(restored[0] - point[0]) < 1e-10);
    assert.ok(Math.abs(restored[1] - point[1]) < 1e-10);
    assert.equal(restored[2], point[2]);
  }
});

test("the third feature moves a projected point vertically and in depth", () => {
  const projection = createProjection(440, 2.4, 0.3, 0.55);
  const ground = projection.project([0.5, -0.25, 0]);
  const raised = projection.project([0.5, -0.25, 0.75]);

  assert.equal(raised.x, ground.x);
  assert.ok(raised.depth > ground.depth);
  assert.ok(raised.y < ground.y);
});

test("2D slices interpolate the sampled z planes", () => {
  assert.deepEqual(sliceInterpolation(0, 2.4, 8), { lower: 3, upper: 4, mix: 0.5 });
  assert.deepEqual(sliceInterpolation(-2.4, 2.4, 8), { lower: 0, upper: 0, mix: 0 });
  assert.deepEqual(sliceInterpolation(2.4, 2.4, 8), { lower: 7, upper: 7, mix: 0 });
});

test("isosurface intersections locate a threshold between grid samples", () => {
  const values = new Float32Array([
    0, 0, 0, 0,
    1, 1, 1, 1,
  ]);
  assert.deepEqual(isosurfaceIntersections(values, 2, 1), [
    [0, -0.5, -0.5],
    [0, -0.5, 0.5],
    [0, 0.5, -0.5],
    [0, 0.5, 0.5],
  ]);
});

test("isosurface intersections ignore incomplete grids", () => {
  assert.deepEqual(isosurfaceIntersections(null, 8, 2.4), []);
  assert.deepEqual(isosurfaceIntersections(new Float32Array(7), 2, 1), []);
});

test("2D contour segments trace the binary decision edge", () => {
  assert.deepEqual(contourSegments2d(new Float32Array([0, 1, 0, 1]), 2), [
    [[0.5, 0], [0.5, 1]],
  ]);
});

test("2D contour segments handle saddles and incomplete cells", () => {
  assert.equal(contourSegments2d(new Float32Array([1, 0, 0, 1]), 2).length, 2);
  assert.deepEqual(contourSegments2d(new Float32Array([0, -1, 0, 1]), 2), []);
});

test("2D contour segments handle exact-threshold vertices without degeneracy", () => {
  assert.deepEqual(contourSegments2d(new Float32Array([0.5, 0, 0, 0]), 2), []);
  assert.deepEqual(contourSegments2d(new Float32Array([0.5, 1, 0, 0]), 2), [
    [[1, 0.5], [0, 0]],
  ]);
});

test("2D contour saddles use the bilinear asymptotic decider", () => {
  const connectedCenter = contourSegments2d(new Float32Array([0.9, 0.4, 0.1, 0.8]), 2);
  assert.deepEqual(connectedCenter[0].map(([x, y]) => [Number(x.toFixed(3)), Number(y.toFixed(3))]), [
    [0.8, 0],
    [1, 0.25],
  ]);
  const separatedCenter = contourSegments2d(new Float32Array([0.6, 0, 0, 0.6]), 2);
  assert.deepEqual(separatedCenter[0].map(([x, y]) => [Number(x.toFixed(3)), Number(y.toFixed(3))]), [
    [0.167, 0],
    [0, 0.167],
  ]);
});
