import assert from "node:assert/strict";
import test from "node:test";

import { predictKnn } from "../materials/website/js/knn.js";

test("predictKnn selects the nearest k rows and returns their vote", () => {
  const result = predictKnn(
    [[0, 0], [1, 0], [0, 2], [4, 4], [-1, 0]],
    [0, 1, 1, 1, 0],
    [0, 0],
    3,
  );

  assert.deepEqual(result.neighbors.map(({ index }) => index), [0, 1, 4]);
  assert.ok(Math.abs(result.proba[0] - 2 / 3) < Number.EPSILON);
  assert.ok(Math.abs(result.proba[1] - 1 / 3) < Number.EPSILON);
});

test("predictKnn resolves equal distances by original row order", () => {
  const result = predictKnn([[1, 0], [-1, 0], [0, 1]], [1, 0, 0], [0, 0], 1);

  assert.equal(result.neighbors[0].index, 0);
  assert.deepEqual(result.proba, [0, 1]);
});

test("predictKnn includes the third feature in distance", () => {
  const result = predictKnn(
    [[0, 0, 2], [0.2, 0, 0], [0, 0, -0.4]],
    [1, 0, 1],
    [0, 0, 0],
    1,
  );

  assert.equal(result.neighbors[0].index, 1);
  assert.equal(result.classIndex, 0);
});

test("predictKnn uses every available row and breaks vote ties by the nearest row", () => {
  const result = predictKnn([[0, 0], [0.1, 0]], [0, 1], [0, 0], 5);

  assert.equal(result.neighbors.length, 2);
  assert.deepEqual(result.proba, [0.5, 0.5]);
  assert.equal(result.classIndex, 0);
});

test("predictKnn tie regions follow the nearest row", () => {
  const X = [[-1, 0], [1, 0]];
  const y = [0, 1];

  assert.equal(predictKnn(X, y, [-0.5, 0], 5).classIndex, 0);
  assert.equal(predictKnn(X, y, [0.5, 0], 5).classIndex, 1);
});

test("predictKnn returns no prediction for an empty context", () => {
  assert.equal(predictKnn([], [], [0, 0], 5), null);
});
