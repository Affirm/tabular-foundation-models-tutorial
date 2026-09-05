import assert from "node:assert/strict";
import test from "node:test";

import {
  fitDecisionTree,
  fitLogisticRegression,
  predictDecisionTree,
  predictLogisticRegression,
} from "../materials/website/js/classical-models.js";

test("logistic regression learns a linear boundary", () => {
  const X = [
    [-2, -1], [-1, -2], [-1, 0], [0, -1],
    [1, 0], [0, 1], [1, 2], [2, 1],
  ];
  const y = [0, 0, 0, 0, 1, 1, 1, 1];
  const model = fitLogisticRegression(X, y);
  const predictions = X.map((row) => Number(predictLogisticRegression(model, row) >= 0.5));
  assert.deepEqual(predictions, y);
});

test("decision tree learns an axis-aligned nonlinear boundary", () => {
  const X = [
    [-2, -2], [-1, -1], [-2, 2], [-1, 1],
    [1, -1], [2, -2], [1, 1], [2, 2],
  ];
  const y = [0, 0, 0, 0, 0, 0, 1, 1];
  const model = fitDecisionTree(X, y, { maxDepth: 3, minLeaf: 1 });
  const predictions = X.map((row) => Number(predictDecisionTree(model, row) >= 0.5));
  assert.deepEqual(predictions, y);
});

test("classical probabilities stay bounded", () => {
  const X = [[-1], [0], [1], [2]];
  const y = [0, 0, 1, 1];
  const linear = fitLogisticRegression(X, y);
  const tree = fitDecisionTree(X, y, { minLeaf: 1 });
  for (const row of [[-10], [0.5], [10]]) {
    assert.ok(predictLogisticRegression(linear, row) >= 0 && predictLogisticRegression(linear, row) <= 1);
    assert.ok(predictDecisionTree(tree, row) >= 0 && predictDecisionTree(tree, row) <= 1);
  }
});
