import {
  fitDecisionTree,
  fitLogisticRegression,
  predictDecisionTree,
  predictLogisticRegression,
} from "./classical-models.js";
import { predictKnn } from "./knn.js";

let X = [];
let y = [];
let k = 5;
let linear = null;
let tree = null;

function probabilities(row) {
  const knn = predictKnn(X, y, row, k);
  return {
    linear: predictLogisticRegression(linear, row),
    tree: predictDecisionTree(tree, row),
    knn: knn?.proba?.[1] ?? 0.5,
  };
}

self.onmessage = (event) => {
  const message = event.data;
  try {
    if (message.type === "prepare") {
      X = message.X;
      y = message.y;
      k = message.k || 5;
      linear = fitLogisticRegression(X, y);
      tree = fitDecisionTree(X, y);
      self.postMessage({ type: "prepared", tag: message.tag });
      return;
    }
    if (message.type === "setK") {
      k = message.k || 5;
      return;
    }
    if (message.type === "query") {
      if (!linear || !tree) throw new Error("Prepare classical models before prediction.");
      self.postMessage({
        type: "query",
        tag: message.tag,
        probabilities: probabilities(message.X),
      });
      return;
    }
    if (message.type === "grid") {
      if (!linear || !tree) throw new Error("Prepare classical models before field prediction.");
      const linearGrid = new Float32Array(message.X.length);
      const treeGrid = new Float32Array(message.X.length);
      const knnGrid = new Float32Array(message.X.length);
      message.X.forEach((row, index) => {
        const result = probabilities(row);
        linearGrid[index] = result.linear;
        treeGrid[index] = result.tree;
        knnGrid[index] = result.knn;
      });
      self.postMessage(
        {
          type: "grid",
          tag: message.tag,
          grids: { linear: linearGrid, tree: treeGrid, knn: knnGrid },
        },
        [linearGrid.buffer, treeGrid.buffer, knnGrid.buffer]
      );
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      tag: message.tag,
      message: String(error?.stack || error),
    });
  }
};
