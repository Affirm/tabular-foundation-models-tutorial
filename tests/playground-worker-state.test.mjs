import assert from "node:assert/strict";
import test from "node:test";

import { recoverWorkerError } from "../materials/website/js/playground-worker-state.js";

function pendingState() {
  return {
    loading: true, prepared: true, prepTag: 3,
    qInFlight: true, qQueued: true, queryTag: 7,
    qProba: [0.3, 0.7], qClass: 1,
    selectedView: { viewIndex: 0 }, selectedViewProbability: [0.2, 0.8],
    attention: [1], attentionBlock: 12, attentionHeads: 8,
    gridBusy: true, gridTag: 5, grid: new Float32Array([0.4]), surface: {},
  };
}

test("a failed query releases the request and preserves queued work and the active grid", () => {
  const state = pendingState();
  assert.equal(recoverWorkerError(state, { requestType: "inspect", tag: 7 }), true);
  assert.equal(state.qInFlight, false);
  assert.equal(state.qQueued, true);
  assert.equal(state.qProba, null);
  assert.equal(state.selectedView, null);
  assert.equal(state.attention, null);
  assert.equal(state.loading, true);
  assert.equal(state.gridBusy, true);
  assert.equal(state.prepared, true);
});

test("errors from an old query, context, or grid do not clear current requests", () => {
  for (const requestType of ["inspect", "predict", "prepare", "grid"]) {
    const state = pendingState();
    const before = structuredClone(state);
    assert.equal(recoverWorkerError(state, { requestType, tag: -1 }), false);
    assert.deepEqual(state, before);
  }
});

test("a failed grid clears the partial field without releasing an active query", () => {
  const state = pendingState();
  assert.equal(recoverWorkerError(state, { requestType: "grid", tag: 5 }), true);
  assert.equal(state.gridBusy, false);
  assert.equal(state.grid, null);
  assert.equal(state.surface, null);
  assert.equal(state.qInFlight, true);
});

test("a failed context prevents prediction until preparation succeeds", () => {
  const state = pendingState();
  assert.equal(recoverWorkerError(state, { requestType: "prepare", tag: 3 }), true);
  assert.equal(state.prepared, false);
  assert.equal(state.qInFlight, false);
  assert.equal(state.qQueued, false);
});

test("a failed load permits retry without clearing unrelated work", () => {
  const state = pendingState();
  assert.equal(recoverWorkerError(state, { requestType: "load" }), true);
  assert.equal(state.loading, false);
  assert.equal(state.qInFlight, true);
});
