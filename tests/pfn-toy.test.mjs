import assert from "node:assert/strict";
import test from "node:test";

import {
  PFN_TOY_HYPOTHESES,
  pfnToyConsistentRules,
  pfnToyPosterior,
  pfnToyPredict,
} from "../materials/website/js/pfn-toy.js";

test("toy prior contains twelve threshold rules", () => {
  assert.equal(PFN_TOY_HYPOTHESES.length, 12);
  assert.equal(pfnToyPredict({ tau: 5, direction: "up" }, 5), 1);
  assert.equal(pfnToyPredict({ tau: 5, direction: "down" }, 6), 0);
});

test("a sampled rule remains consistent with context it generated", () => {
  for (const rule of PFN_TOY_HYPOTHESES) {
    const context = [1, 3, 5, 7, 10].map((x) => ({
      x,
      y: pfnToyPredict(rule, x),
    }));
    assert.ok(pfnToyConsistentRules(PFN_TOY_HYPOTHESES, context).includes(rule));
  }
});

test("posterior predictive is the vote among consistent rules", () => {
  const context = [
    { x: 1, y: 0 },
    { x: 10, y: 1 },
  ];
  const posterior = pfnToyPosterior(PFN_TOY_HYPOTHESES, context, 5);

  assert.equal(posterior.survivors.length, 6);
  assert.equal(posterior.votesForOne, 3);
  assert.equal(posterior.p1, 0.5);
});

test("inconsistent context is rejected", () => {
  const impossible = [
    { x: 1, y: 1 },
    { x: 5, y: 0 },
    { x: 10, y: 1 },
  ];
  assert.throws(
    () => pfnToyPosterior(PFN_TOY_HYPOTHESES, impossible, 6),
    /inconsistent/
  );
});
