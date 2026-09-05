export const PFN_TOY_HYPOTHESES = Object.freeze(
  [3, 4, 5, 6, 7, 8].flatMap((tau) => [
    Object.freeze({ tau, direction: "up" }),
    Object.freeze({ tau, direction: "down" }),
  ])
);

export function pfnToyPredict(rule, x) {
  return Number(rule.direction === "up" ? x >= rule.tau : x <= rule.tau);
}

export function pfnToyConsistentRules(hypotheses, context) {
  return hypotheses.filter((rule) =>
    context.every((point) => pfnToyPredict(rule, point.x) === point.y)
  );
}

export function pfnToyPosterior(hypotheses, context, queryX) {
  const survivors = pfnToyConsistentRules(hypotheses, context);
  if (!survivors.length) {
    throw new RangeError("Context is inconsistent with every rule in the toy prior.");
  }
  const votesForOne = survivors.filter(
    (rule) => pfnToyPredict(rule, queryX) === 1
  ).length;
  return {
    survivors,
    votesForOne,
    p1: votesForOne / survivors.length,
  };
}
