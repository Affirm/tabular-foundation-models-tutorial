/* Small, deterministic k-nearest neighbors classifier for the playground. */
export function predictKnn(X, y, query, requestedK) {
  if (!X.length) return null;

  const count = Math.min(requestedK, X.length);

  const neighbors = X.map((point, index) => ({
    index,
    distance: point.reduce((sum, value, axis) => sum + (value - (query[axis] ?? 0)) ** 2, 0),
  })).sort((a, b) => a.distance - b.distance || a.index - b.index).slice(0, count);
  const class1 = neighbors.reduce((sum, item) => sum + (y[item.index] === 1 ? 1 : 0), 0);
  const classIndex = class1 === count / 2 ? y[neighbors[0].index] : (class1 > count / 2 ? 1 : 0);

  return { proba: [1 - class1 / count, class1 / count], classIndex, neighbors };
}
