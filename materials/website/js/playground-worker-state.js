export function recoverWorkerError(state, message) {
  switch (message.requestType) {
    case "inspect":
    case "predict":
      if (message.tag !== state.queryTag) return false;
      state.qInFlight = false;
      state.qProba = null;
      state.qClass = null;
      state.selectedView = null;
      state.selectedViewProbability = null;
      state.attention = null;
      state.attentionBlock = null;
      state.attentionHeads = null;
      break;
    case "prepare":
      if (message.tag !== state.prepTag) return false;
      state.prepared = false;
      state.qInFlight = false;
      state.qQueued = false;
      break;
    case "grid":
      if (message.tag !== state.gridTag) return false;
      state.gridBusy = false;
      state.grid = null;
      state.surface = null;
      break;
    case "load":
      state.loading = false;
      break;
    default:
      return false;
  }
  return true;
}
