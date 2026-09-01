/* Every gateway endpoint the dashboard touches, named once.
 *
 * Views never call fetch() directly. When app.py changes a route, this file is
 * the only place that has to follow.
 */

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function getJSON(path) {
  let res;
  try {
    res = await fetch(path, { headers: { Accept: "application/json" } });
  } catch (cause) {
    // Network-level failure: the gateway is not running or was restarted.
    throw new ApiError(`Cannot reach the gateway at ${path}`, 0, String(cause));
  }
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).detail || "";
    } catch {
      /* body was not JSON; the status alone is the message */
    }
    throw new ApiError(`${res.status} ${res.statusText}`, res.status, detail);
  }
  return res.json();
}

export const api = {
  health: () => getJSON("/healthz"),
  scenarios: () => getJSON("/demo/scenarios"),
  packs: () => getJSON("/policy/packs"),
  results: () => getJSON("/results"),
  ledgerRecent: (limit = 25) => getJSON(`/ledger/recent?limit=${limit}`),
  ledgerVerify: () => getJSON("/ledger/verify"),

  /** SSE URL for one pane of the A/B demo. EventSource is opened by the view. */
  demoStreamURL: (arm, scenario) =>
    `/demo/stream?arm=${encodeURIComponent(arm)}&scenario=${encodeURIComponent(scenario)}`,
};
