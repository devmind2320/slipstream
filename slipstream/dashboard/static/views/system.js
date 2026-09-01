/* System status: which engines are actually live behind this gateway.
 *
 * This exists so nobody has to trust a README about which backends are in play.
 * The registry fingerprint shown here is the same value the eval harness asserts
 * is identical across all three arms (INV-3).
 */

import { api } from "../api.js";
import { el, mount, table, loading, errorBox } from "../util.js";

function boolPill(v) {
  return el("span", { class: `pill ${v ? "on" : ""}` }, v ? "enabled" : "default");
}

function backendsCard(backends) {
  const entries = Object.entries(backends || {});
  if (!entries.length) return null;
  return el("section", { class: "card" },
    el("h2", {}, "Active backends"),
    table(["Component", "Backend"],
      entries.map(([k, v]) => [k, typeof v === "boolean" ? boolPill(v) : String(v)]),
      { numeric: false }));
}

export const systemView = {
  title: "System",

  async mount(root) {
    mount(root, loading("Querying gateway…"));
    let health;
    try {
      health = await api.health();
    } catch (err) {
      mount(root, el("div", { class: "card" }, el("div", { class: "body" }, errorBox(err))));
      return;
    }

    mount(root, el("div", {},
      el("p", { class: "prose" },
        el("strong", {}, "First-run download size on the default path is 0 bytes. "),
        "The default detector engines are deterministic pure-Python stand-ins, which is what keeps " +
        "the offline guarantee (INV-5) true on a clean clone. The heavier models named in the " +
        "specification are opt-in through environment variables and are listed below when active."),
      el("div", { class: "grid two" },
        el("section", { class: "card" },
          el("h2", {}, "Gateway",
            el("span", { class: `pill ${health.ok ? "ok" : "bad"}` }, health.ok ? "healthy" : "degraded")),
          el("div", { class: "body" },
            el("dl", { class: "kv" },
              el("dt", {}, "provider"), el("dd", {}, String(health.provider ?? "—")),
              el("dt", {}, "registry fingerprint"), el("dd", {}, String(health.registry_fingerprint ?? "—")),
              el("dt", {}, "policy packs"), el("dd", {}, (health.packs || []).join(", ") || "—")))),
        backendsCard(health.backends) || el("div")),
      el("section", { class: "card", style: "margin-top:16px" },
        el("h2", {}, "Detectors in the registry"),
        el("div", { class: "body" },
          el("div", { style: "display:flex;gap:7px;flex-wrap:wrap" },
            (health.detectors || []).map((d) =>
              el("span", { class: `pill ${d === "pii" || d === "injection" ? "on" : ""}` }, d))),
          el("p", { class: "prose", style: "margin:14px 0 0" },
            el("code", {}, "pii"), " and ", el("code", {}, "injection"),
            " are floor detectors: they run inline, at full depth, in every band including the " +
            "lowest, and degradation cannot drop them. A misjudged risk estimate can cost " +
            "latency; it cannot open the leak path (INV-1)."))),
      el("section", { class: "card", style: "margin-top:16px" },
        el("h2", {}, "Endpoints"),
        table(["Route", "Purpose"], [
          [el("code", {}, "POST /v1/chat/completions"), "OpenAI-shaped, SSE when stream=true"],
          [el("code", {}, "GET /demo/stream"), "A/B demo feed used by the Live demo tab"],
          [el("code", {}, "GET /demo/scenarios"), "Available demo scenarios"],
          [el("code", {}, "GET /policy/packs"), "Loaded policy packs"],
          [el("code", {}, "GET /ledger/recent"), "Recent decision records"],
          [el("code", {}, "GET /ledger/verify"), "Walk the hash chain, report the first break"],
          [el("code", {}, "GET /results"), "Evaluation results, if `make eval` has run"],
          [el("code", {}, "GET /healthz"), "This page"],
        ], { numeric: false }))));
  },
};
