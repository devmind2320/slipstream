/* Policy packs, as the gateway currently has them loaded.
 *
 * Packs are YAML, hot-reloaded on change, so this view is the fastest way to
 * confirm an edit took effect without restarting anything.
 */

import { api } from "../api.js";
import { el, mount, int, num, table, loading, errorBox } from "../util.js";

const BAND_ORDER = ["low", "medium", "high"];

function bandTable(pack) {
  const bands = pack.bands || {};
  const present = BAND_ORDER.filter((b) => bands[b]).concat(
    Object.keys(bands).filter((b) => !BAND_ORDER.includes(b)));
  const rows = present.map((b) => {
    const cfg = bands[b] || {};
    return [
      el("span", { class: `pill ${b}` }, b),
      int(cfg.holdback_tokens),
      String(cfg.sync ?? "—"),
      cfg.guard_budget_ms == null ? "—" : `${int(cfg.guard_budget_ms)} ms`,
    ];
  });
  if (!rows.length) return el("div", { class: "empty" }, "No bands declared.");
  return table(["Band", "Holdback (tok)", "Sync", "Guard budget"], rows);
}

function detectorRow(label, names, isFloor) {
  return el("div", { style: "display:flex;gap:8px;align-items:baseline;padding:3px 0;flex-wrap:wrap" },
    el("span", { class: "note", style: "min-width:74px" }, label),
    (names || []).length
      ? (names || []).map((n) => el("span", { class: `pill ${isFloor ? "on" : ""}` }, n))
      : el("span", { class: "note" }, "—"));
}

function packCard(name, pack) {
  const det = pack.detectors || {};
  const geo = pack.geography || {};
  return el("section", { class: "card" },
    el("h2", {}, name,
      pack.min_band ? el("span", { class: `pill ${pack.min_band}` }, `min ${pack.min_band}`) : null),
    el("div", { class: "body" },
      el("p", { class: "prose", style: "margin-bottom:12px" },
        "Uncertainty margin ",
        el("code", {}, num(pack.uncertainty_margin, 2)),
        " — INV-2 rounds a score inside this zone up to the next band, never down."),
      detectorRow("floor", det.floor, true),
      (det.low || []).length ? detectorRow("low", det.low) : null,
      (det.medium || []).length ? detectorRow("medium", det.medium) : null,
      (det.high || []).length ? detectorRow("high", det.high) : null,
      el("div", { style: "margin-top:14px" }, bandTable(pack)),
      Object.keys(geo).length
        ? el("div", { style: "margin-top:14px" },
            el("div", { class: "note", style: "margin-bottom:6px" }, "Geography"),
            table(["Region", "PII entities", "Ledger required"],
              Object.entries(geo).map(([region, cfg]) => [
                region,
                (cfg.pii_entities || []).join(", ") || "—",
                String(cfg.require_ledger ?? false),
              ]), { numeric: false }))
        : null,
      pack.escalation
        ? el("p", { class: "prose", style: "margin:14px 0 0" },
            "Escalation budget ",
            el("code", {}, pack.escalation.budget == null ? "none" : String(pack.escalation.budget)),
            pack.escalation.queue ? " → queue " : "",
            pack.escalation.queue ? el("code", {}, pack.escalation.queue) : null)
        : null));
}

export const policyView = {
  title: "Policy",

  async mount(root) {
    mount(root, loading("Loading policy packs…"));
    let packs;
    try {
      packs = await api.packs();
    } catch (err) {
      mount(root, el("div", { class: "card" }, el("div", { class: "body" }, errorBox(err))));
      return;
    }

    const entries = Object.entries(packs);
    mount(root, el("div", {},
      el("p", { class: "prose" },
        el("strong", {}, "Behaviour varies by use case and geography through YAML, not code. "),
        "These packs are read from slipstream/policy/packs/ and hot-reloaded on change — edit a " +
        "file and reload this tab to see it applied. Detectors listed under ",
        el("code", {}, "floor"),
        " run inline in every band and cannot be degraded away (INV-1)."),
      entries.length
        ? el("div", { class: "grid auto" }, entries.map(([n, p]) => packCard(n, p)))
        : el("div", { class: "card" }, el("div", { class: "empty" }, "No packs loaded."))));
  },
};
