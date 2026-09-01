/* Decision ledger: append-only, hash-chained, tamper-evident.
 *
 * record_hash = sha256(prev_hash + canonical_json(record)), so altering any
 * record invalidates every hash after it. Verify walks the chain server-side
 * and reports the FIRST break, which is where tampering started.
 */

import { api } from "../api.js";
import { el, mount, ms, int, shortHash, when, loading, empty, errorBox } from "../util.js";

/* Two shapes carry a decision record and they are not the same:
 *   - the SSE `record` event is a flat DecisionRecord dump
 *   - /ledger/recent returns stored rows, where the DecisionRecord is a JSON
 *     *string* under `payload` and only a few columns are hoisted to the top
 * Normalise both to the flat shape so recordCard only has to know one.
 */
function normalise(row) {
  let inner = {};
  if (typeof row.payload === "string") {
    try {
      inner = JSON.parse(row.payload);
    } catch {
      inner = {};
    }
  } else if (row.payload && typeof row.payload === "object") {
    inner = row.payload;
  }
  // Hoisted columns win: they are what the chain was actually computed over.
  return { ...inner, ...row, payload: undefined };
}

function recordCard(r) {
  const row = (label, value, cls) =>
    el("div", { class: "row" },
      el("span", { class: "lbl" }, label),
      el("span", { class: cls || "" }, value));

  const flags = [
    r.degraded_mode && "degraded",
    r.abstained && "abstained",
    r.regenerated && "regenerated",
    r.escalated && "escalated",
    r.cache_hit && "cache hit",
    r.cache_gated_miss && "cache gated-miss",
    r.leaked_chunks > 0 && `${r.leaked_chunks} leaked chunk(s)`,
  ].filter(Boolean);

  return el("div", { class: "rec" },
    row("record_id", r.seq === undefined ? r.record_id : `#${r.seq} · ${r.record_id}`),
    row("arm / band", `${r.arm} · ${r.policy?.band ?? r.band ?? "—"} · action ${r.action}`),
    row("use case", `${r.use_case} · ${r.geography}`),
    r.corrects_record_id ? row("corrects", r.corrects_record_id) : null,
    r.policy?.detectors ? row("detectors", r.policy.detectors.join(", ")) : null,
    row("entities", `${(r.entities_masked || []).join(", ") || "—"} (types only, never values)`),
    flags.length ? row("flags", flags.join(" · ")) : null,
    row("prev_hash", shortHash(r.prev_hash), "chain"),
    row("record_hash", shortHash(r.record_hash), "chain"),
    row("timings", `ttft ${ms(r.ttft_ms)} · total ${ms(r.total_latency_ms)} · guard ${ms(r.guard_overhead_ms)}`),
    row("tokens", `in ${int(r.tokens_in)} · out ${int(r.tokens_out)} · ${r.signal_mode}`),
    row("timestamp", when(r.timestamp)));
}

export const ledgerView = {
  title: "Ledger",

  async mount(root, ctx) {
    const status = el("span", { class: "note" });
    const verifyBtn = el("button", {}, "Verify hash chain");
    const refreshBtn = el("button", {}, "Refresh");
    const limitSel = el("select", { "aria-label": "Records to show" },
      [10, 25, 50, 100].map((n) => el("option", { value: n, selected: n === 25 }, `${n} records`)));
    const list = el("div", {}, loading("Loading records…"));

    mount(root, el("div", {},
      el("p", { class: "prose" },
        "Every decision appends a record where ",
        el("code", {}, "record_hash = sha256(prev_hash + canonical_json(record))"),
        ". Append-only is enforced by SQLite triggers — ",
        el("code", {}, "UPDATE"), " and ", el("code", {}, "DELETE"),
        " raise — so corrections are new records referencing the original (INV-6). ",
        el("strong", {}, "Raw PII is never stored: "),
        "records carry entity types, span offsets and a salted hash."),
      el("div", { class: "toolbar" }, verifyBtn, refreshBtn, limitSel, status),
      list));

    const load = async () => {
      list.replaceChildren(loading("Loading records…"));
      try {
        const data = await api.ledgerRecent(Number(limitSel.value));
        const records = data.records || [];
        status.textContent = `${int(data.count)} record(s) in the ledger`;
        if (!records.length) {
          list.replaceChildren(el("div", { class: "card" },
            empty("The ledger is empty.",
              "Run the A/B demo on the Live demo tab to append decision records.")));
          return;
        }
        // Newest first reads better than chain order for an operator.
        list.replaceChildren(...[...records].reverse().map((row) => recordCard(normalise(row))));
      } catch (err) {
        list.replaceChildren(el("div", { class: "card" }, el("div", { class: "body" }, errorBox(err))));
      }
    };

    verifyBtn.addEventListener("click", async () => {
      verifyBtn.disabled = true;
      status.replaceChildren(document.createTextNode("verifying…"));
      try {
        const d = await api.ledgerVerify();
        status.replaceChildren(d.ok
          ? el("span", {}, el("span", { class: "pill ok" }, "chain intact"),
              ` ${int(d.records)} record(s)`)
          : el("span", {}, el("span", { class: "pill bad" }, "CHAIN BREAK"),
              ` at seq ${d.first_break.seq} — ${d.first_break.kind} ` +
              `(record ${d.first_break.record_id})`));
      } catch (err) {
        status.replaceChildren(errorBox(err));
      }
      verifyBtn.disabled = false;
    });

    refreshBtn.addEventListener("click", load);
    ctx.onRecordAppended(load);
    await load();
  },
};
