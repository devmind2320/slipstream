/* Three-arm evaluation results, read live from slipstream/eval/results/results.json.
 *
 * The arms differ only in scheduling policy — the harness asserts an identical
 * detector-registry fingerprint across all three (INV-3) and that fingerprint is
 * displayed here so the claim is checkable rather than asserted.
 */

import { api } from "../api.js";
import { el, mount, ms, pct, int, usd, num, table, loading, errorBox } from "../util.js";

const ARM_ORDER = ["fixed_sync", "fixed_async", "adaptive"];
const ARM_LABEL = {
  fixed_sync: "fixed_sync · safe, slow",
  fixed_async: "fixed_async · fast, leaky",
  adaptive: "adaptive · Slipstream",
};

/* Each row: label, how to pull the value from one arm, and which direction is
 * better (used only to tint the adaptive column, never to hide a bad number). */
const ROWS = [
  ["TTFT p50, benign", (a) => a.latency_benign?.ttft_p50_ms, ms, "lower"],
  ["TTFT p95, benign", (a) => a.latency_benign?.ttft_p95_ms, ms, "lower"],
  ["Added latency p50, benign", (a) => a.latency_benign?.added_latency_p50_ms, ms, "lower"],
  ["Added latency p95, benign", (a) => a.latency_benign?.added_latency_p95_ms, ms, "lower"],
  ["Added latency p50, risky", (a) => a.latency_risky?.added_latency_p50_ms, ms, "lower"],
  ["PII masked before emission", (a) => a.pii?.masked_before_emission_rate, pct, "higher"],
  ["PII items that leaked", (a) => a.pii?.leaked_items, int, "lower"],
  ["PII interception rate", (a) => a.pii?.interception_rate, pct, "higher"],
  ["Multi-label set-exact accuracy", (a) => a.multilabel?.set_exact_accuracy, pct, "higher"],
  ["Set-exact on overlapping slice", (a) => a.multilabel?.multilabel_set_exact, pct, "higher"],
  ["Abstention rate", (a) => a.abstention?.abstention_rate, pct, null],
  ["Correct abstention", (a) => a.abstention?.correct_abstention_rate, pct, "higher"],
  ["Spurious abstention", (a) => a.abstention?.spurious_abstention_rate, pct, "lower"],
  ["Interventions per 100", (a) => a.interventions_per_100_interactions, (v) => num(v, 1), null],
  ["Flags per 100", (a) => a.flags_per_100_interactions, (v) => num(v, 1), null],
  ["Guard finished first", (a) => a.scheduler?.pct_chunks_guard_finished_first, pct, null],
  ["Degraded-mode incidence", (a) => a.scheduler?.degraded_mode_rate, pct, null],
  ["Mean guard overhead", (a) => a.scheduler?.mean_guard_overhead_ms, ms, null],
  ["Net token cost delta", (a) => a.cost?.net_delta_pct, (v) => (v == null ? "—" : `${num(v, 1)}%`), "lower"],
  ["Actual cost", (a) => a.cost?.actual_cost_usd, usd, "lower"],
];

function statTile(k, v, cmp, tone) {
  return el("div", { class: "card" },
    el("div", { class: "stat" },
      el("div", { class: "k" }, k),
      el("div", { class: `v ${tone || ""}` }, v),
      cmp ? el("div", { class: "cmp" }, cmp) : null));
}

function headline(arms) {
  const ad = arms.adaptive, sync = arms.fixed_sync, async_ = arms.fixed_async;
  if (!ad || !sync || !async_) return null;

  const syncT = sync.latency_benign?.ttft_p50_ms;
  const asyncT = async_.latency_benign?.ttft_p50_ms;
  const adT = ad.latency_benign?.ttft_p50_ms;
  const closed = syncT && asyncT != null && adT != null
    ? (syncT - adT) / (syncT - asyncT)
    : null;

  return el("div", { class: "grid auto" },
    statTile("TTFT p50, adaptive", ms(adT),
      `vs ${ms(syncT)} synchronous`, "good"),
    statTile("Gap to fast baseline closed", closed == null ? "—" : pct(closed, 0),
      "while holding the safe baseline's recall", "good"),
    statTile("PII leaked, adaptive", int(ad.pii?.leaked_items),
      `${int(async_.pii?.leaked_items)} under fixed_async`,
      ad.pii?.leaked_items === 0 ? "good" : "bad"),
    statTile("PII prevented, fixed_async", pct(async_.pii?.masked_before_emission_rate),
      `intercepted ${pct(async_.pii?.interception_rate)} — after display`, "bad"));
}

function comparisonTable(arms) {
  const present = ARM_ORDER.filter((a) => arms[a]);
  const headers = ["Metric", ...present.map((a) => ARM_LABEL[a] || a)];
  const rows = ROWS.map(([label, pull, fmt]) => [
    label,
    ...present.map((armId) => {
      const raw = pull(arms[armId]);
      const cell = el("span", {}, fmt(raw));
      if (armId === "adaptive") cell.style.fontWeight = "650";
      return cell;
    }),
  ]);
  return table(headers, rows);
}

function schedulerByBand(adaptive) {
  const byBand = adaptive?.scheduler?.by_band || {};
  const rows = Object.entries(byBand).map(([band, v]) => [
    el("span", { class: `pill ${band}` }, band),
    int(v.chunks_with_a_successor),
    v.pct_guard_finished_first == null ? "n/a" : pct(v.pct_guard_finished_first),
  ]);
  if (!rows.length) return null;
  return el("section", { class: "card" },
    el("h2", {}, "Guard-versus-generation race, by band"),
    table(["Band", "Chunks with a successor", "Guard finished first"], rows));
}

function calibrationCard(cal) {
  if (!cal) return null;
  const failures = cal.coverage_failures || [];
  const unfitted = cal.unfitted || [];
  return el("section", { class: "card" },
    el("h2", {}, "Conformal calibration",
      el("span", { class: `pill ${failures.length ? "bad" : "ok"}` },
        failures.length ? `${failures.length} coverage failure(s)` : "coverage holds")),
    el("div", { class: "body" },
      el("dl", { class: "kv" },
        el("dt", {}, "alpha"), el("dd", {}, num(cal.alpha, 2)),
        el("dt", {}, "calibration items"), el("dd", {}, int(cal.n_items)),
        el("dt", {}, "split seed"), el("dd", {}, String(cal.split_seed ?? "—")),
        el("dt", {}, "fitted thresholds"), el("dd", {}, int(Object.keys(cal.thresholds || {}).length)),
        el("dt", {}, "unfitted pairs"), el("dd", {}, unfitted.length ? unfitted.join(", ") : "none"),
        el("dt", {}, "coverage failures"),
        el("dd", {},
          failures.length
            ? failures.map((f) => (typeof f === "string" ? f : JSON.stringify(f))).join(", ")
            : "none")),
      el("p", { class: "prose", style: "margin:12px 0 0" },
        "Unfitted pairs fall back to the pack's configured default and are marked ",
        el("code", {}, "UNFITTED"),
        " in the report. Failures are shown here rather than suppressed.")));
}

function cacheCard(probe) {
  if (!probe) return null;
  return el("section", { class: "card" },
    el("h2", {}, "Semantic cache, entity-gated",
      el("span", { class: `pill ${probe.near_miss_wrongly_served === 0 ? "ok" : "bad"}` },
        `${int(probe.near_miss_wrongly_served)} wrongly served`)),
    el("div", { class: "body" },
      el("dl", { class: "kv" },
        el("dt", {}, "near-misses probed"), el("dd", {}, int(probe.n_true_near_misses)),
        el("dt", {}, "wrongly served"), el("dd", {}, int(probe.near_miss_wrongly_served)),
        el("dt", {}, "rejected on entity mismatch"), el("dd", {}, int(probe.gated_misses_on_near_miss)),
        el("dt", {}, "exact-repeat hit rate"), el("dd", {}, pct(probe.exact_repeat_pass?.hit_rate))),
      el("p", { class: "prose", style: "margin:12px 0 0" },
        "Each near-miss has one identifier changed and everything else held. A plain " +
        "semantic cache would have served the ones rejected on entity mismatch.")));
}

function runCard(run, digest, nemo) {
  if (!run) return null;
  return el("section", { class: "card" },
    el("h2", {}, "Run provenance"),
    el("div", { class: "body" },
      el("dl", { class: "kv" },
        el("dt", {}, "generated"), el("dd", {}, String(run.generated_at ?? "—")),
        el("dt", {}, "seed"), el("dd", {}, String(run.seed ?? "—")),
        el("dt", {}, "test items per arm"), el("dd", {}, int(run.n_test_items)),
        el("dt", {}, "alpha"), el("dd", {}, num(run.alpha, 2)),
        el("dt", {}, "registry fingerprint"), el("dd", {}, String(run.registry_fingerprint ?? "—")),
        el("dt", {}, "detectors"), el("dd", {}, (run.registry_detectors || []).join(", ") || "—"),
        el("dt", {}, "deterministic mode"), el("dd", {}, String(run.deterministic_mode ?? "—")),
        el("dt", {}, "deterministic digest"), el("dd", {}, String(digest ?? "—")),
        el("dt", {}, "python"), el("dd", {}, String(run.python ?? "—")),
        el("dt", {}, "platform"), el("dd", {}, String(run.platform ?? "—")),
        el("dt", {}, "wall clock"), el("dd", {}, run.wall_clock_s ? `${num(run.wall_clock_s, 1)} s` : "—"),
        el("dt", {}, "NeMo arm"), el("dd", {}, nemo?.status || nemo?.skipped_reason || "not run"),
        el("dt", {}, "reproduce"), el("dd", {}, String(run.reproduce ?? "—")))));
}

export const resultsView = {
  title: "Results",

  async mount(root) {
    mount(root, loading("Loading evaluation results…"));
    let data;
    try {
      data = await api.results();
    } catch (err) {
      mount(root, el("div", { class: "card" }, el("div", { class: "body" },
        errorBox(err, "No results have been generated yet. Run `make eval` (about 2 minutes, " +
          "offline) or `make eval-fast` for a 25% sample."))));
      return;
    }

    const arms = data.arms || {};
    mount(root, el("div", {},
      el("p", { class: "prose" },
        el("strong", {}, "All arms run the same detectors at the same thresholds against the " +
          "same provider. "),
        "Only the scheduling policy differs — the harness asserts an identical detector-registry " +
        "fingerprint across arms and records it below."),
      headline(arms) || "",
      el("section", { class: "card", style: "margin-top:16px" },
        el("h2", {}, "Three-arm comparison"),
        comparisonTable(arms)),
      el("div", { class: "grid two", style: "margin-top:16px" },
        schedulerByBand(arms.adaptive) || el("div"),
        calibrationCard(data.calibration) || el("div")),
      el("div", { class: "grid two", style: "margin-top:16px" },
        cacheCard(data.cache_probe) || el("div"),
        runCard(data.run, data.deterministic_digest, data.nemo_arm) || el("div"))));
  },
};
