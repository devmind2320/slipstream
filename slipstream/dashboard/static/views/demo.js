/* Live A/B demo: the same request through two scheduling policies.
 *
 * Both panes run the same detectors at the same thresholds against the same
 * provider. Only *when* verification happens relative to token generation
 * differs. That is the whole claim, so the view is careful not to imply
 * anything else changed.
 */

import { api } from "../api.js";
import { el, esc, mount, ms, errorBox } from "../util.js";

const ARMS = [
  { id: "fixed_async", label: "fixed_async — stream first, scan behind", cls: "async" },
  { id: "adaptive", label: "adaptive — hold back, verify concurrently, release", cls: "adaptive" },
];

/* Identifiers that should have been masked. Highlighting them red makes an
 * un-prevented leak visible rather than something the viewer has to spot. */
const RAW_IDENTIFIER =
  /(\b[A-Z]{5}\d{4}[A-Z]\b|\b\d{12}\b|\b[\w.+-]+@[\w-]+\.\w{2,}\b|INR\s[\d,]+|\b9\d{9}\b)/g;

function renderStream(node, text, streaming) {
  let html = esc(text).replace(/\[([A-Z_]{2,})\]/g, '<span class="mask">$1 masked</span>');
  html = html.replace(RAW_IDENTIFIER, '<span class="raw">$1</span>');
  node.innerHTML = html + (streaming ? '<span class="caret">▍</span>' : "");
}

function paneMarkup(arm) {
  return el("section", { class: `card ${arm.cls}`, id: `pane-${arm.id}` },
    el("h2", {}, arm.label),
    el("div", { class: "meta", id: `meta-${arm.id}` }, el("span", {}, "waiting…")),
    el("div", { class: "counters" },
      counter("TTFT", `ttft-${arm.id}`),
      counter("Added latency", `add-${arm.id}`),
      counter("Guard overhead", `guard-${arm.id}`)),
    el("div", { class: "out", id: `out-${arm.id}` }),
    el("div", { id: `ban-${arm.id}` }));
}

function counter(label, id) {
  return el("div", { class: "counter" },
    el("div", { class: "k" }, label),
    el("div", { class: "v pending", id }, "—"));
}

/** Run one arm to completion. Resolves when the record arrives or the stream dies. */
function runArm(armId, scenario, onRecord) {
  return new Promise((resolve) => {
    const out = document.getElementById(`out-${armId}`);
    const banHost = document.getElementById(`ban-${armId}`);
    const meta = document.getElementById(`meta-${armId}`);
    const setCounter = (k, v) => {
      const n = document.getElementById(`${k}-${armId}`);
      n.textContent = v;
      n.classList.toggle("pending", v === "—");
    };

    out.textContent = "";
    banHost.replaceChildren();
    meta.replaceChildren(el("span", {}, "connecting…"));
    ["ttft", "add", "guard"].forEach((k) => setCounter(k, "—"));

    const banner = (cls, text) =>
      banHost.replaceChildren(el("div", { class: `banner ${cls}` }, text));

    let buf = "";
    let displacedText = null;
    const t0 = performance.now();
    let sawFirstChunk = false;
    let settled = false;
    const es = new EventSource(api.demoStreamURL(armId, scenario));

    const finish = () => {
      if (settled) return;
      settled = true;
      es.close();
      resolve();
    };

    es.addEventListener("chunk", (e) => {
      const d = JSON.parse(e.data);
      if (!sawFirstChunk) {
        sawFirstChunk = true;
        setCounter("ttft", ms(performance.now() - t0));
      }
      buf += d.delta;
      renderStream(out, buf, true);
    });

    es.addEventListener("annotation", (e) => {
      const d = JSON.parse(e.data);
      if (d.meta && d.meta.restart) {
        // A regeneration replaces the response. What was already on screen was
        // still shown to a user, so keep it and label it rather than pretending
        // the display can be taken back.
        displacedText = buf;
        buf = "";
        banner("info", `Regenerated under a grounding constraint after ${ms(performance.now() - t0)}. ` +
          "The text above had already been displayed.");
      } else {
        banner("info", d.text);
      }
    });

    for (const kind of ["blocked", "escalated"]) {
      es.addEventListener(kind, (e) => {
        const d = JSON.parse(e.data);
        const reasons = (d.meta && d.meta.reasons) || [];
        banner("leak", `${kind === "blocked" ? "BLOCKED" : "ESCALATED"} — ${d.text || "withheld"}` +
          (reasons.length ? ` · ${reasons.join("; ")}` : ""));
      });
    }

    es.addEventListener("error", (e) => {
      // A server-sent `error` event carries a payload; a transport failure does not.
      try {
        const d = JSON.parse(e.data);
        banner("leak", `Stream error — ${d.message}`);
      } catch {
        if (!settled) banner("leak", "Stream closed unexpectedly. Is the gateway still running?");
      }
      finish();
    });

    es.addEventListener("record", (e) => {
      const rec = JSON.parse(e.data);
      const p = rec.policy;
      meta.innerHTML = [
        `band <b>${esc(p.band)}</b>`,
        `holdback <b>${esc(p.holdback_tokens)}</b> tok`,
        `sync <b>${esc(p.sync)}</b>`,
        `action <b>${esc(rec.action)}</b>`,
        `model <b>${esc(rec.model_used)}</b>`,
        `detectors <b>${esc((p.detectors || []).join(","))}</b>`,
      ].join("");

      setCounter("ttft", ms(rec.ttft_ms));
      setCounter("add", ms(rec.total_latency_ms));
      setCounter("guard", ms(rec.guard_overhead_ms));

      if (rec.leaked_chunks > 0) {
        banner("leak", `LEAKED — ${rec.leaked_chunks} chunk(s) reached the screen before their ` +
          "verdict existed. Flagged after the fact.");
        if (displacedText !== null) renderStream(out, displacedText, false);
        else renderStream(out, buf, false);
      } else {
        renderStream(out, buf, false);
        if (rec.entities_masked && rec.entities_masked.length) {
          banner("safe", `MASKED BEFORE EMISSION — ${rec.entities_masked.join(", ")} ` +
            "never reached the screen.");
        }
      }
      onRecord(rec);
      finish();
    });

    es.onerror = () => {
      if (!settled) {
        renderStream(out, buf, false);
        banner("leak", "Stream closed unexpectedly. Is the gateway still running?");
      }
      finish();
    };
  });
}

export const demoView = {
  title: "Live demo",

  async mount(root, ctx) {
    const select = el("select", { id: "scenario", "aria-label": "Scenario" });
    const runBtn = el("button", { class: "primary" }, "Run both arms");
    const status = el("span", { class: "note" }, "loading scenarios…");
    const prompt = el("p", { class: "prose" });

    const panes = el("div", { class: "grid two" }, ARMS.map(paneMarkup));

    mount(root, el("div", {},
      el("p", { class: "prose" },
        el("strong", {}, "Both panes run the same detectors at the same thresholds against the " +
          "same provider. "),
        "The only difference is when verification happens relative to token generation. Left " +
        "streams immediately and scans behind it; right holds a buffer whose depth the risk " +
        "estimator chose for this specific request."),
      el("div", { class: "toolbar" },
        el("label", { class: "note", for: "scenario" }, "Scenario"),
        select, runBtn, status),
      prompt,
      panes));

    let scenarios = {};
    try {
      scenarios = await api.scenarios();
    } catch (err) {
      mount(root, errorBox(err));
      return;
    }

    for (const [key, sc] of Object.entries(scenarios)) {
      select.append(el("option", { value: key }, sc.title));
    }
    const showPrompt = () => {
      const sc = scenarios[select.value];
      prompt.replaceChildren(
        el("strong", {}, `${sc.use_case} · `),
        document.createTextNode(`“${sc.prompt}”`));
    };
    select.addEventListener("change", showPrompt);
    showPrompt();
    status.textContent = "ready — offline, no API key";

    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true;
      select.disabled = true;
      status.textContent = "running both arms…";
      // Sequential, not parallel: the two panes share one event loop and one
      // CPU, and racing them would make the latency counters measure contention
      // rather than scheduling policy.
      for (const arm of ARMS) {
        await runArm(arm.id, select.value, (rec) => ctx.addRecord(rec));
      }
      status.textContent = "done — see the Ledger tab for the decision records";
      runBtn.disabled = false;
      select.disabled = false;
    });
  },
};
