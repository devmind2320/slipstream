/* Dashboard shell: hash router, theme, header health indicator.
 *
 * Views are plain objects with {title, mount(root, ctx)}. Adding one means
 * adding a module under views/ and one entry to ROUTES — nothing else.
 */

import { api } from "./api.js";
import { el, $ } from "./util.js";
import { demoView } from "./views/demo.js";
import { resultsView } from "./views/results.js";
import { policyView } from "./views/policy.js";
import { ledgerView } from "./views/ledger.js";
import { systemView } from "./views/system.js";

const ROUTES = [
  ["demo", demoView],
  ["results", resultsView],
  ["policy", policyView],
  ["ledger", ledgerView],
  ["system", systemView],
];
const DEFAULT_ROUTE = "demo";

/* Shared context handed to every view. Kept deliberately small: cross-view
 * coupling is one callback list, not a state container. */
const recordListeners = new Set();
const ctx = {
  addRecord() {
    for (const fn of recordListeners) {
      try { fn(); } catch { /* a stale listener must not break the demo run */ }
    }
  },
  onRecordAppended(fn) {
    recordListeners.add(fn);
  },
};

function currentRoute() {
  const id = (location.hash || "").replace(/^#\/?/, "").split("?")[0];
  return ROUTES.some(([r]) => r === id) ? id : DEFAULT_ROUTE;
}

async function render() {
  const routeId = currentRoute();
  const [, view] = ROUTES.find(([r]) => r === routeId);

  for (const a of document.querySelectorAll("nav.tabs a")) {
    const active = a.dataset.route === routeId;
    if (active) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  document.title = `Slipstream — ${view.title}`;

  // A view owning its own listeners is fine; they are dropped with the subtree.
  // The record listeners are the exception, so clear them on every navigation.
  recordListeners.clear();

  const root = $("main.view");
  root.replaceChildren();
  try {
    await view.mount(root, ctx);
  } catch (err) {
    root.replaceChildren(el("div", { class: "card" },
      el("div", { class: "error" }, `This view failed to render: ${err.message}`)));
    // Surfacing the stack in the console beats swallowing it during a demo.
    console.error(err);
  }
}

function buildNav() {
  const nav = $("nav.tabs");
  nav.replaceChildren(...ROUTES.map(([id, view]) =>
    el("a", { href: `#/${id}`, "data-route": id }, view.title)));
}

/* ------------------------------------------------------------------ theme */
const THEME_KEY = "slipstream.theme";

function applyTheme(theme) {
  if (theme === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  const btn = $("#theme");
  if (btn) {
    btn.textContent = { system: "Auto", light: "Light", dark: "Dark" }[theme];
    btn.setAttribute("aria-label", `Theme: ${theme}. Click to change.`);
  }
}

function initTheme() {
  let theme = "system";
  try { theme = localStorage.getItem(THEME_KEY) || "system"; } catch { /* private mode */ }
  applyTheme(theme);
  $("#theme").addEventListener("click", () => {
    const order = ["system", "light", "dark"];
    let current = "system";
    try { current = localStorage.getItem(THEME_KEY) || "system"; } catch { /* ignore */ }
    const next = order[(order.indexOf(current) + 1) % order.length];
    try { localStorage.setItem(THEME_KEY, next); } catch { /* ignore */ }
    applyTheme(next);
  });
}

/* ----------------------------------------------------------------- health */
async function pollHealth() {
  const dot = $("#health-dot");
  const label = $("#health-label");
  try {
    const h = await api.health();
    dot.className = "dot ok";
    label.textContent = `${h.provider} · ${(h.detectors || []).length} detectors`;
    label.title = `registry ${h.registry_fingerprint}`;
  } catch {
    dot.className = "dot down";
    label.textContent = "gateway unreachable";
    label.title = "";
  }
}

/* ------------------------------------------------------------------- boot */
buildNav();
initTheme();
window.addEventListener("hashchange", render);
render();
pollHealth();
setInterval(pollHealth, 15000);
