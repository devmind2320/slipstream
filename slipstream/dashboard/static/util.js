/* Shared DOM and formatting helpers.
 *
 * Rule for this file: nothing here knows about Slipstream domain concepts. It
 * is escaping, number formatting and element construction only.
 */

/** Escape text for interpolation into innerHTML. Used on every dynamic string. */
export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

/** Build an element: el("div", {class: "card"}, child, "text"). */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export const $ = (sel, root = document) => root.querySelector(sel);

/** Replace a container's contents with one node. */
export function mount(container, node) {
  container.replaceChildren(node);
  return node;
}

/* ------------------------------------------------------------ formatting */

export function ms(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${Math.round(v)} ms`;
}

export function pct(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function num(v, digits = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

export function int(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString("en");
}

export function usd(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `$${Number(v).toFixed(4)}`;
}

export function shortHash(h, keep = 12) {
  if (!h) return "—";
  return h.length <= keep * 2 ? h : `${h.slice(0, keep)}…${h.slice(-keep)}`;
}

export function when(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}

/* --------------------------------------------------------- state helpers */

export const loading = (what = "Loading…") => el("div", { class: "loading" }, what);

export const empty = (msg, hint) =>
  el("div", { class: "empty" }, msg, hint ? el("div", { style: "margin-top:8px" }, hint) : null);

/** Render an ApiError consistently, with the hint that usually fixes it. */
export function errorBox(err, hint) {
  const box = el("div", { class: "error" }, err.status === 404 ? (err.detail || err.message) : err.message);
  if (err.status === 0) {
    box.append(el("div", { style: "margin-top:8px", class: "note" },
      "Is the gateway running? Start it with `make serve`."));
  } else if (hint) {
    box.append(el("div", { style: "margin-top:8px", class: "note" }, hint));
  }
  return box;
}

/** A table from a header row and an array of row-arrays. Cells may be nodes. */
export function table(headers, rows, opts = {}) {
  const thead = el("thead", {}, el("tr", {}, headers.map((h, i) =>
    el("th", { class: i === 0 ? "metric" : "" }, h))));
  const tbody = el("tbody", {}, rows.map((cells) =>
    el("tr", {}, cells.map((c, i) => {
      const cls = i === 0 ? "metric" : (opts.numeric === false ? "" : "num");
      return c instanceof Node
        ? el("td", { class: cls }, c)
        : el("td", { class: cls }, c);
    }))));
  return el("div", { class: "table-wrap" }, el("table", {}, thead, tbody));
}
