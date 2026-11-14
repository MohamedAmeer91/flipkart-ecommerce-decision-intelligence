// Thin fetch wrapper with readable errors.
export async function api(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch (e) {
    throw new Error("Cannot reach the server. Is it running?");
  }
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) {
    const detail = body && body.errors
      ? body.errors.map((x) => x.field + ": " + x.message).join("; ")
      : (body && body.detail) || res.statusText;
    throw new Error(detail);
  }
  return body;
}

// DOM helper: h("div", {class: "x"}, "text", childNode). Text is always set as text, never HTML.
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export const fmt = (v, d = 4) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(d)) : String(v));
export const inr = (v) => "INR " + Math.round(v).toLocaleString("en-IN");

export function debounce(fn, ms = 120) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export function table(caption, header, rows) {
  return h("div", { class: "scroll" }, h("table", {},
    h("caption", {}, caption),
    h("thead", {}, h("tr", {}, header.map((c) => h("th", { scope: "col" }, c)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c) => h("td", {}, typeof c === "number" ? fmt(c) : c)))))));
}
