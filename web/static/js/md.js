// Tiny, safe markdown subset renderer (bold, code, bullets, pipe tables, paragraphs). Builds DOM nodes only.
import { h } from "./api.js";

function inline(text) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    out.push(tok[0] === "`" ? h("code", {}, tok.slice(1, -1)) : h("strong", {}, tok.slice(2, -2)));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function markdown(src) {
  const root = h("div", { class: "prose" });
  const lines = src.split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (/^\s*-\s+/.test(line)) {
      const ul = h("ul");
      while (i < lines.length && (/^\s*-\s+/.test(lines[i]) || /^\s{2,}\S/.test(lines[i]))) {
        if (/^\s*-\s+/.test(lines[i])) {
          const nested = /^\s{2,}-/.test(lines[i]);
          const li = h("li", {}, inline(lines[i].replace(/^\s*-\s+/, "")));
          (nested && ul.lastChild ? ul.lastChild.appendChild(h("ul")).appendChild(li) : ul.appendChild(li));
        } else if (ul.lastChild) {
          ul.lastChild.append(" ", ...inline(lines[i].trim()));
        }
        i++;
      }
      root.append(ul);
    } else if (line.trim().startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
        i++;
      }
      const body = rows.filter((r, idx) => idx !== 1);
      root.append(h("div", { class: "scroll" }, h("table", {},
        h("thead", {}, h("tr", {}, body[0].map((c) => h("th", { scope: "col" }, c)))),
        h("tbody", {}, body.slice(1).map((r) => h("tr", {}, r.map((c) => h("td", {}, inline(c)))))))));
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^\s*-\s+/.test(lines[i]) && !lines[i].trim().startsWith("|")) {
        para.push(lines[i].trim()); i++;
      }
      root.append(h("p", {}, inline(para.join(" "))));
    }
  }
  return root;
}
