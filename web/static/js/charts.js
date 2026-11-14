// Inline SVG charts. All inputs are numbers/labels from our own API; labels are escaped anyway.
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const W = 560, H = 300, M = { l: 48, r: 16, t: 14, b: 36 };

function scale(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range;
  return (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
}

function frame(xs, ys, xLabel, yLabel, xticks, yticks) {
  let s = "";
  yticks.forEach((t) => {
    const y = ys(t);
    s += `<line x1="${M.l}" x2="${W - M.r}" y1="${y}" y2="${y}" stroke="var(--grid)"/>` +
      `<text x="${M.l - 6}" y="${y + 4}" text-anchor="end">${esc(+t.toFixed(3))}</text>`;
  });
  xticks.forEach((t) => {
    s += `<text x="${xs(t)}" y="${H - M.b + 16}" text-anchor="middle">${esc(+t.toFixed(3))}</text>`;
  });
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(xLabel)}</text>` +
    `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text>`;
  return s;
}

const ticks = (a, b, n = 5) => Array.from({ length: n + 1 }, (_, i) => a + ((b - a) * i) / n);

// series: [{points:[[x,y]...], color, dashed, markers, label}]
export function lineChart({ series, xLabel, yLabel, aria, xDomain, yDomain, xTicks, extra = () => "" }) {
  const all = series.flatMap((s) => s.points);
  const xd = xDomain || [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))];
  const yd = yDomain || [Math.min(0, ...all.map((p) => p[1])), Math.max(...all.map((p) => p[1])) * 1.05];
  const xs = scale(xd, [M.l, W - M.r]), ys = scale(yd, [H - M.b, M.t]);
  let s = frame(xs, ys, xLabel, yLabel, xTicks || ticks(xd[0], xd[1]), ticks(yd[0], yd[1]));
  series.forEach((ser) => {
    const d = ser.points.map((p, i) => (i ? "L" : "M") + xs(p[0]).toFixed(1) + " " + ys(p[1]).toFixed(1)).join("");
    s += `<path d="${d}" fill="none" stroke="${ser.color || "var(--accent)"}" stroke-width="2.2"` +
      `${ser.dashed ? ' stroke-dasharray="5 4"' : ""}/>`;
    if (ser.markers) ser.points.forEach((p) => {
      s += `<circle cx="${xs(p[0])}" cy="${ys(p[1])}" r="3.5" fill="${ser.color || "var(--accent)"}"/>`;
    });
  });
  s += extra(xs, ys);
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(aria)}">${s}</svg>`;
}

export function rocChart(points, current) {
  const pts = points.map((p) => [p.fpr, p.tpr]);
  const extra = (xs, ys) => {
    const diag = `<line x1="${xs(0)}" y1="${ys(0)}" x2="${xs(1)}" y2="${ys(1)}" stroke="var(--neutral)" stroke-dasharray="4 4"/>`;
    const c = current ? `<circle cx="${xs(current.fpr)}" cy="${ys(current.recall)}" r="7" fill="var(--accent)" stroke="var(--card)" stroke-width="2"/>` : "";
    return diag + c;
  };
  return lineChart({
    series: [{ points: pts }], xLabel: "False-positive rate", yLabel: "True-positive rate (recall)",
    xDomain: [0, 1], yDomain: [0, 1], extra,
    aria: "ROC curve of the logistic model with the current threshold marked",
  });
}

// items: [{label, value}] signed horizontal bars (HTML, accessible)
export function bars(items, { fmtv = (v) => v.toFixed(4), signed = false } = {}) {
  const max = Math.max(...items.map((i) => Math.abs(i.value))) || 1;
  const rows = items.map((it) => {
    const pct = (Math.abs(it.value) / max) * (signed ? 50 : 100);
    const style = signed
      ? (it.value >= 0 ? `left:50%;width:${pct}%` : `left:${50 - pct}%;width:${pct}%`)
      : `left:0;width:${pct}%`;
    return `<div class="bar-row"><span>${esc(it.label)}</span><span class="bar-track"><span class="bar-fill${it.value < 0 ? " neg" : ""}" style="${style}"></span></span><span>${esc(fmtv(it.value))}</span></div>`;
  });
  return `<div role="img" aria-label="Bar chart">${rows.join("")}</div>`;
}
