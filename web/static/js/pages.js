import { api, h, fmt, inr, debounce, table } from "./api.js";
import { markdown } from "./md.js";
import { lineChart, rocChart, bars } from "./charts.js";

const html = (s) => { const d = h("div"); d.innerHTML = s; return d; };
const card = (title, ...kids) => h("section", { class: "card" }, title ? h("h2", {}, title) : null, ...kids);
const stat = (l, v, n) => h("div", { class: "stat" }, h("div", { class: "l" }, l), h("div", { class: "v" }, v), h("div", { class: "n" }, n || ""));
const kv = (cap, res, keys) => table(cap, ["Key", "Value"], keys.map((k) => [k, res[k]]));
const errBox = (e) => h("p", { class: "err", role: "alert" }, "Could not load: " + e.message);
const FEAT = ["tenure (months)", "complaints (90d)", "discount share", "avg order (hundreds INR)", "loyalty score", "COD user", "plus member"];

// ---------------------------------------------------------------- overview
export async function overview() {
  const d = await api("/api/overview");
  const root = h("div");
  root.append(h("h1", {}, d.title),
    h("p", { class: "lead" }, "An end-to-end analytics project: estimate order value, predict churn and price the alert threshold in rupees, find shopper segments, and compare models. Every number below is read from the project's committed outputs."),
    h("p", { class: "note" }, d.synthetic_notice),
    card("Three questions", h("ol", {}, d.questions.map((q) => h("li", {}, q)))),
    h("div", { class: "grid" }, d.headline.map((x) => stat(x.label, typeof x.value === "number" ? fmt(x.value, Number.isInteger(x.value) ? 0 : 4) : x.value, x.note))));
  for (const [k, v] of Object.entries(d.findings)) {
    const det = h("details", { class: "card" }, h("summary", {}, h("strong", {}, k)), markdown(v));
    root.append(det);
  }
  return root;
}

// ---------------------------------------------------------------- part 1
export async function part1() {
  const { results: r } = await api("/api/part1");
  const root = h("div", {}, h("h1", {}, "Part 1: order value"),
    h("p", { class: "lead" }, "Is the INR 800 planning average right, how big a sample is needed, and how far does a simple regression go?"));
  root.append(h("div", { class: "grid" },
    stat("Mean order value", fmt(r.mean_value), "95% CI " + fmt(r.ci95_low) + " to " + fmt(r.ci95_high)),
    stat("z statistic vs 8.0", fmt(r.z_stat), "reject H0: " + r.reject_h0),
    stat("Sample needed", r.n_required, "for a margin of INR 50"),
    stat("Regression R2 (X4)", fmt(r.r2), "RMSE " + fmt(r.rmse))));
  const gd = [0, 1, 10, 100, 500].map((s) => [s, r["gd_J" + s]]);
  root.append(h("div", { class: "grid2" },
    card("Gradient descent cost", html(lineChart({
      series: [{ points: gd.map(([s, v]) => [Math.log10(s + 1), v]), markers: true }],
      xLabel: "log10(updates + 1)", yLabel: "cost J", aria: "Cost J falls from " + fmt(r.gd_J0) + " to " + fmt(r.gd_J500) + " over 500 updates",
    })), h("p", { class: "muted small" }, "Gap between gradient-descent and closed-form slope: " + fmt(r.gd_slope_gap))),
    card("Residual spread (funnel check)", table("SD ratio, upper half / lower half of fitted values", ["Model", "Ratio"],
      [["Raw regression", r.resid_sd_ratio_raw], ["Log-log regression", r.resid_sd_ratio_log]]),
    h("p", { class: "muted small" }, "The log-log model removes the funnel; its R2 is " + fmt(r.loglog_r2) + "."))));
  root.append(card("Regression baseline", kv("Closed-form and multiple regression", r,
    ["reg_slope", "reg_intercept", "tss", "ess", "rss", "r2", "rmse", "mae", "loglog_r2", "mr_r2", "mr_adj_r2", "loglog_adj_r2"])));
  root.append(card("All Part 1 results", kv("outputs/part1_results.txt", r, Object.keys(r))));
  return root;
}

// ---------------------------------------------------------------- part 2
function thresholdLab(curve) {
  const out = h("div");
  const tau = h("input", { type: "range", min: 0, max: 1, step: 0.01, value: 0.15, id: "tau", "aria-describedby": "taulbl" });
  const tauv = h("strong", {}, "0.15");
  const fp = h("input", { type: "number", min: 0, value: 50, id: "fpc" });
  const fn = h("input", { type: "number", min: 0, value: 400, id: "fnc" });
  const ds = h("select", { id: "ds" }, h("option", { value: "test" }, "test rows (240)"), h("option", { value: "train" }, "training rows (560)"));
  const cmBox = h("div"), rocBox = h("div"), msg = h("div", { "aria-live": "polite" });
  async function update() {
    tauv.textContent = Number(tau.value).toFixed(2);
    try {
      const q = new URLSearchParams({ tau: tau.value, fp_cost: fp.value || 0, fn_cost: fn.value || 0, dataset: ds.value });
      const r = await api("/api/threshold?" + q);
      msg.replaceChildren();
      cmBox.replaceChildren(
        h("table", { class: "cm" }, h("caption", {}, "Confusion matrix at tau " + r.tau.toFixed(2)),
          h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", { scope: "col" }, "Predicted churn"), h("th", { scope: "col" }, "Predicted stay"))),
          h("tbody", {},
            h("tr", {}, h("th", { scope: "row" }, "Actually churned"), h("td", { class: "good" }, r.tp), h("td", { class: "bad" }, r.fn)),
            h("tr", {}, h("th", { scope: "row" }, "Actually stayed"), h("td", { class: "bad" }, r.fp), h("td", { class: "good" }, r.tn)))),
        h("div", { class: "grid", style: "margin-top:12px" },
          stat("Precision", fmt(r.precision)), stat("Recall", fmt(r.recall)), stat("F1", fmt(r.f1)), stat("Accuracy", fmt(r.accuracy))),
        h("div", { class: "grid", style: "margin-top:12px" },
          stat("INR cost", inr(r.cost), r.fp + " false alarms x " + r.fp_cost + " + " + r.fn + " misses x " + r.fn_cost),
          stat("Flag nobody", inr(r.cost_flag_nobody), "every churner missed"),
          stat("Cheapest tau (k/20)", r.cost_best_tau.toFixed(2), inr(r.cost_at_best_tau))));
      rocBox.replaceChildren(html(rocChart(curve.points, r)));
    } catch (e) { msg.replaceChildren(errBox(e)); }
  }
  const go = debounce(update, 80);
  [tau, fp, fn, ds].forEach((el) => el.addEventListener("input", go));
  ds.addEventListener("change", async () => { curve.points = (await api("/api/curve?dataset=" + ds.value)).points; update(); });
  out.append(h("div", { class: "grid2" },
    h("div", {}, h("label", { for: "tau", id: "taulbl" }, "Alert threshold tau: flag when probability is at least ", tauv), tau,
      h("div", { class: "form" }, h("div", {}, h("label", { for: "fpc" }, "Cost per false alarm (INR)"), fp),
        h("div", {}, h("label", { for: "fnc" }, "Cost per missed churner (INR)"), fn),
        h("div", {}, h("label", { for: "ds" }, "Evaluate on"), ds)), msg, cmBox),
    h("div", {}, rocBox)));
  update();
  return out;
}

function knnLab(cv) {
  const box = h("div"), out = h("div", { "aria-live": "polite" });
  const pills = h("div", { class: "pills", role: "group", "aria-label": "Choose K" });
  async function pick(k) {
    pills.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.k == k)));
    try {
      const r = await api("/api/knn?k=" + k);
      out.replaceChildren(h("div", { class: "grid" },
        stat("K", r.k, r.k === r.best_k ? "best by cross-validation" : "best K is " + r.best_k),
        stat("Cross-validated F1", fmt(r.cv_f1)), stat("Test F1", fmt(r.f1)), stat("Test recall", fmt(r.recall)), stat("Test precision", fmt(r.precision))),
        table("Test confusion cells for K = " + r.k, ["tp", "fp", "tn", "fn"], [[r.tp, r.fp, r.tn, r.fn]]));
    } catch (e) { out.replaceChildren(errBox(e)); }
  }
  for (let k = 1; k <= 21; k += 2) pills.append(h("button", { type: "button", "data-k": k, "aria-pressed": "false", onclick: () => pick(k) }, "K=" + k));
  box.append(pills, out, html(lineChart({
    series: [{ points: cv.map((c) => [c.K, c.mean_cv_f1]), markers: true }], xLabel: "K", yLabel: "mean 5-fold CV F1", yDomain: [0, 0.6],
    xTicks: cv.map((c) => c.K), aria: "Cross-validated F1 for each K",
  })));
  pick(5);
  return box;
}

function scoreForm() {
  const fields = [
    ["tenure_months", "Tenure (months)", 12, { min: 1, max: 120, step: 0.1 }],
    ["complaints_90d", "Complaints, last 90 days", 5, { min: 0, max: 30, step: 1 }],
    ["discount_share", "Discount order share (0 to 1)", 0.7, { min: 0, max: 1, step: 0.01 }],
    ["avg_order_value", "Average past order (hundreds INR)", 15, { min: 0.5, max: 200, step: 0.1 }],
    ["loyalty_score", "Loyalty score (0 to 100)", 40, { min: 0, max: 100, step: 0.1 }],
  ];
  const inputs = {};
  const form = h("form", { class: "form", novalidate: true });
  fields.forEach(([k, l, v, a]) => {
    inputs[k] = h("input", { type: "number", id: "f_" + k, value: v, ...a, required: true });
    form.append(h("div", {}, h("label", { for: "f_" + k }, l), inputs[k]));
  });
  [["cod_flag", "Uses cash on delivery"], ["plus_member_flag", "Plus member"]].forEach(([k, l]) => {
    inputs[k] = h("select", { id: "f_" + k }, h("option", { value: 0 }, "No"), h("option", { value: 1 }, "Yes"));
    form.append(h("div", {}, h("label", { for: "f_" + k }, l), inputs[k]));
  });
  const res = h("div", { "aria-live": "polite" });
  form.append(h("div", {}, h("button", { type: "submit" }, "Score record")));
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const body = {};
    for (const k of Object.keys(inputs)) body[k] = Number(inputs[k].value);
    try {
      const r = await api("/api/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      res.replaceChildren(h("div", { class: "grid" },
        stat("Churn probability", (r.probability * 100).toFixed(1) + "%", "risk band: " + r.risk_band),
        stat("Coupon at tau " + r.tau, r.flag_at_tau ? "Send" : "Hold", "tau 0.15 is the cost-optimal threshold"),
        stat("Nearest segment", r.nearest_segment, r.nearest_segment_label)),
        h("h3", {}, "What drives this score (weight x standardised value, log-odds)"),
        html(bars(r.contributions.map((c, i) => ({ label: FEAT[i], value: c.effect })), { signed: true, fmtv: (v) => v.toFixed(3) })),
        h("p", { class: "muted small" }, "Baseline (bias) " + r.bias.toFixed(3) + " log-odds. Positive values push toward churn."));
    } catch (e) { res.replaceChildren(errBox(e)); }
  });
  return h("div", {}, form, res);
}

export async function part2() {
  const [d, curve] = await Promise.all([api("/api/part2"), api("/api/curve?dataset=test")]);
  const r = d.results;
  const root = h("div", {}, h("h1", {}, "Part 2: churn prediction"),
    h("p", { class: "lead" }, "A logistic regression trained from scratch by gradient descent. The threshold decides who gets a retention coupon, and money decides the threshold."));
  root.append(h("div", { class: "grid" },
    stat("Test cost at tau " + fmt(r.cost_tau, 2), inr(r.cost_test_at_tau), "chosen on training rows"),
    stat("Test cost at tau 0.5", inr(r["cost_test_at_0.5"])), stat("Flag nobody", inr(r.cost_flag_nobody ?? r.cost_test_flag_nobody)),
    stat("Blind accuracy", fmt(r.blind_accuracy), "model at 0.5: " + fmt(r["tau_0.5_accuracy"]))));
  root.append(card("Threshold lab", h("p", { class: "muted" }, "Drag the threshold. The confusion matrix, rates, ROC position and INR cost update from the trained model's probabilities."), thresholdLab(curve)));
  root.append(h("div", { class: "grid2" },
    card("Weights (standardised features)", html(bars(FEAT.map((n, i) => ({ label: n, value: r["weight_X" + (i + 1)] })), { signed: true }))),
    card("Odds ratios per 1 SD", html(bars(FEAT.map((n, i) => ({ label: n, value: r["odds_ratio_X" + (i + 1)] })))))));
  root.append(card("Training loss", html(lineChart({
    series: [{ points: [0, 1, 10, 100, 1000].map((e) => [Math.log10(e + 1), r["loss_epoch_" + e]]), markers: true }],
    xLabel: "log10(epochs + 1)", yLabel: "cross-entropy", yDomain: [0.3, 0.7], aria: "Loss falls from 0.6931 to 0.3704",
  }))));
  root.append(card("K-nearest neighbours: choose K", knnLab(d.knn_cv)));
  root.append(card("Score a record", h("p", { class: "muted" }, "Enter customer features. The trained logistic weights return a churn probability, validated server-side."), scoreForm()));
  root.append(card("Threshold sweep (committed file)", table("outputs/part2_thresholds.csv, test rows", Object.keys(d.thresholds[0]), d.thresholds.map((x) => Object.values(x)))));
  return root;
}

// ---------------------------------------------------------------- part 3
export async function part3() {
  const [d, s] = await Promise.all([api("/api/part3"), api("/api/segments")]);
  const r = d.results;
  const root = h("div", {}, h("h1", {}, "Part 3: shopper segments"),
    h("p", { class: "lead" }, "K-means on five standardised features finds three groups. Pick a segment to compare it with the whole base."));
  const sel = h("div", { class: "pills", role: "group", "aria-label": "Segment" });
  const view = h("div", { "aria-live": "polite" });
  const metrics = [["mean_X1", "Tenure (months)"], ["mean_X2", "Complaints"], ["mean_X3", "Discount share"], ["mean_X4", "Avg order (hundreds)"],
    ["mean_X5", "Loyalty score"], ["outcome_rate", "Churn rate"], ["mean_value", "Mean order value"]];
  function show(seg) {
    sel.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.s === seg.segment)));
    view.replaceChildren(h("h3", {}, "Segment " + seg.segment + ": " + seg.label + " (" + seg.size + " customers, " + (100 * seg.size / s.overall.size).toFixed(0) + "% of base)"),
      table("Segment vs overall average", ["Metric", "Segment", "Overall", "Ratio"],
        metrics.map(([k, l]) => [l, seg[k], s.overall[k], (seg[k] / s.overall[k]).toFixed(2) + "x"])));
  }
  s.segments.forEach((seg) => sel.append(h("button", { type: "button", "data-s": seg.segment, "aria-pressed": "false", onclick: () => show(seg) }, seg.segment + " " + seg.label)));
  root.append(card("Segment explorer", sel, view),
    card("Churn rate by segment", html(bars(s.segments.map((x) => ({ label: "Segment " + x.segment, value: x.outcome_rate }))))));
  show(s.segments[2]);
  const w = [2, 3, 4, 5, 6].map((k) => [k, r["wcss_k" + k]]);
  root.append(h("div", { class: "grid2" },
    card("Elbow: within-cluster sum of squares", html(lineChart({ series: [{ points: w, markers: true }], xLabel: "K", yLabel: "WCSS", xTicks: [2, 3, 4, 5, 6], yDomain: [0, 2400], aria: "WCSS by K with elbow at 3" })),
      h("p", { class: "muted small" }, "Elbow K = " + r.elbow_k)),
    card("PCA explained variance", html(bars([1, 2, 3, 4, 5].map((i) => ({ label: "PC" + i, value: r["explained_ratio_" + i] })))),
      h("p", { class: "muted small" }, "Components for 80% of variance: " + r.components_for_80pct + ". Cumulative at 3: " + fmt(r.cumulative_ratio_3)))));
  root.append(card("PC1 and PC2 loadings", table("Loadings on standardised X1 to X5", ["Feature", "PC1", "PC2"],
    ["X1 tenure", "X2 complaints", "X3 discount share", "X4 avg order", "X5 loyalty"].map((n, i) => [n, r["pc1_loading_X" + (i + 1)], r["pc2_loading_X" + (i + 1)]]))));
  const link = {};
  d.hierarchical.forEach((x) => { (link[x.linkage] = link[x.linkage] || []).push([x.merge_no, x.cluster_a, x.cluster_b, x.height]); });
  root.append(card("Hierarchical clustering of the six-cohort table", h("div", { class: "grid" },
    Object.entries(link).map(([k, rows]) => table(k + " linkage", ["Merge", "A", "B", "Height"], rows))),
    h("p", { class: "muted small" }, "Clusters at the cut: single " + r.single_clusters_at_cut + ", complete " + r.complete_clusters_at_cut + ", average " + r.average_clusters_at_cut + ".")));
  return root;
}

// ---------------------------------------------------------------- part 4
function ensembleLab() {
  const tau = h("input", { type: "range", min: 0.05, max: 0.95, step: 0.05, value: 0.3, id: "etau" });
  const tv = h("strong", {}, "0.30");
  const rule = h("select", { id: "erule" }, h("option", { value: "hard" }, "Hard vote (majority of members)"), h("option", { value: "soft" }, "Soft vote (average probability)"));
  const names = { logistic: "Logistic", knn: "KNN (K=5)", stump: "Stump (tenure <= 24)" };
  const checks = Object.entries(names).map(([k, l]) => h("label", {}, h("input", { type: "checkbox", value: k, checked: true }), l));
  const out = h("div", { "aria-live": "polite" });
  async function update() {
    tv.textContent = Number(tau.value).toFixed(2);
    const models = checks.map((c) => c.firstChild).filter((c) => c.checked).map((c) => c.value);
    try {
      if (!models.length) throw new Error("Select at least one model.");
      const q = new URLSearchParams({ tau: tau.value, rule: rule.value });
      models.forEach((m) => q.append("models", m));
      const r = await api("/api/ensemble?" + q);
      out.replaceChildren(table("Test rows (240), members: " + r.members.join(", "), ["Model", "tp", "fp", "tn", "fn", "Precision", "Recall", "F1"],
        r.rows.map((x) => [x.model, x.tp, x.fp, x.tn, x.fn, x.precision, x.recall, x.f1])));
    } catch (e) { out.replaceChildren(errBox(e)); }
  }
  const go = debounce(update, 80);
  [tau, rule, ...checks].forEach((el) => el.addEventListener("input", go));
  update();
  return h("div", {}, h("label", { for: "etau" }, "Threshold tau ", tv), tau, h("div", { class: "form" }, h("div", {}, h("label", { for: "erule" }, "Voting rule"), rule)),
    h("div", { class: "checks", role: "group", "aria-label": "Ensemble members" }, checks), out);
}

export async function part4() {
  const d = await api("/api/part4");
  const r = d.results;
  const root = h("div", {}, h("h1", {}, "Part 4: model choice and sizing"),
    h("p", { class: "lead" }, "Which split a tree would use, whether combining models helps, how boosting reweights mistakes, and how large later image or sequence models would be."));
  root.append(card("Ensemble voting lab", h("p", { class: "muted" }, "Combine the logistic model, KNN and the one-split stump. At tau 0.3 with all three this reproduces the committed voting table."), ensembleLab()));
  root.append(h("div", { class: "grid2" },
    card("Candidate splits", table("Information and Gini gain (best: " + r.best_split_by_info_gain + ")", Object.keys(d.splits[0]), d.splits.map((x) => Object.values(x))),
      h("p", { class: "muted small" }, "Root entropy " + fmt(r.root_entropy) + ", root Gini " + fmt(r.root_gini) + ". Stump churn rate: " + fmt(r.stump_rate_yes) + " (yes) vs " + fmt(r.stump_rate_no) + " (no).")),
    card("Committed voting results", table("outputs/part4_voting.csv", Object.keys(d.voting[0]), d.voting.map((x) => Object.values(x))))));
  const rounds = [1, 2].map((n) => d.adaboost.filter((x) => x.round === n));
  root.append(card("AdaBoost by hand", h("p", { class: "muted" }, "Round 1 error " + fmt(r.round1_error) + ", alpha " + fmt(r.round1_alpha) + "; round 2 error " + fmt(r.round2_error) + ", alpha " + fmt(r.round2_alpha) + "; ensemble accuracy " + fmt(r.ensemble_accuracy) + "."),
    h("div", { class: "grid2" }, rounds.map((rows, i) => table("Round " + (i + 1) + " weights", ["Point", "Before", "Unnormalised", "After"], rows.map((x) => [x.point, x.weight_before, x.weight_unnormalised, x.weight_after]))))));
  root.append(card("Architecture sizing (parameters)", html(bars(d.architecture.map((x) => ({ label: x.layer, value: x.parameters })), { fmtv: (v) => v.toLocaleString("en-IN") })),
    table("Totals", ["Quantity", "Parameters"], ["mlp_total", "cnn_total", "rnn_total", "fc_layer_params", "conv_layer_params"].map((k) => [k, r[k].toLocaleString("en-IN")])),
    h("p", { class: "muted small" }, "A fully connected layer on a 128x128x3 image needs " + r.fc_layer_params.toLocaleString("en-IN") + " weights; the 64-filter 3x3 convolution needs " + r.conv_layer_params + ". Out-of-bag fraction " + fmt(r.oob_fraction) + ".")));
  return root;
}

// ---------------------------------------------------------------- text pages
export async function memo() {
  const c = await api("/api/content");
  return h("div", {}, h("h1", {}, "Model-choice memo"), card(null, markdown(c.memo)));
}
export async function limitations() {
  const c = await api("/api/content");
  return h("div", {}, h("h1", {}, "Limitations"), card(null, markdown(c.limitations)));
}
