# Web layer design

## Module layout
- `web/app.py` - FastAPI app, routes, error handlers, static mount.
- `web/engine.py` - cached compute: split, logistic model, KNN shares, stump, voting, cost rule.
- `web/outputs_io.py` - read-only parsers for `outputs/*.txt`, `outputs/*.csv`, README sections.
- `web/schemas.py` - pydantic request/response validation.
- `web/static/` - `index.html`, `css/style.css`, `js/{main,api,charts,md,pages}.js`.
- `web/tests/` - unittest suites with FastAPI TestClient.

## API schema (abridged)
- `/api/threshold` -> `{tau, dataset, fp_cost, fn_cost, n, positives, tp, fp, tn, fn, accuracy, precision, recall, f1, specificity, fpr, cost, cost_flag_nobody, cost_best_tau}`
- `/api/curve` -> `{dataset, points:[{tau, tpr, fpr}]}`
- `/api/knn` -> `{k, cv_f1, tp, fp, tn, fn, accuracy, precision, recall, f1, best_k, cv:[{k, f1}]}`
- `/api/segments` -> `{segments:[{segment, label, size, mean_X1.., outcome_rate, mean_value}], overall:{...}}`
- `/api/ensemble` -> `{tau, rule, models:[...], rows:[{model, tp, fp, tn, fn, precision, recall, f1}]}`
- `/api/score` body `{tenure_months, complaints_90d, discount_share, avg_order_value, loyalty_score, cod_flag, plus_member_flag}` -> `{probability, flag_at_tau, tau, risk_band, contributions:[{feature, z, weight, effect}], nearest_segment}`
- Errors: `{"detail": ...}` with 422 for validation, 404 unknown route, 500 generic message.

## Frontend structure
Single `index.html` shell with a nav; `main.js` is the hash router; `pages.js` has one render function per page; `charts.js` draws inline SVG (line, bar, ROC); `md.js` is a tiny escaping markdown renderer for README text; `api.js` wraps fetch with error handling.

## Visual design
Neutral background, one accent (indigo), card layout, system font stack with a tight type scale, 16px gutters, responsive grid. Colors are CSS variables on `:root`, overridden under `prefers-color-scheme: dark`. Text contrast meets WCAG AA. Charts use the accent plus a neutral grey; no other hues except a muted red for cost/false-alarm emphasis. Focus rings visible, sliders labelled, tables with captions, charts with text summaries (`role=img` plus aria-label).
