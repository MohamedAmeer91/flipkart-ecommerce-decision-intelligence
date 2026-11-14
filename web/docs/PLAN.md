# Web layer plan

## Purpose
An interactive portfolio view of the Flipkart E-commerce capstone (churn / order-value analytics on synthetic data). The web layer only reads `outputs/` and the root README text, and imports `common.py` to obtain model probabilities on the project's pinned 560/240 split. Nothing in the graded code is modified.

## Pages (hash-routed, one HTML shell)
1. Overview - problem, headline numbers, key findings.
2. Part 1 - order value: estimate, hypothesis test, regression baseline, gradient-descent loss.
3. Part 2 - churn classifier: weights and odds ratios, loss curve, interactive threshold lab (confusion matrix, precision/recall/F1, ROC, INR cost), KNN K selector, score-a-record form.
4. Part 3 - segments: K-means profiles explorer, WCSS elbow, PCA scree and loadings, hierarchical merge tables.
5. Part 4 - model comparison: split gains, interactive ensemble voting, AdaBoost weights table, architecture sizing.
6. Model-choice memo.
7. Limitations.

## API endpoints (JSON)
- GET /api/health
- GET /api/overview
- GET /api/part1, /api/part2, /api/part3, /api/part4 (parsed outputs files and tables)
- GET /api/content (memo, limitations, findings text from the root README)
- GET /api/threshold?tau&fp_cost&fn_cost&dataset
- GET /api/curve?dataset (ROC points at 0.01 steps)
- GET /api/knn?k
- GET /api/segments
- GET /api/ensemble?tau&rule&models
- POST /api/score

## Data flow
outputs/*.txt, *.csv -> parser (read only) -> JSON. common.get_split/get_model/knn -> cached probabilities (computed lazily, warmed at startup) -> threshold, knn, ensemble, score endpoints. Frontend ES modules fetch JSON and draw inline SVG charts.

## Scope
In: everything above, validation, error handling, tests, Docker. Out: retraining from the UI, authentication, database, uploads, any build step, any external network resources.
