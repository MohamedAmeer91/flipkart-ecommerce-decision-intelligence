# DESIGN: Flipkart E-commerce Decision Intelligence

Source of truth: `docs/REQUIREMENTS.md` (brief) and `docs/PLAN.md`. This note fixes layout, module
signatures, per-Part algorithms, exact output keys and headers, README/memo outline, tests, deploy checklist.

## 1. Repository layout (exactly per the brief, plus docs/ and tests/ and common.py)

```
README.md  requirements.txt (numpy==1.26.4)  generate_data.py (verbatim)  common.py (shared helpers)
data/records.csv
part1.py part2.py part3.py part4.py
outputs/ part1..4_results.txt, part2_thresholds.csv, part2_knn_cv.csv, part3_segments.csv,
         part3_hierarchical.csv, part4_splits.csv, part4_voting.csv, part4_adaboost.csv, part4_architecture.csv
docs/ (PLAN, REQUIREMENTS, DESIGN, DECISIONS)   tests/ (unittest)
.gitattributes  .gitignore
```
`docs/` and `tests/` are extra to the brief's tree; the brief does not forbid extra files ("shared helper files are fine",
no "only these files" rule), so they stay tracked (decision recorded in DECISIONS.md).

## 2. Data file and line endings
`generate_data.py` writes CRLF (csv default, `newline=""`); the MD5 `c6f249ba...` is over those bytes. So
`data/records.csv` and `generate_data.py` are marked `-text` in `.gitattributes` (no conversion, ever); verified with
`git cat-file blob HEAD:data/records.csv | md5sum`. Output files are written by our code with `newline="\n"` and are LF.

## 3. Shared helper `common.py` (import-safe: no top-level work)
- `load_data(path="data/records.csv") -> (X: float64 [800,7], outcome: int [800], value: float [800], header)`; stdlib csv.
- `split_stratified(y) -> (train_idx, test_idx)`: per class in file order, first `7*count//10` to train; each index list sorted.
- `standardise(Xtr, Xte) -> (Ztr, Zte, mean, sd)`: train mean, sample SD (ddof via explicit n-1 arithmetic).
- `sigmoid`, `bce(y, p)`, `fit_logistic(Z, y, eta, steps, record=()) -> (w, b, losses dict)` (zero start, full batch).
- `predict_proba(Z, w, b)`.
- `confusion(y, pred) -> (tp, fp, tn, fn)`, `rates(tp,fp,tn,fn) -> dict` (accuracy, precision, recall, f1, specificity, fpr; 0 on zero denominator).
- `knn_predict(Zref, yref, Zq, K) -> (shares, preds)`: squared Euclidean distances, `argsort(kind="stable")`, first K, positive if count >= (K+1)/2.
- `fmt(v)`: float -> `%.4f`, int/np.integer -> `%d`, bool -> `yes`/`no`, str as is. `write_results(path, items)` (list of (key, value), LF), `write_csv(path, header, rows)`.
- `get_split_and_model()` helper used by Parts 2 and 4: returns split, standardised matrices and the trained logistic model (deterministic recompute; Part 4 reuses Part 2 code, not typed numbers).
- `mean`, `sample_sd` helpers with plain numpy sums (no np.cov/polyfit/lstsq/corrcoef).

## 4. Part algorithms and exact outputs

### Part 1 (all 800 rows; X4 = col 3, value = col 9)
Keys in order: `n, mean_value, sd_value, se_value, ci95_low, ci95_high, z_stat, reject_h0, n_required, reg_slope, reg_intercept, tss, ess, rss, tss_equals_ess_plus_rss, r2, rmse, mae, gd_J0, gd_J1, gd_J10, gd_J100, gd_J500, gd_slope, gd_intercept, gd_slope_gap, resid_sd_lower_raw, resid_sd_upper_raw, resid_sd_ratio_raw, loglog_intercept, loglog_slope, loglog_r2, loglog_rmse, resid_sd_lower_log, resid_sd_upper_log, resid_sd_ratio_log, mr_beta_0..mr_beta_7, mr_r2, mr_adj_r2, loglog_adj_r2`.
- `n_required = ceil((1.96 sd/0.5)^2)` (smallest integer >= value).
- Closed form via explicit sums. GD on z-scored X4, eta 0.1, 500 steps, J recorded at 0,1,10,100,500; convert back.
- Funnel: `order = argsort(fitted, kind="stable")`; lower = first 400, upper = last 400; sample SD of residuals per group (residuals of that model, raw or log).
- Multiple regression: design [1, ln X4, X1, X2, X3, X5, X6, X7], `np.linalg.solve(XtX, XtY)`; adj R2 with p=7; loglog adj R2 with p=1.

### Part 2 (keys in order)
`train_rows, train_positives, test_rows, test_positives, loss_epoch_0, _1, _10, _100, _1000, weight_X1..X7, bias, test_bce, loss_epoch_200_eta_0.01, _0.1, _1.0, odds_ratio_X1..X7, tau_0.5_{tp,fp,tn,fn,accuracy,precision,recall,f1,specificity}, tau_0.3_{...}, blind_accuracy, knn_best_k, knn_tp, knn_fp, knn_tn, knn_fn, knn_accuracy, knn_precision, knn_recall, knn_f1, cost_tau, cost_train_at_tau, cost_test_at_tau, cost_test_at_0.5, cost_test_flag_nobody`.
- Standardise on train; logistic as specified (1000 steps eta 0.1). Learning-rate runs: 200 steps from zero.
- Thresholds CSV `tau,tp,fp,tn,fn,tpr,fpr`, tau 0.1..0.9 (written as 0.1000 ...; tau computed as k/10 for k=1..9).
- KNN CV: fold = train position % 5; reference = other folds; F1 per fold (0 if undefined); mean over folds; K in 1,3,..,21; header `K,mean_cv_f1` (K printed as integer). Best K = max mean F1, ties to smaller K (strict > comparison in ascending order).
- Cost: tau = k/20, k=1..19, training cost 50 fp + 400 fn, argmin with ties to smaller tau (strict <). `cost_tau` printed 4 dp (e.g. 0.2500). Test costs at chosen tau, 0.5, and flag-nobody = 400 * test positives. Costs are integers (printed as integers).
- Predict positive when `p >= tau`; to avoid floating issues with tau = k/20 use `p >= tau` with tau computed `k/20` as float (as specified).

### Part 3 (all 800, X1..X5 = columns 0..4 of X)
Keys in order: `wcss_k2..wcss_k6, elbow_k, size_segment_A/B/C, single_clusters_at_cut, complete_clusters_at_cut, average_clusters_at_cut, eigenvalue_sum, eigenvalue_1..5, explained_ratio_1..5, cumulative_ratio_1..5, components_for_80pct, pc1_loading_X1..5, pc2_loading_X1..5, row1_pc1, row1_pc2, row2_pc1, row2_pc2, row3_pc1, row3_pc2, mean_pc1_segment_A/B/C`.
- K-means: init rows `i * (800//K)`; assign by squared distance with `argmin` (first minimum = lower index); update means (empty cluster keeps centroid); stop when an assignment pass changes no row; pass count logged to stdout only (expect K=4:21, 5:15, 6:27; self-check, not graded). WCSS from final centroids (the centroids after the last update equal the means of the final, unchanged assignment).
- Elbow: smallest K in {3,4,5} with W(K)-W(K+1) < 0.5 (W(K-1)-W(K)); if none, report the last... defined fallback: 0 if none (documented in DECISIONS). Use K=3 afterwards.
- Segments: order clusters by ascending mean X3 original units -> A,B,C. CSV header `segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value`.
- Hierarchical: 6x6 table, brute-force agglomeration for single/complete/average (average = unweighted mean over member pairs). CSV header `linkage,merge_no,cluster_a,cluster_b,height`; clusters named `E1+E3`; cluster_a = the one with smaller min member number. Cut: gaps between consecutive heights, largest gap index g (first on tie) means merges 1..g are below the cut; clusters = 6 - g.
- PCA: standardised matrix, centre (mean ~0), S = X^T X/(n-1), `np.linalg.eigh`, sort descending, sign flip so largest-|entry| positive, project top 2. Row projections for rows 0,1,2; mean PC1 per segment.

### Part 4
Keys in order: `root_entropy, root_gini, best_split_by_info_gain, best_split_by_gini_gain, stump_rate_yes, stump_rate_no, round1_error, round1_alpha, round2_error, round2_alpha, ensemble_accuracy, oob_fraction, mlp_total, cnn_total, fc_layer_params, conv_layer_params, rnn_total`.
- Splits on 560 training rows, original units; S1 X6==1, S2 X7==1, S3 X3>0.5, S4 X1<=24. CSV `split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain`. Ties in best split: first (smallest label).
- Stump: rate on each side from train; test probability by side. Voting CSV `model,tp,fp,tn,fn,precision,recall,f1`, rows logistic, knn, stump, hard_vote, soft_vote (all threshold 0.3 except knn majority rule and hard vote count >= 2). KNN uses Part 2 best K, found by calling the common CV function.
- AdaBoost: arrays y,h1,h2 constants from the brief; CSV `round,point,weight_before,weight_unnormalised,weight_after`, 16 rows; ensemble accuracy via sign(a1 h1 + a2 h2) (sign 0 counts as wrong; check whether it occurs).
- Architecture: counts: MLP 784*16+16, 16*16+16, 16*10+10; CNN conv1 5*5*1*6+6, spatial 28 -> 28 -> pool 14; conv2 5*5*6*16+16 -> 10 -> pool 5; flatten 400 -> dense 32 -> 10. fc_layer_params = 49152*1024+1024; conv = 3*3*3*64+64; rnn = 128*64+128*128+16*128 + 128 + 16. Computed in code from shape arithmetic (functions), not typed. CSV `layer,parameters`.
- `oob_fraction = (1-1/560)**560`.

## 5. README outline (order fixed by the brief)
`Domain: E-commerce` / `## How to run` / `## Part 1 findings` (R1.1-R1.5) / `## Part 2 findings` (R2.1-R2.5) /
`## Part 3 findings` (R3.1-R3.4) / `## Part 4 findings` (R4.1-R4.4, incl. layer table) / `## Model-choice memo` (<=300 words) /
`## Limitations` (>=3). Numbers are copied from `outputs/` by a one-off helper script run during authoring (not committed as a graded
number source) and checked by tests. No URLs, no dollar signs, money as INR.

## 6. Memo content plan
Recommend logistic at tau from Task 2.8 (or whichever of the five Task 4.2 models has lowest computed test cost, computing the 2.8 cost for it
by the same rule only if done); cite >= 3 output numbers; threshold and INR cost; logistic cost as comparator; one limitation; one sentence on Model A vs B.

## 7. Test strategy (stdlib unittest, `python -m unittest discover tests`)
- keys: each results file has exactly the expected keys in order, no duplicates; format regexes (4 dp floats, ints, yes/no).
- CSV headers and row counts (9, 11, 3, 15, 4, 5, 16, 7).
- Invariants: confusion cells sum to 240, rates consistent with cells, segment sizes sum 800, TSS = ESS + RSS, eigenvalue_sum ~5, cumulative ratio monotone, adaboost weights sum 1.
- Checkpoints from the brief (mean_value, 560/119/240/51, loss_epoch_0, wcss_k2, eigenvalue_sum, root_entropy, round1_error), data MD5 over git-independent bytes.
- Forbidden-content grep over `*.py`: `polyfit|np.cov|corrcoef|lstsq|sklearn|scipy|pandas|numpy.cov`; `$` absent from README/outputs; no `http` in README; no image/PDF files tracked.
- README: first line, heading order, memo word count <= 300, quoted numbers appear in outputs.
- Determinism: run all scripts twice into a scratch copy and byte-compare outputs.

## 8. Deploy checklist (local only)
Clean clone in `_cleanclone_tmp`, fresh venv, `pip install -r requirements.txt`, README commands, `git diff --no-index` outputs vs committed, delete temp clone,
working tree clean, only branch `main`, `git ls-files` matches layout, no remotes added.
