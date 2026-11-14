"""Part 2: classification and evaluation. Run: python part2.py"""
import math

import numpy as np

from common import (bce, confusion, fit_logistic, get_model, get_split, knn_cv, knn_predict,
                    predict_proba, rates, write_csv, write_results)


def main():
    S = get_split()
    ytr, yte, Ztr, Zte = S["ytr"], S["yte"], S["Ztr"], S["Zte"]
    out = [("train_rows", len(ytr)), ("train_positives", int(ytr.sum())),
           ("test_rows", len(yte)), ("test_positives", int(yte.sum()))]

    # 2.2
    w, b, L = get_model(S)
    for e in (0, 1, 10, 100, 1000):
        out.append(("loss_epoch_%d" % e, L[e]))
    for j in range(7):
        out.append(("weight_X%d" % (j + 1), float(w[j])))
    ptest = predict_proba(Zte, w, b)
    ptrain = predict_proba(Ztr, w, b)
    out += [("bias", b), ("test_bce", bce(yte, ptest))]

    # 2.3
    for eta in (0.01, 0.1, 1.0):
        _, _, l200 = fit_logistic(Ztr, ytr, eta, 200, record=(200,))
        out.append(("loss_epoch_200_eta_%s" % eta, l200[200]))

    # 2.4
    for j in range(7):
        out.append(("odds_ratio_X%d" % (j + 1), math.exp(w[j])))

    # 2.5
    for tau in (0.5, 0.3):
        cells = confusion(yte, ptest >= tau)
        r = rates(*cells)
        for k, v in zip(("tp", "fp", "tn", "fn"), cells):
            out.append(("tau_%s_%s" % (tau, k), v))
        for k in ("accuracy", "precision", "recall", "f1", "specificity"):
            out.append(("tau_%s_%s" % (tau, k), r[k]))
    out.append(("blind_accuracy", float(np.sum(yte == 0)) / len(yte)))

    # 2.6
    rows = []
    for k in range(1, 10):
        tau = k / 10
        tp, fp, tn, fn = confusion(yte, ptest >= tau)
        r = rates(tp, fp, tn, fn)
        rows.append([tau, tp, fp, tn, fn, r["recall"], r["fpr"]])
    write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

    # 2.7
    scores, best = knn_cv(Ztr, ytr)
    write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], [[K, scores[K]] for K in sorted(scores)])
    _, pred = knn_predict(Ztr, ytr, Zte, best)
    cells = confusion(yte, pred)
    r = rates(*cells)
    out.append(("knn_best_k", best))
    for k, v in zip(("tp", "fp", "tn", "fn"), cells):
        out.append(("knn_" + k, v))
    for k in ("accuracy", "precision", "recall", "f1"):
        out.append(("knn_" + k, r[k]))

    # 2.8 (tau chosen on training rows only)
    def cost(y, p, tau):
        _, fp, _, fn = confusion(y, p >= tau)
        return 50 * fp + 400 * fn

    best_tau, best_cost = None, None
    for k in range(1, 20):
        tau = k / 20
        c = cost(ytr, ptrain, tau)
        if best_cost is None or c < best_cost:
            best_tau, best_cost = tau, c
    out += [("cost_tau", best_tau), ("cost_train_at_tau", best_cost),
            ("cost_test_at_tau", cost(yte, ptest, best_tau)),
            ("cost_test_at_0.5", cost(yte, ptest, 0.5)),
            ("cost_test_flag_nobody", 400 * int(yte.sum()))]

    write_results("part2_results.txt", out)


if __name__ == "__main__":
    main()
