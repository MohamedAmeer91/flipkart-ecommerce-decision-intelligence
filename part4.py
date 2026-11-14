"""Part 4: model-choice and architecture memo numbers. Run: python part4.py"""
import math

import numpy as np

from common import (confusion, get_model, get_split, knn_cv, knn_predict,
                    predict_proba, rates, write_csv, write_results)


def entropy(pos, n):
    if n == 0:
        return 0.0
    e = 0.0
    for c in (pos, n - pos):
        if c > 0:
            p = c / n
            e -= p * math.log2(p)
    return e


def gini(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    return 1.0 - p * p - (1 - p) * (1 - p)


def conv_out(size, f, pad, stride):
    return (size - f + 2 * pad) // stride + 1


def main():
    S = get_split()
    Xtr, ytr, Xte, yte = S["Xtr"], S["ytr"], S["Xte"], S["yte"]
    Ztr, Zte = S["Ztr"], S["Zte"]
    ntr = len(ytr)
    out = []

    # 4.1
    P = int(ytr.sum())
    out += [("root_entropy", entropy(P, ntr)), ("root_gini", gini(P, ntr))]
    conds = [("S1", lambda X: X[:, 5] == 1), ("S2", lambda X: X[:, 6] == 1),
             ("S3", lambda X: X[:, 2] > 0.5), ("S4", lambda X: X[:, 0] <= 24)]
    rows, gains_i, gains_g = [], {}, {}
    for nm, f in conds:
        yes = f(Xtr)
        ny, py = int(yes.sum()), int(ytr[yes].sum())
        nn, pn = ntr - ny, P - py
        ig = entropy(P, ntr) - (ny / ntr * entropy(py, ny) + nn / ntr * entropy(pn, nn))
        gg = gini(P, ntr) - (ny / ntr * gini(py, ny) + nn / ntr * gini(pn, nn))
        gains_i[nm], gains_g[nm] = ig, gg
        rows.append([nm, ny, py, nn, pn, ig, gg])
    write_csv("part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no",
                                   "info_gain", "gini_gain"], rows)
    best_i = max(gains_i, key=lambda k: gains_i[k])   # first maximum wins ties
    best_g = max(gains_g, key=lambda k: gains_g[k])
    out += [("best_split_by_info_gain", best_i), ("best_split_by_gini_gain", best_g)]

    # 4.2
    f = dict(conds)[best_i]
    yes_tr = f(Xtr)
    rate_yes = float(ytr[yes_tr].sum()) / int(yes_tr.sum())
    rate_no = float(ytr[~yes_tr].sum()) / int((~yes_tr).sum())
    out += [("stump_rate_yes", rate_yes), ("stump_rate_no", rate_no)]
    w, b, _ = get_model(S)
    p_log = predict_proba(Zte, w, b)
    _, best_k = knn_cv(Ztr, ytr)
    p_knn, pred_knn = knn_predict(Ztr, ytr, Zte, best_k)
    p_stump = np.where(f(Xte), rate_yes, rate_no)
    pred_log = p_log >= 0.3
    pred_stump = p_stump >= 0.3
    hard = (pred_log.astype(int) + pred_knn.astype(int) + pred_stump.astype(int)) >= 2
    soft = (p_log + p_knn + p_stump) / 3 >= 0.3
    vrows = []
    for nm, pr in (("logistic", pred_log), ("knn", pred_knn), ("stump", pred_stump),
                   ("hard_vote", hard), ("soft_vote", soft)):
        tp, fp, tn, fn = confusion(yte, pr)
        r = rates(tp, fp, tn, fn)
        vrows.append([nm, tp, fp, tn, fn, r["precision"], r["recall"], r["f1"]])
    write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vrows)

    # 4.3
    y = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    hs = [np.array([1, 1, -1, -1, -1, -1, -1, -1]), np.array([-1, 1, 1, -1, 1, -1, 1, -1])]
    wt = np.full(8, 1 / 8)
    arows, alphas = [], []
    for t, h in enumerate(hs):
        eps = float(np.sum(wt[h != y]))
        alpha = 0.5 * math.log((1 - eps) / eps)
        unnorm = wt * np.exp(-alpha * y * h)
        new = unnorm / np.sum(unnorm)
        for i in range(8):
            arows.append([t + 1, i + 1, float(wt[i]), float(unnorm[i]), float(new[i])])
        out += [("round%d_error" % (t + 1), eps), ("round%d_alpha" % (t + 1), alpha)]
        alphas.append(alpha)
        wt = new
    write_csv("part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised",
                                     "weight_after"], arows)
    score = alphas[0] * hs[0] + alphas[1] * hs[1]
    out.append(("ensemble_accuracy", float(np.sum(np.sign(score) == y)) / 8))

    # 4.4
    out.append(("oob_fraction", (1 - 1 / ntr) ** ntr))

    # 4.5
    mlp = [("mlp_hidden1", 784 * 16 + 16), ("mlp_hidden2", 16 * 16 + 16), ("mlp_output", 16 * 10 + 10)]
    s1 = conv_out(28, 5, 2, 1)
    s1p = conv_out(s1, 2, 0, 2)
    s2 = conv_out(s1p, 5, 0, 1)
    s2p = conv_out(s2, 2, 0, 2)
    flat = s2p * s2p * 16
    cnn = [("cnn_conv1", 5 * 5 * 1 * 6 + 6), ("cnn_conv2", 5 * 5 * 6 * 16 + 16),
           ("cnn_dense1", flat * 32 + 32), ("cnn_dense2", 32 * 10 + 10)]
    write_csv("part4_architecture.csv", ["layer", "parameters"], [list(r) for r in mlp + cnn])
    out += [("mlp_total", sum(v for _, v in mlp)), ("cnn_total", sum(v for _, v in cnn)),
            ("fc_layer_params", 128 * 128 * 3 * 1024 + 1024),
            ("conv_layer_params", 3 * 3 * 3 * 64 + 64),
            ("rnn_total", 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16)]
    print("cnn shapes: conv1 %dx%dx6, pool %dx%dx6, conv2 %dx%dx16, pool %dx%dx16, flatten %d"
          % (s1, s1, s1p, s1p, s2, s2, s2p, s2p, flat))

    write_results("part4_results.txt", out)


if __name__ == "__main__":
    main()
