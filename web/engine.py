"""Cached model computations on the project's pinned split (imports common.py only)."""
import os
import sys
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import common  # noqa: E402  (no side effects on import)
from web import outputs_io  # noqa: E402

FEATURES = ["customer_tenure_months", "complaint_count_90d", "discount_order_share",
            "avg_order_value_inr_hundreds", "loyalty_score", "cod_flag", "plus_member_flag"]
KS = list(range(1, 22, 2))
SEGMENT_LABELS = {"A": "Premium loyal", "B": "High-spend, complaint-prone",
                  "C": "Bargain-focused newcomers"}
CELL_KEYS = ("tp", "fp", "tn", "fn", "precision", "recall", "f1")


class Engine:
    def __init__(self):
        S = common.get_split()
        self.S = S
        self.w, self.b, _ = common.get_model(S)
        self.y = {"train": S["ytr"], "test": S["yte"]}
        self.p = {"train": common.predict_proba(S["Ztr"], self.w, self.b),
                  "test": common.predict_proba(S["Zte"], self.w, self.b)}
        self.cv_scores, self.best_k = common.knn_cv(S["Ztr"], S["ytr"])
        self.knn = {K: common.knn_predict(S["Ztr"], S["ytr"], S["Zte"], K) for K in KS}
        # Part 4 stump: best information-gain split (tenure <= 24 months, X1)
        Xtr, ytr, Xte = S["Xtr"], S["ytr"], S["Xte"]
        yes = Xtr[:, 0] <= 24
        rate_yes = float(ytr[yes].sum()) / int(yes.sum())
        rate_no = float(ytr[~yes].sum()) / int((~yes).sum())
        self.p_stump = np.where(Xte[:, 0] <= 24, rate_yes, rate_no)
        self.segments = outputs_io.read_csv_table("part3_segments.csv")
        n = sum(s["size"] for s in self.segments)
        cols = ["mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"]
        self.overall = {c: sum(s[c] * s["size"] for s in self.segments) / n for c in cols}
        self.overall["size"] = n
        self.sd5 = S["sd"][:5]

    def cells(self, y, pred):
        tp, fp, tn, fn = common.confusion(y, pred)
        return dict(tp=tp, fp=fp, tn=tn, fn=fn, **common.rates(tp, fp, tn, fn))

    def threshold(self, tau, dataset, fp_cost, fn_cost):
        y, p = self.y[dataset], self.p[dataset]
        c = self.cells(y, p >= tau)
        cost = fp_cost * c["fp"] + fn_cost * c["fn"]
        best_tau, best_cost = None, None
        for k in range(1, 20):
            t = k / 20
            _, fp, _, fn = common.confusion(y, p >= t)
            cc = fp_cost * fp + fn_cost * fn
            if best_cost is None or cc < best_cost:
                best_tau, best_cost = t, cc
        return dict(tau=tau, dataset=dataset, fp_cost=fp_cost, fn_cost=fn_cost, n=len(y),
                    positives=int(y.sum()), cost=cost, cost_flag_nobody=fn_cost * int(y.sum()),
                    cost_best_tau=best_tau, cost_at_best_tau=best_cost, **c)

    def curve(self, dataset):
        y, p = self.y[dataset], self.p[dataset]
        pts = []
        for k in range(0, 101):
            t = k / 100
            r = common.rates(*common.confusion(y, p >= t))
            pts.append(dict(tau=t, tpr=r["recall"], fpr=r["fpr"]))
        return pts

    def knn_result(self, K):
        out = self.cells(self.y["test"], self.knn[K][1])
        out.update(k=K, cv_f1=self.cv_scores[K], best_k=self.best_k)
        return out

    def ensemble(self, tau, rule, models):
        y = self.y["test"]
        probs = {"logistic": self.p["test"], "knn": self.knn[self.best_k][0], "stump": self.p_stump}
        preds = {"logistic": probs["logistic"] >= tau,
                 "knn": self.knn[self.best_k][1],  # Part 4: majority of K neighbours
                 "stump": probs["stump"] >= tau}
        names = ("logistic", "knn", "stump")
        rows = []
        for m in names:
            r = self.cells(y, preds[m])
            rows.append(dict(model=m, **{k: r[k] for k in CELL_KEYS}))
        sel = [m for m in names if m in models]
        if rule == "hard":
            vote = sum(preds[m].astype(int) for m in sel) > len(sel) / 2.0
        else:
            vote = sum(probs[m] for m in sel) / len(sel) >= tau
        r = self.cells(y, vote)
        rows.append(dict(model=rule + "_vote", **{k: r[k] for k in CELL_KEYS}))
        return dict(tau=tau, rule=rule, members=sel, rows=rows)

    def score(self, rec, tau=0.15):
        x = np.array([rec.tenure_months, rec.complaints_90d, rec.discount_share, rec.avg_order_value,
                      rec.loyalty_score, rec.cod_flag, rec.plus_member_flag], dtype=float)
        z = (x - self.S["mu"]) / self.S["sd"]
        contrib = z * self.w
        prob = float(common.sigmoid(float(contrib.sum()) + self.b))
        band = "high" if prob >= 0.5 else ("elevated" if prob >= tau else "low")
        d = [sum(((x[i] - s["mean_X%d" % (i + 1)]) / self.sd5[i]) ** 2 for i in range(5))
             for s in self.segments]
        seg = self.segments[int(np.argmin(d))]["segment"]
        return dict(probability=prob, tau=tau, flag_at_tau=bool(prob >= tau), risk_band=band,
                    bias=float(self.b),
                    contributions=[dict(feature=FEATURES[i], z=float(z[i]), weight=float(self.w[i]),
                                        effect=float(contrib[i])) for i in range(7)],
                    nearest_segment=seg, nearest_segment_label=SEGMENT_LABELS[seg])


_engine = None
_lock = threading.Lock()


def get_engine():
    global _engine
    with _lock:
        if _engine is None:
            _engine = Engine()
    return _engine
