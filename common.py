"""Shared helpers for part1.py to part4.py. Importing this module runs nothing."""
import csv
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "records.csv")
OUT_DIR = os.path.join(HERE, "outputs")


# ---------------------------------------------------------------- data
def load_data(path=DATA_PATH):
    """Return X (800 x 7: X1..X7), outcome (int array), value (float array)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    X = np.array([[float(r[j]) for j in range(1, 8)] for r in rows])
    y = np.array([int(r[8]) for r in rows])
    value = np.array([float(r[9]) for r in rows])
    return X, y, value


def mean(a):
    return float(np.sum(a)) / len(a)


def sample_sd(a):
    m = mean(a)
    return math.sqrt(float(np.sum((a - m) ** 2)) / (len(a) - 1))


def col_mean_sd(M):
    n = M.shape[0]
    mu = np.sum(M, axis=0) / n
    sd = np.sqrt(np.sum((M - mu) ** 2, axis=0) / (n - 1))
    return mu, sd


# ---------------------------------------------------------------- split
def split_stratified(y):
    """First floor(7*count/10) rows of each class (file order) go to train."""
    train, test = [], []
    for cls in (0, 1):
        idx = [i for i in range(len(y)) if y[i] == cls]
        cut = (7 * len(idx)) // 10
        train += idx[:cut]
        test += idx[cut:]
    return np.array(sorted(train)), np.array(sorted(test))


def get_split():
    """Task 2.1: split plus standardised features (train mean/SD only)."""
    X, y, value = load_data()
    tr, te = split_stratified(y)
    mu, sd = col_mean_sd(X[tr])
    Ztr = (X[tr] - mu) / sd
    Zte = (X[te] - mu) / sd
    return dict(X=X, y=y, value=value, tr=tr, te=te, Xtr=X[tr], Xte=X[te],
                ytr=y[tr], yte=y[te], Ztr=Ztr, Zte=Zte, mu=mu, sd=sd)


# ---------------------------------------------------------------- logistic regression
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y))


def fit_logistic(Z, y, eta, steps, record=()):
    """Batch gradient descent from zero. Returns w, b and {step: loss} for steps in record."""
    n, d = Z.shape
    w = np.zeros(d)
    b = 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(y, sigmoid(Z @ w + b))
    for s in range(1, steps + 1):
        p = sigmoid(Z @ w + b)
        err = p - y
        w = w - eta * (Z.T @ err) / n
        b = b - eta * float(np.sum(err)) / n
        if s in record:
            losses[s] = bce(y, sigmoid(Z @ w + b))
    return w, b, losses


def predict_proba(Z, w, b):
    return sigmoid(Z @ w + b)


def get_model(S):
    """The Task 2.2 model (1000 updates, eta 0.1) on the training rows of split S."""
    return fit_logistic(S["Ztr"], S["ytr"], 0.1, 1000, record=(0, 1, 10, 100, 1000))


# ---------------------------------------------------------------- metrics
def confusion(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return tp, fp, tn, fn


def _div(a, b):
    return a / b if b else 0.0


def rates(tp, fp, tn, fn):
    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    return dict(accuracy=_div(tp + tn, tp + fp + tn + fn), precision=prec, recall=rec,
                f1=_div(2 * prec * rec, prec + rec), specificity=_div(tn, tn + fp),
                fpr=_div(fp, fp + tn))


# ---------------------------------------------------------------- KNN
def knn_share(Zref, yref, Zq, K):
    """Share of positive neighbours among the K nearest (ties by reference order)."""
    shares = np.zeros(len(Zq))
    for i in range(len(Zq)):
        d2 = np.sum((Zref - Zq[i]) ** 2, axis=1)
        order = np.argsort(d2, kind="stable")[:K]
        shares[i] = float(np.sum(yref[order])) / K
    return shares


def knn_predict(Zref, yref, Zq, K):
    """Positive when at least (K+1)/2 of K neighbours are positive."""
    shares = knn_share(Zref, yref, Zq, K)
    need = (K + 1) // 2
    pred = np.array([int(round(s * K)) >= need for s in shares])
    return shares, pred


def knn_cv(Ztr, ytr, ks=range(1, 22, 2), folds=5):
    """Mean over folds of held-out F1 for each K. Fold = position mod 5."""
    pos = np.arange(len(ytr)) % folds
    scores = {}
    for K in ks:
        f1s = []
        for f in range(folds):
            ref = pos != f
            _, pred = knn_predict(Ztr[ref], ytr[ref], Ztr[~ref], K)
            f1s.append(rates(*confusion(ytr[~ref], pred))["f1"])
        scores[K] = sum(f1s) / folds
    best = None
    for K in ks:
        if best is None or scores[K] > scores[best]:
            best = K
    return scores, best


# ---------------------------------------------------------------- output
def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%.4f" % v
    return str(v)


def write_results(name, items):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", newline="\n") as f:
        for k, v in items:
            f.write("%s = %s\n" % (k, fmt(v)))


def write_csv(name, header, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")
