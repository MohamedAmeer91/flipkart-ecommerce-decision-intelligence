"""Part 1: estimation, inference and a regression baseline. Run: python part1.py"""
import math

import numpy as np

from common import load_data, mean, sample_sd, write_results


def simple_fit(x, y):
    """Closed-form simple regression with divisor n-1 for cov and var."""
    n = len(x)
    mx, my = mean(x), mean(y)
    cov = float(np.sum((x - mx) * (y - my))) / (n - 1)
    var = float(np.sum((x - mx) ** 2)) / (n - 1)
    slope = cov / var
    return slope, my - slope * mx


def halves_sd_ratio(fitted, resid):
    order = np.argsort(fitted, kind="stable")
    lo = sample_sd(resid[order[:400]])
    hi = sample_sd(resid[order[400:]])
    return lo, hi, hi / lo


def main():
    X, _, value = load_data()
    x4 = X[:, 3]
    y = value
    n = len(y)
    out = []

    # 1.1
    m = mean(y)
    sd = sample_sd(y)
    se = sd / math.sqrt(n)
    out += [("n", n), ("mean_value", m), ("sd_value", sd), ("se_value", se),
            ("ci95_low", m - 1.96 * se), ("ci95_high", m + 1.96 * se)]
    # 1.2
    z = (m - 8.0) / se
    out += [("z_stat", z), ("reject_h0", abs(z) > 1.96)]
    # 1.3
    out += [("n_required", math.ceil((1.96 * sd / 0.5) ** 2))]

    # 1.4
    slope, icpt = simple_fit(x4, y)
    fitted = icpt + slope * x4
    tss = float(np.sum((y - m) ** 2))
    ess = float(np.sum((fitted - m) ** 2))
    rss = float(np.sum((y - fitted) ** 2))
    resid = y - fitted
    out += [("reg_slope", slope), ("reg_intercept", icpt), ("tss", tss), ("ess", ess),
            ("rss", rss), ("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss),
            ("r2", 1 - rss / tss), ("rmse", math.sqrt(rss / n)),
            ("mae", float(np.sum(np.abs(resid))) / n)]

    # 1.5 gradient descent on standardised X4
    mx, sx = mean(x4), sample_sd(x4)
    zx = (x4 - mx) / sx
    mm, cc = 0.0, 0.0
    costs = {}

    def cost(a, b):
        e = y - (a * zx + b)
        return float(np.sum(e ** 2)) / n

    costs[0] = cost(mm, cc)
    for step in range(1, 501):
        e = y - (mm * zx + cc)
        gm = -2.0 / n * float(np.sum(zx * e))
        gc = -2.0 / n * float(np.sum(e))
        mm -= 0.1 * gm
        cc -= 0.1 * gc
        if step in (1, 10, 100, 500):
            costs[step] = cost(mm, cc)
    gd_slope = mm / sx
    gd_icpt = cc - mm * mx / sx
    out += [("gd_J0", costs[0]), ("gd_J1", costs[1]), ("gd_J10", costs[10]),
            ("gd_J100", costs[100]), ("gd_J500", costs[500]), ("gd_slope", gd_slope),
            ("gd_intercept", gd_icpt), ("gd_slope_gap", abs(gd_slope - slope))]

    # 1.6 funnel and log remedy
    lo, hi, ratio = halves_sd_ratio(fitted, resid)
    out += [("resid_sd_lower_raw", lo), ("resid_sd_upper_raw", hi), ("resid_sd_ratio_raw", ratio)]
    lx, ly = np.log(x4), np.log(y)
    ls, li = simple_fit(lx, ly)
    lfit = li + ls * lx
    lres = ly - lfit
    ltss = float(np.sum((ly - mean(ly)) ** 2))
    lrss = float(np.sum(lres ** 2))
    lr2 = 1 - lrss / ltss
    out += [("loglog_intercept", li), ("loglog_slope", ls), ("loglog_r2", lr2),
            ("loglog_rmse", math.sqrt(lrss / n))]
    lo, hi, ratio = halves_sd_ratio(lfit, lres)
    out += [("resid_sd_lower_log", lo), ("resid_sd_upper_log", hi), ("resid_sd_ratio_log", ratio)]

    # 1.7 normal equation: [1, ln X4, X1, X2, X3, X5, X6, X7]
    D = np.column_stack([np.ones(n), lx, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
    beta = np.linalg.solve(D.T @ D, D.T @ ly)
    rss_m = float(np.sum((ly - D @ beta) ** 2))
    r2_m = 1 - rss_m / ltss
    for j in range(8):
        out.append(("mr_beta_%d" % j, float(beta[j])))
    out += [("mr_r2", r2_m), ("mr_adj_r2", 1 - (1 - r2_m) * (n - 1) / (n - 7 - 1)),
            ("loglog_adj_r2", 1 - (1 - lr2) * (n - 1) / (n - 1 - 1))]

    write_results("part1_results.txt", out)


if __name__ == "__main__":
    main()
