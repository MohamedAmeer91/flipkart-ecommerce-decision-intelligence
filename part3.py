"""Part 3: segmentation and dimensionality reduction. Run: python part3.py"""
import numpy as np

from common import col_mean_sd, load_data, write_csv, write_results


def kmeans(Z, K):
    """Pinned K-means. Returns labels, centroids, number of assignment passes."""
    n = Z.shape[0]
    step = n // K
    cent = np.array([Z[i * step] for i in range(K)], dtype=float)
    labels = np.full(n, -1)
    passes = 0
    while True:
        passes += 1
        d2 = np.sum((Z[:, None, :] - cent[None, :, :]) ** 2, axis=2)
        new = np.argmin(d2, axis=1)  # first minimum = lower-numbered centroid
        if np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            members = Z[labels == k]
            if len(members) > 0:
                cent[k] = np.sum(members, axis=0) / len(members)
    return labels, cent, passes


def wcss(Z, labels, cent):
    return float(np.sum((Z - cent[labels]) ** 2))


DISS = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def agglomerate(D, linkage):
    """Return list of (cluster_a, cluster_b, height); clusters are tuples of 0-based members."""
    clusters = [(i,) for i in range(len(D))]
    merges = []

    def link(a, b):
        ds = [D[i, j] for i in a for j in b]
        if linkage == "single":
            return min(ds)
        if linkage == "complete":
            return max(ds)
        return sum(ds) / len(ds)

    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                h = link(clusters[i], clusters[j])
                if best is None or h < best[0]:
                    best = (h, i, j)
        h, i, j = best
        a, b = clusters[i], clusters[j]
        merged = tuple(sorted(a + b))
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
        first, second = (a, b) if min(a) < min(b) else (b, a)
        merges.append((first, second, h))
    return merges


def name(c):
    return "+".join("E%d" % (m + 1) for m in sorted(c))


def clusters_at_cut(heights):
    gaps = [heights[i + 1] - heights[i] for i in range(len(heights) - 1)]
    g = 0
    for i in range(len(gaps)):
        if gaps[i] > gaps[g]:
            g = i
    return 6 - (g + 1)  # merges 1..g+1 lie below the cut


def main():
    X, y, value = load_data()
    X5 = X[:, :5]
    mu, sd = col_mean_sd(X5)
    Z = (X5 - mu) / sd
    out = []

    # 3.1
    W = {}
    runs = {}
    for K in range(2, 7):
        labels, cent, passes = kmeans(Z, K)
        W[K] = wcss(Z, labels, cent)
        runs[K] = (labels, cent)
        print("K=%d assignment passes=%d" % (K, passes))
    for K in range(2, 7):
        out.append(("wcss_k%d" % K, W[K]))

    # 3.2
    elbow = 0
    for K in (3, 4, 5):
        if W[K] - W[K + 1] < 0.5 * (W[K - 1] - W[K]):
            elbow = K
            break
    out.append(("elbow_k", elbow))

    # 3.3 segments from K = 3
    labels = runs[3][0]
    x3_means = [float(np.sum(X5[labels == k, 2])) / int(np.sum(labels == k)) for k in range(3)]
    order = sorted(range(3), key=lambda k: x3_means[k])
    seg_rows = []
    seg_of = np.zeros(len(labels), dtype=int)
    for rank, k in enumerate(order):
        m = labels == k
        seg_of[m] = rank
        cnt = int(np.sum(m))
        seg_rows.append(["ABC"[rank], cnt] + [float(np.sum(X5[m, j])) / cnt for j in range(5)]
                        + [float(np.sum(y[m])) / cnt, float(np.sum(value[m])) / cnt])
    write_csv("part3_segments.csv",
              ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
               "outcome_rate", "mean_value"], seg_rows)
    for r in seg_rows:
        out.append(("size_segment_" + r[0], r[1]))

    # 3.4
    hrows = []
    cuts = {}
    for lk in ("single", "complete", "average"):
        merges = agglomerate(DISS, lk)
        for i, (a, b, h) in enumerate(merges):
            hrows.append([lk, i + 1, name(a), name(b), h])
        cuts[lk] = clusters_at_cut([m[2] for m in merges])
    write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hrows)
    for lk in ("single", "complete", "average"):
        out.append(("%s_clusters_at_cut" % lk, cuts[lk]))

    # 3.5 PCA in seven steps
    n = Z.shape[0]
    mean_vec = np.sum(Z, axis=0) / n          # step 1 (zeros)
    Xc = Z - mean_vec                         # step 2
    S = Xc.T @ Xc / (n - 1)                   # step 3
    vals, vecs = np.linalg.eigh(S)            # step 4
    idx = np.argsort(-vals, kind="stable")    # step 5
    vals, vecs = vals[idx], vecs[:, idx]
    for j in range(5):                        # sign convention
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] = -vecs[:, j]
    Wm = vecs[:, :2]                          # step 6
    P = Xc @ Wm                               # step 7
    total = float(np.sum(vals))
    out.append(("eigenvalue_sum", total))
    for j in range(5):
        out.append(("eigenvalue_%d" % (j + 1), float(vals[j])))
    ratios = [float(v) / total for v in vals]
    for j in range(5):
        out.append(("explained_ratio_%d" % (j + 1), ratios[j]))
    cum, c80 = 0.0, 0
    for j in range(5):
        cum += ratios[j]
        out.append(("cumulative_ratio_%d" % (j + 1), cum))
        if c80 == 0 and cum >= 0.8:
            c80 = j + 1
    out.append(("components_for_80pct", c80))
    for pc in (0, 1):
        for j in range(5):
            out.append(("pc%d_loading_X%d" % (pc + 1, j + 1), float(vecs[j, pc])))
    for r in range(3):
        out.append(("row%d_pc1" % (r + 1), float(P[r, 0])))
        out.append(("row%d_pc2" % (r + 1), float(P[r, 1])))
    for s in range(3):
        m = seg_of == s
        out.append(("mean_pc1_segment_" + "ABC"[s], float(np.sum(P[m, 0])) / int(np.sum(m))))

    write_results("part3_results.txt", out)


if __name__ == "__main__":
    main()
