"""Detection metrics with stratified bootstrap CIs.

Primary: TPR at a fixed FPR (default 1%), computed as the highest TPR whose FPR <= target on the
evaluated negatives. With few test negatives (~150) the 1% operating point is coarse, so we also report
5% and always give 95% CIs. `threshold_at_fpr` picks a threshold on validation data for deployment-style
evaluation (score threshold fixed before seeing test).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(y, s, fpr: float = 0.01) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if y.min() == y.max():
        return float("nan")
    f, t, _ = roc_curve(y, s)
    ok = f <= fpr + 1e-12
    return float(t[ok].max()) if ok.any() else 0.0


def threshold_at_fpr(y_neg_scores, fpr: float = 0.01) -> float:
    """Smallest threshold such that at most `fpr` of these negatives score >= it."""
    s = np.sort(np.asarray(y_neg_scores, dtype=float))[::-1]
    k = int(np.floor(fpr * len(s)))
    return float(np.nextafter(s[k], np.inf)) if k < len(s) else float(s[-1])


def rates_at_threshold(y, s, thr: float) -> dict:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pred = s >= thr
    pos, neg = y == 1, y == 0
    return {"tpr": float(pred[pos].mean()) if pos.any() else float("nan"),
            "fpr": float(pred[neg].mean()) if neg.any() else float("nan")}


def auroc(y, s) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if y.min() != y.max() else float("nan")


def bootstrap_ci(metric, y, s, n: int = 1000, ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi) with class-stratified resampling."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))])
        vals.append(metric(y[idx], s[idx]))
    vals = np.asarray(vals)
    a = (1 - ci) / 2
    return float(metric(y, s)), float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a))


def summarize(y, s, fprs=(0.01, 0.05), n_boot: int = 1000, seed: int = 0) -> dict:
    out = {"n_pos": int(np.sum(np.asarray(y) == 1)), "n_neg": int(np.sum(np.asarray(y) == 0))}
    out["auroc"] = bootstrap_ci(auroc, y, s, n_boot, seed=seed)
    for f in fprs:
        out[f"tpr@{f:g}fpr"] = bootstrap_ci(lambda a, b, f=f: tpr_at_fpr(a, b, f), y, s, n_boot, seed=seed)
    return out
