"""Evaluate score vectors on the standard test subsets."""
from __future__ import annotations

import numpy as np

from .metrics import rates_at_threshold, summarize, threshold_at_fpr


def subsets(test: list[dict]) -> dict[str, np.ndarray]:
    """Boolean masks over `test`: every subset keeps all negatives."""
    lab = np.array([e["label"] for e in test])
    neg = lab == 0
    imp = np.array([bool(e.get("is_implicit")) for e in test])
    fam = np.array([bool(e.get("heldout_family")) for e in test])
    return {"all": np.ones(len(test), bool), "implicit": neg | (imp & ~neg),
            "heldout_server_only": neg | (~fam & ~neg), "heldout_family": neg | (fam & ~neg)}


def evaluate(test, s_test, val, s_val, fprs=(0.01, 0.05), n_boot=1000, seed=0) -> dict:
    y = np.array([e["label"] for e in test])
    s_test = np.asarray(s_test, float)
    out = {name: summarize(y[m], s_test[m], fprs, n_boot, seed) for name, m in subsets(test).items()}
    yv = np.array([e["label"] for e in val])
    hn = np.array([bool(e.get("is_hard_negative")) for e in test])
    for f in fprs:
        thr = threshold_at_fpr(np.asarray(s_val)[yv == 0], f)
        r = rates_at_threshold(y, s_test, thr)
        r["hard_neg_fpr"] = float((s_test[(y == 0) & hn] >= thr).mean()) if ((y == 0) & hn).any() else float("nan")
        out[f"val_thr@{f:g}"] = r
    return out
