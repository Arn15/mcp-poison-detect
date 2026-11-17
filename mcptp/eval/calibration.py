"""Registry-relative threshold calibration (own method).

Problem (robust3): thresholds fixed at 1% FPR on validation negatives realize 7-24% FPR on test servers. The
negatives' scores are squeezed near 0 and shift per server (e.g. most false alarms come from 2-3 servers), so a
global quantile from 4 validation servers does not transfer.

Fix: at registration time the server's already-registered tools are known. Score the candidate RELATIVE to them,
    r(c) = logit s(c) - max_{t in R(server), name(t) != name(c)} logit s(t),
i.e. "how much more suspicious is this tool than every tool already on this server". This cancels the
per-server offset without labels from the new server. Thresholds are then fixed on validation as before.
Variants: stat="median" (less conservative), and a worst-server global threshold baseline.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .metrics import threshold_at_fpr


def logit(p: float, eps: float = 1e-7) -> float:
    """log(p / (1 - p)) with p clipped to [eps, 1 - eps]."""
    p = min(max(float(p), eps), 1 - eps)
    return math.log(p / (1 - p))


class RegistryCalibrator:
    """Registry-relative calibration: r(c) = logit s(c) - stat over logit s(t) of the server's other registered tools.

    ref: iterable of (server, tool_name, score) for registered (trusted) tools; stat = "max" (default) or "median".
    transform(server, name, score) returns r(c), or None when the server has no other registered tool.
    """

    def __init__(self, ref, stat: str = "max"):
        self.stat = stat
        self.ref = defaultdict(list)
        for server, name, s in ref:
            self.ref[server].append((name, logit(s)))

    def transform(self, server: str, name: str, score: float) -> float | None:
        r = [v for n, v in self.ref.get(server, []) if n != name]
        if not r:
            return None                       # no reference tools -> cannot calibrate (reported as coverage)
        anchor = max(r) if self.stat == "max" else float(np.median(r))
        return logit(score) - anchor


def worst_group_threshold(scores_by_group: dict, fpr: float) -> float:
    """Global threshold = max over validation servers of each server's own FPR-quantile (conservative baseline)."""
    return max(threshold_at_fpr([logit(s) for s in v], fpr) for v in scores_by_group.values() if len(v))
