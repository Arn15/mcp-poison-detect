"""Threshold calibration study on a score_robustness run (CPU, seconds).

Compares, for each defender/seed, the realized test FPR and detection when thresholds are fixed on validation
negatives at a target FPR (1%, 5%):
  global        - current deployable rule (quantile of val-negative scores)
  worst_server  - max over val servers of per-server quantiles
  rel_median    - registry-relative score, anchor = median of the server's registered tools
  rel_max       - registry-relative score, anchor = max of the server's registered tools (mcptp.eval.calibration)
Reference tools = original benign tools of each server, excluding any tool with the candidate's name (so a
paraphrase is never compared with its own original). If the run has a "registry" set (score_robustness with
score.registry=true) all servers are covered; otherwise only val/test servers are, and coverage n is reported.

Usage: python scripts/calibrate.py --run results/robust3 [--out results/calibration]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcptp.eval.calibration import RegistryCalibrator, logit, worst_group_threshold  # noqa: E402
from mcptp.eval.metrics import threshold_at_fpr  # noqa: E402
from mcptp.utils.env import REPO_ROOT  # noqa: E402

POS_SETS = ("test_pos", "r0_opt", "r1_opt", "r1b_opt", "r2_opt", "r1t_opt", "r2t_opt")
METHODS = ("global", "worst_server", "rel_median", "rel_max")


def item_index(run_dir: Path):
    """id -> (server, tool_name, split) for every scored item (examples, para, attack sets)."""
    idx = {}
    for f in (REPO_ROOT / "data" / "processed" / "examples.jsonl", REPO_ROOT / "data" / "para" / "para_v1.jsonl"):
        for e in map(json.loads, open(f)):
            idx[e["id"]] = (e["server"], e["tool_name"], e.get("split"), e.get("kind"), e["id"].startswith("x_"))
    from score_robustness import attack_sets, load_verified   # same id construction as the scoring run
    cfg = json.loads(json.dumps(__import__("yaml").safe_load(open(run_dir / "config.yaml"))))
    regs = defaultdict(dict)
    for e in map(json.loads, open(REPO_ROOT / "data" / "processed" / "examples.jsonl")):
        if e["kind"] == "benign":
            regs[e["server"]][e["tool_name"]] = e["description"]
    verified = load_verified() if (cfg.get("score") or {}).get("verified", True) else None
    for name, exs in attack_sets(regs, cfg["context"]["top_k"], verified=verified).items():
        for e in exs:
            idx[e["id"]] = (e["server"], e["tool_name"], None, "attack", False)
    return idx


def evaluate(sets, idx, method, fpr):
    """sets: {set_name: {id: score}} for one defender/seed -> {set: (rate, n_covered)} + val threshold info."""
    if "registry" in sets:
        ref = [(idx[i][0], idx[i][1], s) for i, s in sets["registry"].items()]
    else:   # fall back: original benign tools among val/test negatives
        ref = [(idx[i][0], idx[i][1], s) for k in ("val_neg", "test_neg") for i, s in sets[k].items()
               if idx[i][3] == "benign" and not idx[i][4]]
    if method.startswith("rel_"):
        cal = RegistryCalibrator(ref, stat=method[4:])
        tf = lambda k: [v for v in (cal.transform(idx[i][0], idx[i][1], s) for i, s in sets.get(k, {}).items())
                        if v is not None]
        thr = threshold_at_fpr(tf("val_neg"), fpr)
    else:
        tf = lambda k: [logit(s) for s in sets.get(k, {}).values()]
        if method == "global":
            thr = threshold_at_fpr(tf("val_neg"), fpr)
        else:
            by = defaultdict(list)
            for i, s in sets["val_neg"].items():
                by[idx[i][0]].append(s)
            thr = worst_group_threshold(by, fpr)
    out = {}
    for k in ("test_neg", "test_hardneg") + POS_SETS:
        v = np.asarray(tf(k))
        if k in sets:
            out[k] = (float((v >= thr).mean()) if len(v) else None, int(len(v)), len(sets[k]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results/robust3")
    ap.add_argument("--out", default="results/calibration")
    a = ap.parse_args()
    run = Path(a.run)
    S = defaultdict(lambda: defaultdict(dict))
    for r in map(json.loads, open(run / "scores.jsonl")):
        if not r["defender"].startswith("B2"):          # judge scores are not calibrated here
            S[(r["defender"], r["seed"])][r["set"]][r["id"]] = r["score"]
    idx = item_index(run)
    res = defaultdict(lambda: defaultdict(list))        # (defender, method, fpr) -> set -> [rates over seeds]
    cover = {}
    for (d, seed), sets in sorted(S.items()):
        for m in METHODS:
            for fpr in (0.01, 0.05):
                for k, (rate, n, n_all) in evaluate(sets, idx, m, fpr).items():
                    if rate is not None:
                        res[(d, m, fpr)][k].append(rate)
                    cover[(m, k)] = (n, n_all)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(
        {f"{d}|{m}|{f}": {k: [float(np.mean(v)), float(np.std(v)), len(v)] for k, v in sets.items()}
         for (d, m, f), sets in res.items()}, indent=1))
    cols = ["test_neg", "test_hardneg"] + [k for k in POS_SETS if any(k in v for v in res.values())]
    lines = [f"# Threshold calibration ({run.name}); mean±std over seeds", "",
             "Columns: realized FPR on test negatives / hard negatives, detection on positive sets. "
             "Thresholds fixed on validation negatives only.", ""]
    for fpr in (0.01, 0.05):
        lines += [f"## Target {fpr:.0%} FPR", "", "| defender | method | " + " | ".join(cols) + " |",
                  "|---" * (len(cols) + 2) + "|"]
        for (d, m, f), sets in res.items():
            if f != fpr:
                continue
            cell = lambda k: (f"{np.mean(sets[k]):.3f}" + (f"±{np.std(sets[k]):.3f}" if len(sets[k]) > 1 else "")
                              if k in sets else "-")
            lines.append(f"| {d} | {m} | " + " | ".join(cell(k) for k in cols) + " |")
        lines.append("")
    lines += ["Coverage (items with a reference registry / all) for rel_* methods: " +
              ", ".join(f"{k} {cover[('rel_max', k)][0]}/{cover[('rel_max', k)][1]}" for k in cols
                        if ('rel_max', k) in cover)]
    (out / "calibration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
