"""Step 7: adversarial training against the white-box attacker, evaluated against an un-hardened twin.

Adversarial data = the final rewrites of the white-box attack (results/<run>/whitebox.jsonl, Step 6) on TRAIN-split
seeds only (rounds r0/r1/r1b/r2). They are added to the training set as poisoned (label 1). Rewrites of held-out
test seeds (r1t/r2t) never enter training; they are the held-out adversarial test set.

Per seed: train BASE (train + para, as before) and ADV (train + para + adversarial rewrites) with the same seed and
hyperparameters, then score both at thresholds fixed on validation negatives (1% / 5% FPR):
  test_neg / test_hardneg     realized FPR on unseen-server negatives / benign hard negatives
  test_pos                    detection on test positives (orig + para)
  <round>_opt                 control-verified optimized rewrites (black-box rounds, as in robust3/5)
  wb_heldout_<src>            white-box rewrites of held-out test seeds found against the base detector (transfer)
  wb_heldout_genuine          the subset of those that genuinely evaded the base detector
The seed-42 ADV checkpoint is saved to <run_dir>/ckpt_adv so the white-box attack can be rerun against it:
  python scripts/attack_whitebox.py --tag wb2_adv --set whitebox.detector_ckpt=<run_dir>/ckpt_adv

Kaggle (1 GPU is enough; ~5 min per model):  SCRIPT="scripts/adv_train.py"  ARGS="--tag adv1"
E5:  ARGS="--tag adv1e5 --set detector.backbone=intfloat/multilingual-e5-base"
"""
from __future__ import annotations

import json

import numpy as np
from common import setup_run

from mcptp.data.io import ensure_examples, load_para
from mcptp.eval.metrics import auroc, threshold_at_fpr
from mcptp.utils import atomic_write_json
from mcptp.utils.env import REPO_ROOT

DEFAULTS = {"sources": ["results/wb1/whitebox.jsonl", "results/wb1e5/whitebox.jsonl"], "seeds": [42, 43, 44],
            "only_evaded": False, "train_rounds": ["r0", "r1", "r1b", "r2"], "heldout_rounds": ["r1t", "r2t"],
            "save_seed": 42}


def wb_rows(src):
    """Rows of a white-box run's whitebox.jsonl (empty list if missing)."""
    p = REPO_ROOT / src
    return [json.loads(l) for l in open(p)] if p.exists() else []


def verified_opt_sets(floor=0.2):
    """{<round>_opt: examples} of control-verified black-box rewrites (data/attacks/verified.jsonl)."""
    out, seen = {}, set()
    for r in map(json.loads, open(REPO_ROOT / "data" / "attacks" / "verified.jsonl")):
        if r["kind"] != "opt" or r["asr_true"] < floor or (r["round"], r["desc"]) in seen:
            continue
        seen.add((r["round"], r["desc"]))
        out.setdefault(f"{r['round']}_opt", []).append(as_example(r, r["desc"], r["round"], len(seen)))
    return out


def as_example(r, desc, tag, k):
    """Turn a white-box rewrite into a poisoned training/eval example."""
    return {"id": f"{tag}_{k}", "server": r["server"], "tool_name": r["tool_name"], "description": desc,
            "description_stripped": desc, "label": 1, "context": [], "kind": "poison", "split": "train"}


def main():
    """Entry point (see the module docstring for what this script does and how to run it)."""
    cfg, paths, run_dir, log = setup_run("adv_train", __doc__)
    c = {**DEFAULTS, **(cfg.get("advtrain") or {})}
    ex = ensure_examples(cfg, paths)
    split_of = {e["id"]: e["split"] for e in ex}
    v = cfg["para"]["version"]
    para = {s: load_para(paths["splits"], v, s) for s in ("train", "val", "test")}
    train = [e for e in ex if e["split"] == "train"] + para["train"]
    val = [e for e in ex if e["split"] == "val"] + para["val"]

    # ---------------------------------------------------------------- adversarial training data (train seeds only)
    adv, seen = [], {e["description"] for e in train}
    heldout = {}
    for src in c["sources"]:
        rows = wb_rows(src)
        if not rows:
            log.warning("missing %s", src)
            continue
        name = src.split("/")[-2]
        vpath = REPO_ROOT / src.replace("whitebox.jsonl", "wb_verified.jsonl")
        vrows = [json.loads(l) for l in open(vpath)] if vpath.exists() else []
        gen = {x["key"] for x in vrows if x["evaded"] and x["asr_true"] >= 0.2}
        for r in rows:
            if r["n_changed"] == 0:
                continue
            if r["round"] in c["train_rounds"]:
                if split_of.get(r["seed_id"]) != "train":
                    raise ValueError(f"train-round row with non-train seed {r['seed_id']}")
                if (c["only_evaded"] and not r["evaded"]) or r["adv_desc"] in seen:
                    continue
                seen.add(r["adv_desc"])
                adv.append(as_example(r, r["adv_desc"], f"adv_{name}", len(adv)))
            elif r["round"] in c["heldout_rounds"]:
                assert split_of.get(r["seed_id"]) == "test", r["seed_id"]
                heldout.setdefault(f"wb_heldout_{name}", []).append(as_example(r, r["adv_desc"], name, len(heldout)))
                if r["key"] in gen:
                    heldout.setdefault("wb_heldout_genuine", []).append(as_example(r, r["adv_desc"], "gen", 0))
    log.info("adversarial training rewrites (train seeds): %d | held-out sets: %s", len(adv),
             {k: len(x) for k, x in heldout.items()})

    # ---------------------------------------------------------------- eval sets
    test_all = [e for e in ex + para["test"] if e["split"] == "test"]
    sets = {"val_neg": [e for e in val if e["label"] == 0],
            "test_neg": [e for e in test_all if e["label"] == 0],
            "test_hardneg": [e for e in test_all if e["label"] == 0 and e.get("is_hard_negative")],
            "test_pos": [e for e in test_all if e["label"] == 1],
            **verified_opt_sets(), **heldout}
    log.info("eval sets: %s", {k: len(x) for k, x in sets.items()})

    from mcptp.detector.train import save_detector, score, train_detector
    regs = {}
    for e in ex:
        if e["kind"] == "benign":
            regs.setdefault(e["server"], []).append({"tool_name": e["tool_name"], "description": e["description"]})
    dcfg = cfg["detector"]
    results = {}
    for seed in c["seeds"]:
        for variant, tr in (("base", train), ("adv", train + adv)):
            log.info("seed %d | %s | %d training examples", seed, variant, len(tr))
            m, tok, _, dev = train_detector(tr, val, dcfg, False, seed, log, registries=regs)
            sc = {k: score(m, tok, x, dcfg, False, dev, margin=True) for k, x in sets.items()}
            row = {}
            for fp in (0.01, 0.05):
                thr = threshold_at_fpr(sc["val_neg"], fp)
                for k, s in sc.items():
                    if k != "val_neg":
                        row[f"{k}@{fp:g}"] = float(np.mean(np.asarray(s) >= thr))
            for k in sets:
                if k not in ("val_neg", "test_neg", "test_hardneg"):
                    row[f"auroc_{k}"] = float(auroc(np.r_[np.zeros(len(sc["test_neg"])), np.ones(len(sc[k]))],
                                                    np.r_[sc["test_neg"], sc[k]]))
            results.setdefault(variant, {})[seed] = row
            log.info("  %s", {k: round(x, 3) for k, x in row.items() if k.endswith("@0.01")})
            if variant == "adv" and seed == c["save_seed"]:
                save_detector(m, tok, dcfg, False, run_dir / "ckpt_adv")
                log.info("saved %s", run_dir / "ckpt_adv")
            del m
            try:
                import gc, torch
                gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass

    # ---------------------------------------------------------------- summary
    keys = sorted(next(iter(results["base"].values())).keys())
    summ = {var: {k: [float(np.mean([r[k] for r in rs.values()])), float(np.std([r[k] for r in rs.values()]))]
                  for k in keys} for var, rs in results.items()}
    atomic_write_json({"n_adv": len(adv), "sets": {k: len(x) for k, x in sets.items()}, "per_seed": results,
                       "mean_std": summ, "backbone": dcfg["backbone"]}, run_dir / "adv_results.json")
    lines = [f"# Adversarial training ({dcfg['backbone']}, seeds {c['seeds']}, {len(adv)} adversarial rewrites)", "",
             "Rates at thresholds fixed on val negatives (mean ± std over seeds). FPR rows: lower is better.", "",
             "| set | base @1% | adv @1% | base @5% | adv @5% |", "|---|---|---|---|---|"]
    for k in [k for k in sets if k != "val_neg"]:
        f = lambda var, fp: "{:.3f} ± {:.3f}".format(*summ[var][f"{k}@{fp}"])
        lines.append(f"| {k} (n={len(sets[k])}) | {f('base', '0.01')} | {f('adv', '0.01')} | {f('base', '0.05')} | {f('adv', '0.05')} |")
    (run_dir / "adv_results.md").write_text("\n".join(lines) + "\n")
    log.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
