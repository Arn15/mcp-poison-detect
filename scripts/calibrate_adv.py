"""Step 9: the final system = adversarial training + registry-relative calibration, measured together.

Trains BASE and ADV twins (same data and recipe as scripts/adv_train.py, one seed) for detector.backbone, scores
every set with the logit margin, and reports at thresholds fixed on validation negatives (1% / 5% FPR) both the
GLOBAL threshold and registry-relative calibration (REL: r(c) = m(c) - max m(t) over the server's other registered
tools; mcptp.eval.calibration, margins clipped like the calibrator).

Sets
  test_neg / test_hardneg        realized false-alarm rates on unseen servers
  test_pos, <round>_opt          detection on test positives and control-verified black-box rewrites
  wb15_heldout_{evaders,genuine} white-box rewrites (15% budget, wb1 + wb1e5) of HELD-OUT test seeds (never trained on)
  wb30_{evaders,genuine}         white-box rewrites (30% budget) found against BASE twins (wb3_base_b30_*);
                                 not in ADV training (ADV saw only 15% rewrites), but mostly from train seeds
  wb30_heldout_genuine           the held-out-seed subset of wb30_genuine
  wb30adv_{evaders,genuine}      rewrites that evaded the ADV model itself (wb3_adv_b30_*)
White-box sets were found against other model instances, so detection on them measures transfer, not the
adaptive attack (results/wb3_adv_* already has that).

Kaggle (1 GPU, ~12 min per backbone):  SCRIPT="scripts/calibrate_adv.py"  ARGS="--tag cal_adv_deb"
E5: ARGS="--tag cal_adv_e5 --set detector.backbone=intfloat/multilingual-e5-base"
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from adv_train import verified_opt_sets
from common import setup_run
from defense_checks import CLIP

from mcptp.data.io import ensure_examples, load_para
from mcptp.eval.metrics import threshold_at_fpr
from mcptp.utils import atomic_write_json
from mcptp.utils.env import REPO_ROOT

DEFAULTS = {"seed": 42, "adv_sources": ["results/wb1", "results/wb1e5"], "train_rounds": ["r0", "r1", "r1b", "r2"],
            "heldout_rounds": ["r1t", "r2t"], "wb30": ["results/wb3_base_b30_deb", "results/wb3_base_b30_e5"],
            "wb30adv": ["results/wb3_adv_b30_deb", "results/wb3_adv_b30_e5"], "fprs": [0.01, 0.05]}


def ex_(r, desc):
    return {"server": r["server"], "tool_name": r["tool_name"], "description": desc, "label": 1, "context": [],
            "kind": "poison", "split": "train"}


def load_wb(run):
    p = REPO_ROOT / run
    if not (p / "whitebox.jsonl").exists():
        return [], set()
    W = [json.loads(l) for l in open(p / "whitebox.jsonl")]
    gen = set()
    if (p / "wb_verified.jsonl").exists():
        gen = {v["key"] for v in map(json.loads, open(p / "wb_verified.jsonl")) if v["evaded"] and v["asr_true"] >= 0.2}
    return W, gen


def main():
    cfg, paths, run_dir, log = setup_run("calibrate_adv", __doc__)
    c = {**DEFAULTS, **(cfg.get("caladv") or {})}
    ex = ensure_examples(cfg, paths)
    split_of = {e["id"]: e["split"] for e in ex}
    v = cfg["para"]["version"]
    para = {s: load_para(paths["splits"], v, s) for s in ("train", "val", "test")}
    train = [e for e in ex if e["split"] == "train"] + para["train"]
    val = [e for e in ex if e["split"] == "val"] + para["val"]
    test = [e for e in ex if e["split"] == "test"] + para["test"]

    # adversarial training data: identical rule to adv_train.py (train-seed 15% rewrites, deduped, label 1)
    adv, seen = [], {e["description"] for e in train}
    sets = defaultdict(list)
    for run in c["adv_sources"]:
        W, gen = load_wb(run)
        for r in W:
            if r["n_changed"] == 0:
                continue
            if r["round"] in c["train_rounds"]:
                assert split_of.get(r["seed_id"]) == "train", r["seed_id"]
                if r["adv_desc"] not in seen:
                    seen.add(r["adv_desc"]); adv.append(ex_(r, r["adv_desc"]))
            elif r["round"] in c["heldout_rounds"] and r["evaded"]:
                sets["wb15_heldout_evaders"].append(ex_(r, r["adv_desc"]))
                if r["key"] in gen:
                    sets["wb15_heldout_genuine"].append(ex_(r, r["adv_desc"]))
    suffix = "e5" if "e5" in cfg["detector"]["backbone"].lower() else "deb"   # 30% sets found against THIS backbone
    for name, runs in (("wb30", c["wb30"]), ("wb30adv", c["wb30adv"])):
        for run in [r for r in runs if r.endswith(suffix)]:
            W, gen = load_wb(run)
            for r in W:
                if r["evaded"]:
                    sets[f"{name}_evaders"].append(ex_(r, r["adv_desc"]))
                    if r["key"] in gen:
                        sets[f"{name}_genuine"].append(ex_(r, r["adv_desc"]))
                        if name == "wb30" and r["round"] in c["heldout_rounds"]:
                            sets["wb30_heldout_genuine"].append(ex_(r, r["adv_desc"]))
    base_sets = {"val_neg": [e for e in val if e["label"] == 0], "test_neg": [e for e in test if e["label"] == 0],
                 "test_hardneg": [e for e in test if e["label"] == 0 and e.get("is_hard_negative")],
                 "test_pos": [e for e in test if e["label"] == 1], **verified_opt_sets()}
    sets = {**base_sets, **dict(sets)}
    log.info("adversarial rewrites: %d | sets: %s", len(adv), {k: len(x) for k, x in sets.items()})

    reg = [e for e in ex if e["kind"] == "benign"]
    from mcptp.detector.train import score, train_detector
    regs = defaultdict(list)
    for e in reg:
        regs[e["server"]].append({"tool_name": e["tool_name"], "description": e["description"]})
    dcfg = cfg["detector"]
    out = {"backbone": dcfg["backbone"], "n_adv": len(adv), "sizes": {k: len(x) for k, x in sets.items()}}
    for variant, tr in (("base", train), ("adv", train + adv)):
        log.info("training %s (%d examples, seed %d)", variant, len(tr), c["seed"])
        m, tok, _, dev = train_detector(tr, val, dcfg, False, int(c["seed"]), log, registries=dict(regs))
        M = lambda xs: np.clip(score(m, tok, xs, dcfg, False, dev, margin=True), -CLIP, CLIP) if xs else np.array([])
        ms = {k: M(x) for k, x in sets.items()}
        ref = defaultdict(list)
        for e, s in zip(reg, M(reg)):
            ref[e["server"]].append((e["tool_name"], float(s)))

        def rel(xs, mm):
            o = []
            for e, s in zip(xs, mm):
                rr = [val_ for n, val_ in ref.get(e["server"], []) if n != e["tool_name"]]
                o.append(s - max(rr) if rr else np.nan)
            return np.asarray(o, dtype=float)

        rs = {k: rel(sets[k], ms[k]) for k in sets}
        res = {}
        for fp in c["fprs"]:
            tg = threshold_at_fpr(ms["val_neg"], fp)
            tr_ = threshold_at_fpr(rs["val_neg"][~np.isnan(rs["val_neg"])], fp)
            for k in sets:
                if k == "val_neg" or not len(ms[k]):
                    continue
                # no registry for the server -> fall back to the global rule (as a deployment would)
                flag_rel = np.where(np.isnan(rs[k]), ms[k] >= tg, rs[k] >= tr_)
                res[f"{k}@{fp:g}"] = {"global": float((ms[k] >= tg).mean()), "rel": float(flag_rel.mean())}
        out[variant] = res
        log.info("%s @1%%: %s", variant, {k: v_ for k, v_ in res.items() if k.endswith("@0.01")})
        m = None
        try:
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
    atomic_write_json(out, run_dir / "cal_adv.json")
    L = [f"# Adversarial training x registry-relative calibration ({dcfg['backbone']}, seed {c['seed']})", "",
         "Flag rates at thresholds fixed on val negatives. FPR rows (test_neg, test_hardneg): lower is better.", ""]
    for fp in c["fprs"]:
        L += [f"## Target {fp:.0%}", "", "| set | n | base global | base rel | adv global | **adv rel** |",
              "|---|---|---|---|---|---|"]
        for k in sets:
            key = f"{k}@{fp:g}"
            if key in out["base"]:
                b, a = out["base"][key], out["adv"][key]
                L.append(f"| {k} | {len(sets[k])} | {b['global']:.3f} | {b['rel']:.3f} | {a['global']:.3f} | **{a['rel']:.3f}** |")
        L.append("")
    (run_dir / "cal_adv.md").write_text("\n".join(L) + "\n")
    log.info("\n%s", "\n".join(L))


if __name__ == "__main__":
    main()
