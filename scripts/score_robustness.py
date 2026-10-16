"""Robustness comparison across defenders (milestone core table).

Defenders: B0 TF-IDF+LR, B1 E5 classifier (+para), B3 plain DeBERTa (+para), ours context DeBERTa (+para),
B2 LLM judge (Qwen3-8B 4-bit, P(Yes)). Trained models are trained in-run (seed 42) - no Kaggle inputs needed.

Eval sets
  val_neg   : val negatives (orig + para)            -> thresholds at 1% / 5% FPR
  test_neg  : test negatives (orig + para)           -> FPR at those thresholds (+ hard-negative FPR)
  test_pos  : test positives (orig + para)           -> detection on non-adaptive attacks
  <round>_opt / <round>_orig : working optimized rewrites from data/attacks/<round> and the same seeds' originals
                               (r0, r1 = train seeds; r1t = held-out test seeds)

Kaggle: SCRIPT="scripts/score_robustness.py"  ARGS="--tag robust"  GPU_DEPS=True  (~35-45 min)
Skip the judge: --set score.judge=false
"""
from __future__ import annotations

import gc
import json
from collections import defaultdict

import numpy as np
from common import setup_run

from mcptp.data.context import find_references
from mcptp.data.io import ensure_examples, load_para
from mcptp.eval.metrics import threshold_at_fpr
from mcptp.utils.env import REPO_ROOT

ROUNDS = ("r0", "r1", "r1t")


def attack_sets(registries, top_k, floor=0.2):
    """{round_opt: [...], round_orig: [...]} built from committed attack runs."""
    sets = {}
    for rnd in ROUNDS:
        d = REPO_ROOT / "data" / "attacks" / rnd
        if not (d / "attacks.jsonl").exists():
            continue
        seeds = {a["seed_id"]: a for a in map(json.loads, open(d / "attacks.jsonl")) if a}
        opt, seen, orig_ids = [], set(), set()
        for t in map(json.loads, open(d / "trajectory.jsonl")):
            if (t.get("asr") or 0) < floor or t["desc"] in seen or t["seed_id"] not in seeds:
                continue
            seen.add(t["desc"]); orig_ids.add(t["seed_id"])
            opt.append((seeds[t["seed_id"]], t["desc"]))
        orig = [(seeds[i], seeds[i]["seed_desc"]) for i in sorted(orig_ids)]
        for name, items in ((f"{rnd}_opt", opt), (f"{rnd}_orig", orig)):
            exs = []
            for k, (a, text) in enumerate(items):
                reg = registries.get(a["server"], {})
                refs, m = find_references(text, reg, exclude=a["tool_name"], top_k=top_k)
                exs.append({"id": f"{name}_{k}", "server": a["server"], "tool_name": a["tool_name"],
                            "description": text, "description_stripped": text, "label": 1,
                            "context": [{"tool_name": r, "description": reg[r]} for r in refs],
                            "referenced_tools": refs, "ref_method": m, "registry_names": sorted(reg)})
            sets[name] = exs
    return sets


def rates(scores_by_set, val_neg_key="val_neg"):
    """Detection / FPR per set at thresholds fixed on validation negatives."""
    out = {}
    vs = scores_by_set[val_neg_key]
    for fp in (0.01, 0.05):
        thr = threshold_at_fpr(vs, fp)
        out[f"thr@{fp:g}"] = float(thr)
        for name, s in scores_by_set.items():
            if name == val_neg_key:
                continue
            out[f"{name}@{fp:g}"] = float((np.asarray(s) >= thr).mean()) if len(s) else None
    return out


def main():
    cfg, paths, run_dir, log = setup_run("score_robustness", __doc__)
    sc = dict(cfg.get("score") or {})
    ex = ensure_examples(cfg, paths)
    v = cfg["para"]["version"]
    para = {s: load_para(paths["splits"], v, s) for s in ("val", "test")}
    registries = defaultdict(dict)
    for e in ex:
        if e["kind"] == "benign":
            registries[e["server"]][e["tool_name"]] = e["description"]
    for e in ex + para["val"] + para["test"]:
        e.setdefault("registry_names", sorted(registries.get(e["server"], {})))

    sets = {
        "val_neg": [e for e in ex if e["split"] == "val" and e["label"] == 0] + [e for e in para["val"] if e["label"] == 0],
        "test_neg": [e for e in ex if e["split"] == "test" and e["label"] == 0] + [e for e in para["test"] if e["label"] == 0],
        "test_hardneg": [e for e in ex + para["test"] if e["split"] == "test" and e["label"] == 0 and e.get("is_hard_negative")],
        "test_pos": [e for e in ex if e["split"] == "test" and e["label"] == 1] + [e for e in para["test"] if e["label"] == 1],
    }
    sets.update(attack_sets(registries, cfg["context"]["top_k"]))
    log.info("eval sets: %s", {k: len(v_) for k, v_ in sets.items()})

    results = {}

    # B0: TF-IDF + LR on the original train split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline
    tr = [e for e in ex if e["split"] == "train"]
    b0 = make_pipeline(FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                                     ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                                           sublinear_tf=True))]),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    b0.fit([e["description"] for e in tr], [e["label"] for e in tr])
    results["B0_tfidf_lr"] = rates({k: b0.predict_proba([e["description"] for e in s])[:, 1] if s else []
                                    for k, s in sets.items()})
    log.info("B0 done")

    # trained classifiers (in-run, seed 42), scored one at a time to keep GPU memory low
    from mcptp.detector.train import LoadedDetector, train_default_detector
    models = [("B1_e5+para", "intfloat/multilingual-e5-base", False),
              ("B3_plain_deberta+para", cfg["detector"]["backbone"], False),
              ("ours_context_deberta+para", cfg["detector"]["backbone"], True)]
    for name, backbone, use_ctx in models:
        if name in (sc.get("skip") or []):
            continue
        c = {**cfg, "detector": {**cfg["detector"], "backbone": backbone}}
        log.info("training %s", name)
        ck = train_default_detector(c, paths, use_ctx, run_dir / f"ckpt_{name}", log)
        det = LoadedDetector(ck)
        results[name] = rates({k: det.score(s) if s else [] for k, s in sets.items()})
        log.info("%s done", name)
        del det
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # B2: LLM judge (needs GPU + bitsandbytes)
    if sc.get("judge", True):
        from mcptp.baselines.llm_judge import LLMJudge
        judge = LLMJudge(sc.get("judge_model", "Qwen/Qwen3-8B"), device=sc.get("judge_device", "cuda:0"),
                         quant="4bit", batch_size=int(sc.get("judge_batch", 8)))
        results["B2_llm_judge"] = rates({k: judge.score(s) if s else [] for k, s in sets.items()})
        log.info("B2 done")

    out = {"n": {k: len(v_) for k, v_ in sets.items()}, "results": results}
    (run_dir / "robustness.json").write_text(json.dumps(out, indent=1))
    cols = ["test_pos", "test_neg", "test_hardneg"] + [f"{r}_{t}" for r in ROUNDS for t in ("orig", "opt") if f"{r}_{t}" in sets]
    lines = ["Rates at thresholds fixed on validation negatives. *_neg / hardneg columns are FPR; all others are detection.", ""]
    for fp in ("0.01", "0.05"):
        lines += [f"### @ {float(fp):.0%} val FPR", "", "| defender | " + " | ".join(cols) + " |",
                  "|---" * (len(cols) + 1) + "|"]
        for name, r in results.items():
            lines.append(f"| {name} | " + " | ".join(
                "-" if r.get(f"{c}@{float(fp):g}") is None else f"{r[f'{c}@{float(fp):g}']:.3f}" for c in cols) + " |")
        lines.append("")
    lines.append("n per set: " + json.dumps(out["n"]))
    (run_dir / "robustness.md").write_text("\n".join(lines) + "\n")
    log.info("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
