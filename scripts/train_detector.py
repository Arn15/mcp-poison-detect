"""Step 3 / D1 pilot: B0 (TF-IDF+LR) vs B3 (plain DeBERTa-v3) vs context-aware DeBERTa-v3.

Trains on the original MCPTox train split, selects epochs on val, evaluates on
  * orig_test : frozen original test split
  * para_test : frozen MCPTox-Para test (paraphrases of the same test tools)
and, for the context model, with shuffled (wrong-server) context as a control.
Writes run_dir/{results.json, summary.md, scores.jsonl}.

Kaggle:  SCRIPT="scripts/train_detector.py"  ARGS="--tag d1_pilot"
Smoke:   python scripts/train_detector.py --set detector.backbone=hf-internal-testing/tiny-random-DebertaV2Model \
             detector.seeds=[1] detector.epochs=1 detector.max_steps=2 eval.bootstrap_iters=50 --tag smoke
"""
from __future__ import annotations

import json

import numpy as np
from common import setup_run

from mcptp.data.io import ensure_examples, load_para
from mcptp.detector.model import shuffled_contexts
from mcptp.detector.train import score, train_detector
from mcptp.eval.metrics import paired_bootstrap_diff, tpr_at_fpr
from mcptp.eval.report import evaluate


def fit_b0(train, field="description"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline

    feats = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                          ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True))])
    m = make_pipeline(feats, LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit([e[field] for e in train], [e["label"] for e in train])
    return lambda es: m.predict_proba([e[field] for e in es])[:, 1]


def fmt(x):
    return f"{x[0]:.3f} [{x[1]:.2f}, {x[2]:.2f}]"


def main():
    cfg, paths, run_dir, log = setup_run("train_detector", __doc__)
    dcfg, ecfg = cfg["detector"], cfg["eval"]
    fprs = [ecfg["target_fpr"], *ecfg.get("extra_fprs", [])]
    nb, seed0 = int(ecfg["bootstrap_iters"]), cfg["project"]["seed"]
    ex = ensure_examples(cfg, paths)
    train = [e for e in ex if e["split"] == "train"]
    val = [e for e in ex if e["split"] == "val"]
    sets = {"orig_test": [e for e in ex if e["split"] == "test"],
            "para_test": load_para(paths["splits"], cfg["para"]["version"], "test")}
    log.info("train %d  val %d  orig_test %d  para_test %d", len(train), len(val), *map(len, sets.values()))

    scores = {}   # (model, eval_set) -> list of per-seed score arrays
    val_scores = {}
    f = open(run_dir / "scores.jsonl", "w")

    def record(model, seed, name, es, s):
        scores.setdefault((model, name), []).append(s)
        for e, v in zip(es, s):
            f.write(json.dumps({"model": model, "seed": seed, "set": name, "id": e["id"], "label": e["label"],
                                "score": float(v)}) + "\n")

    b0 = fit_b0(train)
    val_scores["B0_tfidf_lr"] = [b0(val)]
    for name, es in sets.items():
        record("B0_tfidf_lr", 0, name, es, b0(es))

    variants = {"B3_plain_deberta": False, "ours_context_deberta": True}
    history = {}
    for vname, use_ctx in variants.items():
        for sd in dcfg.get("seeds", [seed0]):
            log.info("== %s seed %d", vname, sd)
            model, tok, hist, device = train_detector(train, val, dcfg, use_ctx, sd, log)
            history[f"{vname}/{sd}"] = hist
            val_scores.setdefault(vname, []).append(score(model, tok, val, dcfg, use_ctx, device))
            for name, es in sets.items():
                record(vname, sd, name, es, score(model, tok, es, dcfg, use_ctx, device))
                if use_ctx:
                    shuf = shuffled_contexts(es, seed=sd)
                    record("ours_context_SHUFFLED", sd, name, es, score(model, tok, es, dcfg, True, device, contexts=shuf))
            if use_ctx:
                val_scores.setdefault("ours_context_SHUFFLED", []).append(val_scores[vname][-1])
            del model
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    f.close()

    # ---- aggregate: seed-ensemble (mean score) with bootstrap CIs, plus per-seed mean/std of TPR@1% ----
    results = {"config": {k: dcfg[k] for k in ("backbone", "max_len", "lr", "batch_size", "epochs")},
               "seeds": dcfg.get("seeds", [seed0]), "history": history, "models": {}, "paired": {}}
    for (model, name), runs in sorted(scores.items()):
        es = sets[name]
        ens = np.mean(runs, axis=0)
        vs = np.mean(val_scores[model], axis=0)
        r = evaluate(es, ens, val, vs, fprs, nb, seed0)
        y = [e["label"] for e in es]
        per_seed = [tpr_at_fpr(y, s, fprs[0]) for s in runs]
        r["per_seed_tpr@1%"] = {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed)), "values": per_seed}
        results["models"].setdefault(model, {})[name] = r

    for name, es in sets.items():
        y = [e["label"] for e in es]
        ens = {m: np.mean(scores[(m, name)], axis=0) for m in ("ours_context_deberta", "B3_plain_deberta", "B0_tfidf_lr",
                                                                "ours_context_SHUFFLED")}
        for other in ("B3_plain_deberta", "B0_tfidf_lr", "ours_context_SHUFFLED"):
            for fp in fprs:
                results["paired"][f"{name}: ours - {other} @ TPR@{fp:g}"] = paired_bootstrap_diff(
                    lambda a, b, fp=fp: tpr_at_fpr(a, b, fp), y, ens["ours_context_deberta"], ens[other], nb, seed=seed0)

    (run_dir / "results.json").write_text(json.dumps(results, indent=1))

    lines = ["| model | set | AUROC | TPR@1%FPR | TPR@5%FPR | implicit TPR@1% | held-out family TPR@1% | HN-FPR (val thr 5%) |",
             "|---|---|---|---|---|---|---|---|"]
    for model, d in results["models"].items():
        for name, r in d.items():
            a = r["all"]
            lines.append(f"| {model} | {name} | {fmt(a['auroc'])} | {fmt(a['tpr@0.01fpr'])} | {fmt(a['tpr@0.05fpr'])} | "
                         f"{r['implicit']['tpr@0.01fpr'][0]:.3f} | {r['heldout_family']['tpr@0.01fpr'][0]:.3f} | "
                         f"{r['val_thr@0.05']['hard_neg_fpr']:.3f} |")
    lines += ["", "Paired bootstrap (seed-ensemble scores):", ""]
    for k, v in results["paired"].items():
        lines.append(f"- {k}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}], P(diff<=0)={v['p_le_0']:.3f}")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    log.info("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
