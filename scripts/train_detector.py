"""Step 3: detector training and evaluation (B0, B3 plain DeBERTa, context-aware DeBERTa; +/- paraphrase aug).

Variants come from config detector.variants. Train on the original train split (+ train paraphrases when
train_para=true). Select epochs on val = original val + paraphrased val (when frozen), tie-break by val loss.
Evaluate on
  * orig_test : frozen original test split
  * para_test : frozen MCPTox-Para test
and, for context variants, with shuffled (wrong-server) context as a control ("<variant>/SHUFFLED").
Writes run_dir/{results.json, summary.md, scores.jsonl}.

Kaggle:  SCRIPT="scripts/train_detector.py"  ARGS="--tag d1b"
Subset:  --set 'detector.only=[B3_plain,ours_context]'
"""
from __future__ import annotations

import json
from collections import defaultdict

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


def try_load_para(paths, cfg, split):
    try:
        return load_para(paths["splits"], cfg["para"]["version"], split)
    except (FileNotFoundError, KeyError):
        return []


def main():
    cfg, paths, run_dir, log = setup_run("train_detector", __doc__)
    dcfg, ecfg = cfg["detector"], cfg["eval"]
    fprs = [ecfg["target_fpr"], *ecfg.get("extra_fprs", [])]
    nb, seed0 = int(ecfg["bootstrap_iters"]), cfg["project"]["seed"]
    ex = ensure_examples(cfg, paths)
    train = [e for e in ex if e["split"] == "train"]
    para_train, para_val = try_load_para(paths, cfg, "train"), try_load_para(paths, cfg, "val")
    val = [e for e in ex if e["split"] == "val"] + para_val
    sets = {"orig_test": [e for e in ex if e["split"] == "test"], "para_test": try_load_para(paths, cfg, "test")}
    registries = defaultdict(list)
    for e in ex:
        if e["kind"] == "benign":
            registries[e["server"]].append({"tool_name": e["tool_name"], "description": e["description"]})
    log.info("train %d (+%d para)  val %d (incl. %d para)  orig_test %d  para_test %d", len(train), len(para_train),
             len(val), len(para_val), len(sets["orig_test"]), len(sets["para_test"]))

    variants = dcfg["variants"]
    if dcfg.get("only"):
        variants = {k: v for k, v in variants.items() if k in dcfg["only"]}
    if any(v.get("train_para") for v in variants.values()) and not para_train:
        log.warning("no frozen train paraphrases: skipping +para variants")
        variants = {k: v for k, v in variants.items() if not v.get("train_para")}

    scores, val_scores, history = {}, {}, {}
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

    for vname, v in variants.items():
        use_ctx = bool(v["use_context"])
        tr = train + (para_train if v.get("train_para") else [])
        for sd in dcfg.get("seeds", [seed0]):
            log.info("== %s seed %d (n_train=%d)", vname, sd, len(tr))
            model, tok, hist, device = train_detector(tr, val, dcfg, use_ctx, sd, log, registries=registries)
            history[f"{vname}/{sd}"] = hist
            sv = score(model, tok, val, dcfg, use_ctx, device)
            val_scores.setdefault(vname, []).append(sv)
            for name, es in sets.items():
                record(vname, sd, name, es, score(model, tok, es, dcfg, use_ctx, device))
                if use_ctx:
                    shuf = shuffled_contexts(es, seed=sd)
                    record(f"{vname}/SHUFFLED", sd, name, es, score(model, tok, es, dcfg, True, device, contexts=shuf))
            if use_ctx:
                val_scores.setdefault(f"{vname}/SHUFFLED", []).append(sv)
            del model
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    f.close()

    results = {"config": {k: dcfg[k] for k in ("backbone", "max_len", "lr", "batch_size", "epochs",
                                                "context_dropout", "context_swap")},
               "variants": variants, "seeds": dcfg.get("seeds", [seed0]), "n_val": len(val),
               "history": history, "models": {}, "paired": {}}
    for (model, name), runs in sorted(scores.items()):
        es = sets[name]
        ens = np.mean(runs, axis=0)
        r = evaluate(es, ens, val, np.mean(val_scores[model], axis=0), fprs, nb, seed0)
        y = [e["label"] for e in es]
        per_seed = [tpr_at_fpr(y, s, fprs[0]) for s in runs]
        r["per_seed_tpr@1%"] = {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed)), "values": per_seed}
        results["models"].setdefault(model, {})[name] = r

    for name, es in sets.items():
        y = [e["label"] for e in es]
        for a, b in dcfg.get("compare", []):
            if (a, name) not in scores or (b, name) not in scores:
                continue
            sa, sb = np.mean(scores[(a, name)], axis=0), np.mean(scores[(b, name)], axis=0)
            for fp in fprs:
                results["paired"][f"{name}: {a} - {b} @ TPR@{fp:g}"] = paired_bootstrap_diff(
                    lambda yy, ss, fp=fp: tpr_at_fpr(yy, ss, fp), y, sa, sb, nb, seed=seed0)

    (run_dir / "results.json").write_text(json.dumps(results, indent=1))
    lines = ["| model | set | AUROC | TPR@1%FPR | TPR@5%FPR | implicit TPR@1% | held-out family TPR@1% | "
             "test FPR @ val thr 5% | HN-FPR @ val thr 5% | seed std TPR@1% |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for model, d in results["models"].items():
        for name, r in d.items():
            a, t = r["all"], r["val_thr@0.05"]
            lines.append(f"| {model} | {name} | {fmt(a['auroc'])} | {fmt(a['tpr@0.01fpr'])} | {fmt(a['tpr@0.05fpr'])} | "
                         f"{r['implicit']['tpr@0.01fpr'][0]:.3f} | {r['heldout_family']['tpr@0.01fpr'][0]:.3f} | "
                         f"{t['fpr']:.3f} | {t['hard_neg_fpr']:.3f} | {r['per_seed_tpr@1%']['std']:.3f} |")
    lines += ["", "Paired bootstrap on seed-ensemble scores (a - b, 95% CI, P(diff<=0)):", ""]
    for k, v in results["paired"].items():
        lines.append(f"- {k}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}], P={v['p_le_0']:.3f}")
    lines += ["", "Best epochs: " + ", ".join(f"{k}: ep{max(range(len(h)), key=lambda i: (h[i]['val_auroc'] + h[i]['val_tpr@5'], -h[i]['val_loss']))}"
                                         for k, h in history.items())]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    log.info("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
