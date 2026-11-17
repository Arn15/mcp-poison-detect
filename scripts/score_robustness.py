"""Robustness comparison across defenders (milestone core table).

Defenders: B0 TF-IDF+LR, B1 E5 classifier (+para), B3 plain DeBERTa (+para), ours context DeBERTa (+para),
B2 LLM judge (Qwen3-8B 4-bit, P(Yes)). Trained models are trained in-run per seed (score.seeds, default [42,43,44]) - no Kaggle inputs needed.
Reports: val-fixed thresholds (deployable), matched TEST-FPR thresholds (oracle, equal FPR), AUROC; writes
per-example scores to scores.jsonl.

Eval sets
  val_neg   : val negatives (orig + para)            -> thresholds at 1% / 5% FPR
  test_neg  : test negatives (orig + para)           -> FPR at those thresholds (+ hard-negative FPR)
  test_pos  : test positives (orig + para)           -> detection on non-adaptive attacks
  <round>_opt / <round>_orig : working optimized rewrites from data/attacks/<round> and the same seeds' originals
                               (r0, r1 = train seeds; r1t = held-out test seeds)

Kaggle: SCRIPT="scripts/score_robustness.py"  ARGS="--tag robust2"  GPU_DEPS=True  (~1.5-2 h for 3 seeds)
One seed: --set score.seeds=[42]
Registry reference scores (set "registry", all servers' benign tools) are written by default for scripts/calibrate.py.
Skip the judge: --set score.judge=false

Judge comparison (robust4: 8B/14B x zero/few-shot; classifiers skipped, B0 kept as reference):
  ARGS="--tag robust4 --config configs/judges.yaml"  GPU_DEPS=True
  One judge per notebook: --set 'score.only_judges=[B2_judge_14b_fs]'
  Judge scores are Yes/No logit margins (unbounded) unless a judge sets score: prob.
"""
from __future__ import annotations

import gc
import json
from collections import defaultdict

import numpy as np
from common import setup_run

from mcptp.data.context import find_references
from mcptp.data.io import ensure_examples, load_para
from mcptp.eval.metrics import auroc, threshold_at_fpr
from mcptp.utils.env import REPO_ROOT

ROUNDS = ("r0", "r1", "r1b", "r1t", "r2", "r2t")


def load_verified(path=None):
    """data/attacks/verified.jsonl (scripts/verify_attacks.py) -> {(round, kind, desc): asr_true}."""
    path = path or REPO_ROOT / "data" / "attacks" / "verified.jsonl"
    if not path.exists():
        return None
    return {(r["round"], r["kind"], r["desc"]): r["asr_true"] for r in map(json.loads, open(path))}


def attack_base_ids():
    """Seed ids used as bases for any attack round (excluded from few-shot judge demos)."""
    ids = set()
    for rnd in ROUNDS:
        f = REPO_ROOT / "data" / "attacks" / rnd / "attacks.jsonl"
        if f.exists():
            ids |= {a["seed_id"] for a in map(json.loads, open(f)) if a}
    return ids


def attack_sets(registries, top_k, floor=0.2, verified=None):
    """{round_opt: [...], round_orig: [...]} built from committed attack runs.
    verified: if given, keep only rewrites whose control-checked ASR (asr_true) >= floor."""
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
            if verified is not None and verified.get((rnd, "opt", t["desc"]), 0.0) < floor:
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
            if name in (val_neg_key, "registry"):
                continue
            out[f"{name}@{fp:g}"] = float((np.asarray(s) >= thr).mean()) if len(s) else None
    return out




def matched_rates(scores_by_set, neg_key="test_neg"):
    """Oracle operating point: threshold set on TEST negatives so every defender sits at the same realized FPR."""
    out = {}
    ns = np.asarray(scores_by_set[neg_key])
    for fp in (0.01, 0.05):
        thr = threshold_at_fpr(ns, fp)
        out[f"thr@{fp:g}"] = float(thr)
        for name, s in scores_by_set.items():
            if name in (neg_key, "val_neg", "registry"):
                continue
            out[f"{name}@{fp:g}"] = float((np.asarray(s) >= thr).mean()) if len(s) else None
    return out


def aurocs(scores_by_set, neg_key="test_neg"):
    """AUROC of each positive set against test negatives (threshold-free)."""
    ns = np.asarray(scores_by_set[neg_key])
    out = {}
    for name, s in scores_by_set.items():
        if name in (neg_key, "val_neg", "test_hardneg", "registry") or not len(s):
            continue
        y = np.r_[np.zeros(len(ns)), np.ones(len(s))]
        out[name] = float(auroc(y, np.r_[ns, np.asarray(s)]))
    hn = scores_by_set.get("test_hardneg")
    if hn is not None and len(hn) and len(scores_by_set.get("test_pos", [])):
        s = np.asarray(scores_by_set["test_pos"])
        out["test_pos_vs_hardneg"] = float(auroc(np.r_[np.zeros(len(hn)), np.ones(len(s))], np.r_[np.asarray(hn), s]))
    return out


def summarize_runs(runs):
    """runs: list of per-seed metric dicts -> {key: [mean, std, n]}."""
    keys = sorted({k for r in runs for k in r})
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        out[k] = [float(np.mean(vals)), float(np.std(vals)), len(vals)] if vals else None
    return out


def main():
    """Entry point (see the module docstring for what this script does and how to run it)."""
    cfg, paths, run_dir, log = setup_run("score_robustness", __doc__)
    sc = dict(cfg.get("score") or {})
    seeds = [int(s) for s in (sc.get("seeds") or [42, 43, 44])]
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
    if sc.get("registry", True):   # every server's original benign tools -> reference scores for scripts/calibrate.py
        sets["registry"] = [e for e in ex if e["kind"] == "benign"]
    verified = load_verified() if sc.get("verified", True) else None
    log.info("attack sets: %s", "control-verified only (data/attacks/verified.jsonl)" if verified else "raw judge ASR")
    sets.update(attack_sets(registries, cfg["context"]["top_k"], verified=verified))
    n_sets = {k: len(v_) for k, v_ in sets.items()}
    log.info("eval sets: %s | seeds %s", n_sets, seeds)

    per_seed = defaultdict(list)   # defender -> [{"val": .., "matched": .., "auroc": ..}, ...]
    scores_f = open(run_dir / "scores.jsonl", "w")

    def record(name, seed, scores_by_set):
        scores_by_set = {k: np.asarray(s, dtype=float) for k, s in scores_by_set.items()}
        for k, s in scores_by_set.items():
            for i, (e, x) in enumerate(zip(sets[k], s)):
                scores_f.write(json.dumps({"defender": name, "seed": seed, "set": k,
                                           "id": e.get("id", f"{k}_{i}"), "score": float(x)}) + "\n")
        scores_f.flush()
        m = {"val": rates(scores_by_set), "matched": matched_rates(scores_by_set), "auroc": aurocs(scores_by_set)}
        per_seed[name].append({"seed": seed, **m})
        log.info("%s seed %s: matched@1%% %s | auroc %s", name, seed,
                 {k: round(x, 3) for k, x in m["matched"].items() if k.endswith("@0.01") and x is not None},
                 {k: round(x, 3) for k, x in m["auroc"].items()})

    # B0: TF-IDF + LR on the original train split (deterministic -> one run)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline
    tr = [e for e in ex if e["split"] == "train"]
    b0 = make_pipeline(FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                                     ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                                           sublinear_tf=True))]),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    b0.fit([e["description"] for e in tr], [e["label"] for e in tr])
    record("B0_tfidf_lr", 0, {k: b0.predict_proba([e["description"] for e in s])[:, 1] if s else []
                              for k, s in sets.items()})

    # trained classifiers: one per seed, scored one at a time to keep GPU memory and disk low
    models = [("B1_e5+para", "intfloat/multilingual-e5-base", False),
              ("B3_plain_deberta+para", cfg["detector"]["backbone"], False),
              ("ours_context_deberta+para", cfg["detector"]["backbone"], True)]
    models = [m for m in models if m[0] not in (sc.get("skip") or [])]
    if models:
        import shutil
        from mcptp.detector.train import LoadedDetector, train_default_detector
        for seed in seeds:
            for name, backbone, use_ctx in models:
                c = {**cfg, "detector": {**cfg["detector"], "backbone": backbone}}
                log.info("training %s seed %s", name, seed)
                ck_dir = run_dir / f"ckpt_{name}_s{seed}"
                ck = train_default_detector(c, paths, use_ctx, ck_dir, log, seed=seed)
                det = LoadedDetector(ck)
                record(name, seed, {k: det.score(s) if s else [] for k, s in sets.items()})
                del det
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if not sc.get("keep_ckpt", False):
                    shutil.rmtree(ck_dir, ignore_errors=True)

    # B2: LLM judge(s) (deterministic -> one run each; needs GPU + bitsandbytes).
    # score.judges: list of {name, model, quant, shots, batch}; default = the original zero-shot Qwen3-8B judge.
    if sc.get("judge", True):
        from mcptp.baselines.llm_judge import LLMJudge, select_shots
        judges = sc.get("judges") or [{"name": "B2_llm_judge", "model": sc.get("judge_model", "Qwen/Qwen3-8B"),
                                       "quant": "4bit", "shots": 0, "batch": int(sc.get("judge_batch", 8))}]
        if sc.get("only_judges"):            # run a subset by name, e.g. to split judges across notebooks
            judges = [j for j in judges if j["name"] in sc["only_judges"]]
        base_ids = attack_base_ids()
        shot_log = {}
        for j in judges:
            k = int(j.get("shots", 0))
            shots = select_shots(ex, k_pos=k // 2, k_neg=k - k // 2, exclude_ids=base_ids,
                                 seed=cfg["project"]["seed"]) if k else []
            shot_log[j["name"]] = {**j, "shot_ids": [s["id"] for s in shots]}
            log.info("judge %s: model=%s quant=%s shots=%d", j["name"], j["model"], j.get("quant"), len(shots))
            judge = LLMJudge(j["model"], device=sc.get("judge_device", "cuda:0"), quant=j.get("quant", "4bit"),
                             batch_size=int(j.get("batch", 8)), shots=shots, score=j.get("score", "margin"))
            record(j["name"], 0, {k_: judge.score(s) if s else [] for k_, s in sets.items()})
            del judge
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        (run_dir / "judge_shots.json").write_text(json.dumps(shot_log, indent=1))
    scores_f.close()

    summary = {name: {kind: summarize_runs([r[kind] for r in runs]) for kind in ("val", "matched", "auroc")}
               for name, runs in per_seed.items()}
    out = {"n": n_sets, "seeds": seeds, "summary": summary, "per_seed": per_seed}
    (run_dir / "robustness.json").write_text(json.dumps(out, indent=1))

    def fmt(cell):
        if cell is None:
            return "-"
        m, s, n = cell
        return f"{m:.3f}" if n == 1 else f"{m:.3f}±{s:.3f}"

    atk = [f"{r}_{t}" for r in ROUNDS for t in ("orig", "opt") if f"{r}_{t}" in sets]
    lines = [f"Trained classifiers: mean±std over seeds {seeds}; B0 and B2 are deterministic (single run).", ""]
    for fp in ("0.01", "0.05"):
        f = f"{float(fp):g}"
        cols = ["test_pos", "test_neg", "test_hardneg"] + atk
        lines += [f"### Val-fixed threshold @ {float(fp):.0%} val FPR (deployable; realized FPR in *_neg columns)", "",
                  "| defender | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        lines += [f"| {n} | " + " | ".join(fmt(s['val'].get(f"{c}@{f}")) for c in cols) + " |" for n, s in summary.items()]
        cols = ["test_pos", "test_hardneg"] + atk
        lines += ["", f"### Matched threshold @ {float(fp):.0%} TEST FPR (oracle; equal FPR for every defender)", "",
                  "| defender | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        lines += [f"| {n} | " + " | ".join(fmt(s['matched'].get(f"{c}@{f}")) for c in cols) + " |" for n, s in summary.items()]
        lines.append("")
    cols = ["test_pos", "test_pos_vs_hardneg"] + atk
    lines += ["### AUROC vs test negatives (threshold-free)", "",
              "| defender | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {n} | " + " | ".join(fmt(s['auroc'].get(c)) for c in cols) + " |" for n, s in summary.items()]
    lines += ["", "n per set: " + json.dumps(n_sets)]
    (run_dir / "robustness.md").write_text("\n".join(lines) + "\n")
    log.info("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
