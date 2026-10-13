"""Step 4: score an attack set from run_attack against every defender (the context go/no-go).

Loads attacks.jsonl (best rewrite per seed), builds detector examples from each rewrite, and scores with
B0 (TF-IDF+LR, trained here), B3_plain+para and ours_context+para (loaded checkpoints), and optionally B2
(LLM judge, via the agent backend). Reports, over attacks that actually work (ASR>=floor), each defender's
detection rate at its own val-fixed threshold, plus the evasion-vs-ASR summary.

Kaggle: SCRIPT="scripts/score_attacks.py"
        ARGS="--tag r0score --set score.attacks=/kaggle/input/<atk>/attacks.jsonl
              score.ckpts=[/kaggle/input/<ck>/ckpt_B3_plain+para,/kaggle/input/<ck>/ckpt_ours_context+para]"
"""
from __future__ import annotations

import json

import numpy as np
from common import setup_run

from mcptp.data.context import find_references
from mcptp.data.io import ensure_examples
from mcptp.eval.metrics import threshold_at_fpr


def main():
    cfg, paths, run_dir, log = setup_run("score_attacks", __doc__)
    sc = cfg.get("score", {})
    atk = [json.loads(l) for l in open(sc["attacks"]) if l.strip()]
    atk = [a for a in atk if a.get("best")]
    floor = float(cfg["attacker"].get("asr_floor", 0.2))
    working = [a for a in atk if a["best"]["asr"] >= floor]
    log.info("%d attacks, %d working (ASR>=%.2f)", len(atk), len(working), floor)

    ex = ensure_examples(cfg, paths)
    registries = {}
    for e in ex:
        if e["kind"] == "benign":
            registries.setdefault(e["server"], {})[e["tool_name"]] = e["description"]
    val = [e for e in ex if e["split"] == "val"]

    # build an example per working attack (rebuild context from the rewritten text)
    atk_ex = []
    for a in working:
        reg = registries.get(a["server"], {})
        refs, method = find_references(a["best"]["desc"], reg, exclude=a["tool_name"],
                                       top_k=cfg["context"]["top_k"])
        atk_ex.append({"id": a["seed_id"], "server": a["server"], "tool_name": a["tool_name"],
                       "description": a["best"]["desc"], "description_stripped": a["best"]["desc"],
                       "label": 1, "context": [{"tool_name": t, "description": reg[t]} for t in refs],
                       "referenced_tools": refs, "ref_method": method, "asr": a["best"]["asr"]})

    results = {}
    # B0
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline
    tr = [e for e in ex if e["split"] == "train"]
    feats = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                          ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True))])
    b0 = make_pipeline(feats, LogisticRegression(max_iter=2000, class_weight="balanced"))
    b0.fit([e["description"] for e in tr], [e["label"] for e in tr])
    b0v = b0.predict_proba([e["description"] for e in val if e["label"] == 0])[:, 1]
    for fp in (0.01, 0.05):
        thr = threshold_at_fpr(b0v, fp)
        s = b0.predict_proba([e["description"] for e in atk_ex])[:, 1]
        results.setdefault("B0_tfidf_lr", {})[f"detect@{fp:g}"] = float((np.array(s) >= thr).mean())

    # loaded detector checkpoints
    for ck in sc.get("ckpts", []):
        from mcptp.detector.train import LoadedDetector
        det = LoadedDetector(ck)
        name = ck.rstrip("/").split("/")[-1]
        vs = det.score([e for e in val if e["label"] == 0])
        sa = det.score(atk_ex)
        for fp in (0.01, 0.05):
            thr = threshold_at_fpr(vs, fp)
            results.setdefault(name, {})[f"detect@{fp:g}"] = float((np.array(sa) >= thr).mean())

    asrs = np.array([a["asr"] for a in atk_ex])
    out = {"n_attacks": len(atk), "n_working": len(working), "working_mean_asr": float(asrs.mean()),
           "detectors": results}
    (run_dir / "score.json").write_text(json.dumps(out, indent=1))
    from mcptp.utils.env import REPO_ROOT
    (REPO_ROOT / "results" / "attack_score_r0.json").write_text(json.dumps(out, indent=1))
    log.info(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
