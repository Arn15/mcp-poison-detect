"""Step 2 sanity check: can trivial text features or rules solve the benchmark?

If a TF-IDF + logistic-regression model or a keyword rule already reaches high TPR at low FPR, the data
has surface shortcuts (MCP-Poison-Bench's keyword-circularity concern) and gains from a DeBERTa detector
would not mean much. Writes results/leakage_<version>.json.
"""
from __future__ import annotations

import json
import re

import numpy as np
from common import setup_run
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from mcptp.data.io import load_examples
from mcptp.eval.metrics import rates_at_threshold, summarize, threshold_at_fpr

KEYWORDS = ["before", "must", "first", "mandatory", "ignore", "important", "critical", "security", "required",
            "always", "automatically", "instead"]


def tfidf_lr(C=1.0):
    feats = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                          ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True))])
    return make_pipeline(feats, LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


def kw_score(t):
    t = t.lower()
    return sum(len(re.findall(r"\b" + k + r"\b", t)) for k in KEYWORDS)


def subsets(test):
    neg = [e for e in test if e["label"] == 0]
    pos = [e for e in test if e["label"] == 1]
    return {
        "all": test,
        "implicit": neg + [e for e in pos if e["is_implicit"]],
        "heldout_server_only": neg + [e for e in pos if not e["heldout_family"]],
        "heldout_family": neg + [e for e in pos if e["heldout_family"]],
    }


def main():
    cfg, paths, run_dir, log = setup_run("leakage_check", __doc__)
    ex = load_examples(paths["data_processed"], paths["splits"], cfg["data"]["split_version"])
    tr = [e for e in ex if e["split"] == "train"]
    va = [e for e in ex if e["split"] == "val"]
    te = [e for e in ex if e["split"] == "test"]
    fprs = [cfg["eval"]["target_fpr"], *cfg["eval"].get("extra_fprs", [])]
    nb = cfg["eval"]["bootstrap_iters"]

    scorers = {}
    for field in ("description", "description_stripped"):
        m = tfidf_lr().fit([e[field] for e in tr], [e["label"] for e in tr])
        scorers[f"tfidf_lr[{field}]"] = (lambda es, m=m, f=field: m.predict_proba([e[f] for e in es])[:, 1], m)
    tr_nohn = [e for e in tr if not e["is_hard_negative"]]
    m = tfidf_lr().fit([e["description"] for e in tr_nohn], [e["label"] for e in tr_nohn])
    scorers["tfidf_lr[description, no hard negs in train]"] = (lambda es, m=m: m.predict_proba([e["description"] for e in es])[:, 1], m)
    scorers["rule:has_marker"] = (lambda es: np.array([float(e["has_marker"]) for e in es]), None)
    scorers["rule:references_registered_tool"] = (lambda es: np.array([float(e["ref_method"] == "name") for e in es]), None)
    scorers["rule:keyword_count"] = (lambda es: np.array([kw_score(e["description"]) for e in es], float), None)

    results = {}
    for name, (fn, _) in scorers.items():
        r = {}
        for sub, es in subsets(te).items():
            y = [e["label"] for e in es]
            r[sub] = summarize(y, fn(es), fprs, nb, seed=cfg["project"]["seed"])
        # deployment-style: threshold fixed on val at 5% FPR, then FPR on test hard negatives
        sv = fn(va)
        thr = threshold_at_fpr([s for s, e in zip(sv, va) if e["label"] == 0], 0.05)
        hn = [e for e in te if e["is_hard_negative"] or e["label"] == 1]
        r["val_thr@5%fpr"] = {"all_test": rates_at_threshold([e["label"] for e in te], fn(te), thr),
                              "hard_neg_fpr": rates_at_threshold([e["label"] for e in hn], fn(hn), thr)["fpr"]}
        results[name] = r
        a = r["all"]
        log.info("%-48s AUROC %.3f  TPR@1%% %.3f  TPR@5%% %.3f | implicit TPR@5%% %.3f | fam TPR@5%% %.3f | HN-FPR %.2f",
                 name, a["auroc"][0], a["tpr@0.01fpr"][0], a["tpr@0.05fpr"][0],
                 r["implicit"]["tpr@0.05fpr"][0], r["heldout_family"]["tpr@0.05fpr"][0], r["val_thr@5%fpr"]["hard_neg_fpr"])

    # most predictive n-grams of the raw TF-IDF model
    m = scorers["tfidf_lr[description]"][1]
    names = np.concatenate([t.get_feature_names_out() for _, t in m[0].transformer_list])
    coef = m[-1].coef_[0]
    results["top_positive_features"] = [names[i] for i in np.argsort(-coef)[:25]]
    results["top_negative_features"] = [names[i] for i in np.argsort(coef)[:25]]
    log.info("top + features: %s", results["top_positive_features"])
    log.info("top - features: %s", results["top_negative_features"])

    out = {"version": cfg["data"]["split_version"], "metric_format": "[point, ci_lo, ci_hi]", "results": results}
    from mcptp.utils.env import REPO_ROOT
    (REPO_ROOT / "results" / f"leakage_{cfg['data']['split_version']}.json").write_text(json.dumps(out, indent=1))
    (run_dir / "leakage.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
