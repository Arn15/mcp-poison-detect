"""Step 8: do the white-box evasions survive cheap deployable defenses?

A. Fluency filter: mean token negative log-likelihood (log-perplexity) of the description under a small causal LM
   (GPT-2). Threshold = val negatives at the target FPR (higher NLL = flagged). Reports flag rates on test
   negatives / hard negatives (false alarms), test positives, control-verified black-box rewrites, and the
   white-box evaders (all evaders, and genuine evaders) of every white-box run.
B. Registry-relative calibration (mcptp.eval.calibration, stat=max) on the detector the white-box run attacked:
   r(c) = m(c) - max_{t in R(server), name(t) != name(c)} m(t), margins clipped to +/-logit(1-1e-7) like the
   calibrator. The detector is re-trained with the white-box run's backbone and seed (deterministic twin, unless
   defense.ckpts maps the run to the exact checkpoint); we report how many evaders still evade the twin under the
   global threshold, then whether calibration flags them.
C. Combined: flagged by the calibrated detector OR by the fluency filter (FPR is the union of both).

Kaggle (1 GPU; ~15 min): SCRIPT="scripts/defense_checks.py"  ARGS="--tag def1"
Only the fluency part (works on CPU): --set 'defense.parts=[fluency]'
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from common import setup_run

from mcptp.data.io import ensure_examples, load_para
from mcptp.eval.metrics import threshold_at_fpr
from mcptp.utils import atomic_write_json
from mcptp.utils.env import REPO_ROOT

DEFAULTS = {"parts": ["fluency", "calibration"], "wb_runs": ["results/wb1", "results/wb1e5"], "lm": "gpt2",
            "fprs": [0.01, 0.05], "ckpts": {}, "floor": 0.2}
CLIP = math.log((1 - 1e-7) / 1e-7)


def ex_(server, name, desc, label=1):
    return {"server": server, "tool_name": name, "description": desc, "label": label, "context": []}


def wb_sets(run):
    """Evaders and genuine evaders of a white-box run, as detector examples."""
    W = [json.loads(l) for l in open(REPO_ROOT / run / "whitebox.jsonl")]
    vp = REPO_ROOT / run / "wb_verified.jsonl"            # absent when a run has no evaders to verify
    V = {v["key"]: v for v in map(json.loads, open(vp))} if vp.exists() else {}
    ev = [w for w in W if w["evaded"]]
    gen = [w for w in ev if w["key"] in V and V[w["key"]]["asr_true"] >= 0.2]
    f = lambda rows: [ex_(w["server"], w["tool_name"], w["adv_desc"]) for w in rows]
    return {"evaders": f(ev), "genuine": f(gen)}


def lm_nll(texts, model_name, log, bs=16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_name).to(dev).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok([tok.bos_token + t for t in texts[i:i + bs]], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(dev)
            logits = m(**enc).logits[:, :-1].float()
            tgt, mask = enc["input_ids"][:, 1:], enc["attention_mask"][:, 1:].float()
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
            out += ((nll * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().tolist()
    log.info("LM scored %d texts", len(texts))
    del m
    return np.asarray(out)


def rate(scores, thr):
    s = np.asarray(scores, dtype=float)
    return float((s >= thr).mean()) if len(s) else None


def main():
    cfg, paths, run_dir, log = setup_run("defense_checks", __doc__)
    c = {**DEFAULTS, **(cfg.get("defense") or {})}
    ex = ensure_examples(cfg, paths)
    v = cfg["para"]["version"]
    para = {s: load_para(paths["splits"], v, s) for s in ("val", "test")}
    val = [e for e in ex if e["split"] == "val"] + para["val"]
    test = [e for e in ex if e["split"] == "test"] + para["test"]
    base = {"val_neg": [e for e in val if e["label"] == 0], "test_neg": [e for e in test if e["label"] == 0],
            "test_hardneg": [e for e in test if e["label"] == 0 and e.get("is_hard_negative")],
            "test_pos": [e for e in test if e["label"] == 1]}
    seen, bb = set(), []
    for r in map(json.loads, open(REPO_ROOT / "data" / "attacks" / "verified.jsonl")):
        if r["kind"] == "opt" and r["asr_true"] >= c["floor"] and r["desc"] not in seen:
            seen.add(r["desc"]); bb.append(ex_(r["server"], r["tool_name"], r["desc"]))
    base["blackbox_opt"] = bb
    wb = {run: wb_sets(run) for run in c["wb_runs"]}
    out = {"sizes": {k: len(x) for k, x in base.items()} | {f"{Path(r).name}_{k}": len(x) for r, s in wb.items()
                                                              for k, x in s.items()}}
    log.info("sets: %s", out["sizes"])
    fl_flag = {}   # (run, set) -> bool array at 1% (for the combined defense)

    # ---------------------------------------------------------------- A. fluency filter
    if "fluency" in c["parts"]:
        allsets = dict(base) | {f"{Path(r).name}_{k}": x for r, s in wb.items() for k, x in s.items()}
        names = list(allsets)
        flat = [e["description"] for n in names for e in allsets[n]]
        nll = lm_nll(flat, c["lm"], log)
        sc, i = {}, 0
        for n in names:
            sc[n] = nll[i:i + len(allsets[n])]; i += len(allsets[n])
        res = {}
        for fp in c["fprs"]:
            thr = threshold_at_fpr(sc["val_neg"], fp)
            res[f"thr@{fp:g}"] = float(thr)
            for n in names:
                if n != "val_neg":
                    res[f"{n}@{fp:g}"] = rate(sc[n], thr)
            if fp == 0.01:
                fl_flag = {n: np.asarray(sc[n]) >= thr for n in names}
        res["median_nll"] = {n: float(np.median(s)) for n, s in sc.items() if len(s)}
        out["fluency"] = res
        log.info("fluency: %s", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in res.items()
                                 if k.endswith("@0.01")})

    # ---------------------------------------------------------------- B. registry-relative calibration
    if "calibration" in c["parts"]:
        from mcptp.detector.train import LoadedDetector, train_default_detector
        reg = [e for e in ex if e["kind"] == "benign"]          # original benign tools of every server
        out["calibration"] = {}
        for run, s in wb.items():
            rc = yaml.safe_load(open(REPO_ROOT / run / "config.yaml"))
            rcfg = {**cfg, "detector": rc["detector"], "project": rc["project"]}
            ck = c["ckpts"].get(run) or str(train_default_detector(rcfg, paths, False, run_dir / f"ckpt_{Path(run).name}",
                                                                   log, seed=rc["project"]["seed"]))
            det = LoadedDetector(ck)
            M = lambda xs: np.clip(det.margin(xs), -CLIP, CLIP) if len(xs) else np.array([])
            sets = dict(base) | s
            m = {k: M(x) for k, x in sets.items()}
            mreg = M(reg)
            ref = defaultdict(list)
            for e, sc_ in zip(reg, mreg):
                ref[e["server"]].append((e["tool_name"], float(sc_)))

            def rel(xs, ms):
                o = []
                for e, sc_ in zip(xs, ms):
                    r = [v for n, v in ref.get(e["server"], []) if n != e["tool_name"]]
                    o.append(sc_ - max(r) if r else np.nan)
                return np.asarray(o, dtype=float)

            r = {k: rel(sets[k], m[k]) for k in sets}
            res = {"detector": ck, "backbone": rc["detector"]["backbone"]}
            for fp in c["fprs"]:
                tg = threshold_at_fpr(m["val_neg"], fp)
                vr = r["val_neg"][~np.isnan(r["val_neg"])]
                tr = threshold_at_fpr(vr, fp)
                for k in sets:
                    if k == "val_neg":
                        continue
                    rk = r[k][~np.isnan(r[k])]
                    res[f"{k}@{fp:g}"] = {"global": rate(m[k], tg), "rel_max": rate(rk, tr),
                                          "coverage": f"{len(rk)}/{len(r[k])}"}
                    if fp == 0.01 and k in ("test_neg", "test_hardneg", "evaders", "genuine", "test_pos"):
                        flag_rel = np.where(np.isnan(r[k]), m[k] >= tg, r[k] >= tr)
                        fl = fl_flag.get(k if k in base else f"{Path(run).name}_{k}")
                        if fl is not None:
                            res[f"{k}@0.01"]["rel_max_or_fluency"] = float((flag_rel | fl).mean()) if len(fl) else None
            out["calibration"][run] = res
            log.info("calibration %s: %s", run, {k: v for k, v in res.items() if k.endswith("@0.01")})
            del det

    atomic_write_json(out, run_dir / "defense_results.json")
    # ---------------------------------------------------------------- markdown
    L = ["# Defense checks against white-box evasions", "", f"Set sizes: {out['sizes']}", ""]
    if "fluency" in out:
        f = out["fluency"]
        L += [f"## A. Fluency filter ({c['lm']} log-perplexity, threshold on val negatives)", "",
              "| set | flagged @1% | flagged @5% | median NLL |", "|---|---|---|---|"]
        for n in f["median_nll"]:
            if n != "val_neg":
                L.append(f"| {n} | {f[f'{n}@0.01']:.3f} | {f[f'{n}@0.05']:.3f} | {f['median_nll'][n]:.2f} |")
        L.append("")
    for run, res in (out.get("calibration") or {}).items():
        L += [f"## B/C. Calibration on {run} ({res['backbone']}, twin detector)", "",
              "| set @1% | global | rel_max | rel_max OR fluency | coverage |", "|---|---|---|---|---|"]
        for k, val_ in res.items():
            if k.endswith("@0.01"):
                o = val_.get("rel_max_or_fluency")
                L.append(f"| {k[:-5]} | {val_['global']:.3f} | {val_['rel_max']:.3f} | "
                         f"{'' if o is None else f'{o:.3f}'} | {val_['coverage']} |")
        L.append("")
    (run_dir / "defense_results.md").write_text("\n".join(L) + "\n")
    log.info("\n%s", "\n".join(L))


if __name__ == "__main__":
    main()
