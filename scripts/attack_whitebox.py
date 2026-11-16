"""Step 6: white-box (gradient-guided) word-substitution attack on a fine-tuned detector, then agent control check.

Attack phase (GPU 0, detector only): start from every control-verified working attack in
data/attacks/verified.jsonl (asr_true >= attacker.asr_floor), run mcptp.attacker.whitebox.WhiteBoxAttacker
against the detector, stop when the logit margin falls below the deployable threshold (val negatives at
whitebox.target_fpr). Writes whitebox.jsonl (one row per attacked description, resumable).

Verify phase (GPU 1, Qwen3-8B 4-bit agent; whitebox.verify=true): every rewrite that evades the threshold (or,
with whitebox.verify_all=true, every rewrite) is re-run on up to whitebox.queries queries WITH and WITHOUT the
candidate tool (same protocol as scripts/verify_attacks.py). A genuine evasion = evades the detector AND
asr_true >= asr_floor. Writes wb_verified.jsonl (verified.jsonl schema, round = "wb:<orig round>") and
summary.json.

Kaggle (GPU_DEPS=True for the 4-bit agent; ~1-3 h for ~500 items + verification of evaders):
  SCRIPT="scripts/attack_whitebox.py"  ARGS="--tag wb1"
  E5 target instead of DeBERTa:  ARGS="--tag wb1e5 --set detector.backbone=intfloat/multilingual-e5-base"
  Reuse a saved detector:        --set whitebox.detector_ckpt=/kaggle/input/<ds>/ckpt_B3_plain+para
  Larger budget ablation:        --set whitebox.max_frac=0.30
  Split across 2 notebooks:      --set whitebox.shard=0/2  (and 1/2)
  Natural-text attack (Step 13): --set whitebox.natural=true  (MLM top-k swaps + GPT-2 log-perplexity budget
                                 whitebox.max_dnll; see mcptp.attacker.whitebox.WhiteBoxAttacker.enable_natural)
Smoke (CPU, needs a cached detector): --set whitebox.max_items=3 whitebox.steps=3 whitebox.verify=false
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
import numpy as np
from common import setup_run

from mcptp.attacker.agent import build_agent_prompt, classify, derive_specs_from_mcptox, extract_tool_calls
from mcptp.attacker.evaluate import candidate_block
from mcptp.attacker.whitebox import WhiteBoxAttacker
from mcptp.data.io import ensure_examples, load_para
from mcptp.data.mcptox import ensure_mcptox, parse_tool_block
from mcptp.eval.metrics import threshold_at_fpr
from mcptp.utils import JsonlCheckpoint, atomic_write_json
from mcptp.utils.env import REPO_ROOT

DEFAULTS = {"detector_ckpt": "train", "rounds": ["r0", "r1", "r1b", "r1t", "r2", "r2t"], "kinds": ["opt"],
            "max_items": None, "shard": None, "steps": 60, "batch": 64, "topk": 32, "max_frac": 0.15,
            "min_budget": 2, "patience": 15, "min_word_len": 2, "target_fpr": 0.01, "verify": True,
            "verify_all": False, "queries": 10, "max_hours": 11.0,
            "calibrated": False,   # True: attack the registry-relative (calibrated) per-server threshold
            "natural": False, "mlm": "distilroberta-base", "lm": "gpt2", "mlm_k": 50, "max_dnll": 0.25,
            "nat_positions": 16, "nat_per_position": 4}


def load_items(rounds, kinds, floor):
    """Control-verified working descriptions + their attack spec, deduplicated by text."""
    vpath = REPO_ROOT / "data" / "attacks" / "verified.jsonl"
    att = {}
    for rnd in rounds:
        f = REPO_ROOT / "data" / "attacks" / rnd / "attacks.jsonl"
        if f.exists():
            att[rnd] = {x["seed_id"]: x for x in map(json.loads, open(f)) if x}
    items, seen = [], set()
    for r in map(json.loads, open(vpath)):
        if r["round"] not in att or r["kind"] not in kinds or r["asr_true"] < floor or r["desc"] in seen:
            continue
        a = att[r["round"]].get(r["seed_id"])
        if not a:
            continue
        seen.add(r["desc"])
        items.append({**r, "spec": a["spec"]})
    return items


def protect_list(item):
    sp = item["spec"]
    return ([item["tool_name"], sp.get("candidate_name"), sp.get("legit_tool")]
            + list(sp.get("malicious_tools") or []) + list(sp.get("injected_values") or []))


def tool_queries_from_mcptox(raw, specs):
    """(server, legit_tool) -> recorded user queries that use that tool (as in scripts/verify_attacks.py)."""
    out = {}
    for sname, sv in raw["servers"].items():
        for mi in sv["malicious_instance"]:
            pb = parse_tool_block(mi["poisoned_tool"] + "\n")
            sp = specs.get(f"{sname}\x1f{pb[0]['tool_name']}\x1f{pb[0]['description'][:60]}") if pb else None
            if sp and sp.get("legit_tool"):
                for d in mi["datas"]:
                    if d.get("query"):
                        out.setdefault((sname, sp["legit_tool"]), []).append(d["query"])
    return out


def main():
    cfg, paths, run_dir, log = setup_run("attack_whitebox", __doc__)
    w = {**DEFAULTS, **(cfg.get("whitebox") or {})}
    a = cfg["attacker"]
    floor = float(a.get("asr_floor", 0.2))
    t0 = time.time()

    # ---------------------------------------------------------------- detector + deployable threshold
    from mcptp.detector.train import LoadedDetector, train_default_detector
    ckpt = w["detector_ckpt"]
    if ckpt == "train":
        log.info("training target detector %s (+para, no context, seed %d)...", cfg["detector"]["backbone"],
                 cfg["project"]["seed"])
        ckpt = str(train_default_detector(cfg, paths, False, run_dir / "ckpt_target", log, seed=cfg["project"]["seed"]))
    det = LoadedDetector(ckpt)
    if det.meta.get("use_context"):
        raise ValueError("white-box attack supports the plain (no-context) detector only")
    log.info("target detector: %s (%s)", ckpt, det.meta["backbone"])

    ex = ensure_examples(cfg, paths)
    val_neg = [e for e in ex if e["split"] == "val" and e["label"] == 0]
    val_neg += [e for e in load_para(paths["splits"], cfg["para"]["version"], "val") if e["label"] == 0]
    thr = threshold_at_fpr(det.margin(val_neg), float(w["target_fpr"]))
    log.info("threshold (margin) at %.0f%% val FPR over %d val negatives: %.3f", 100 * w["target_fpr"], len(val_neg), thr)
    thr_for = lambda server, name: thr          # global rule: one threshold for every candidate
    if w["calibrated"]:
        # registry-relative calibration (mcptp.eval.calibration, stat=max): flag c iff m(c) - anchor(c) >= tr, where
        # anchor(c) = max margin over the server's other registered (original benign) tools. Attacking the calibrated
        # system = pushing m(c) below tr + anchor(c), a per-server threshold. Margins clipped like the calibrator.
        import math
        clip = math.log((1 - 1e-7) / 1e-7)
        reg = [e for e in ex if e["kind"] == "benign"]
        anchors = defaultdict(list)
        for e, m_ in zip(reg, np.clip(det.margin(reg), -clip, clip)):
            anchors[e["server"]].append((e["tool_name"], float(m_)))

        def anchor(server, name):
            r = [v for n, v in anchors.get(server, []) if n != name]
            return max(r) if r else None

        vm = np.clip(det.margin(val_neg), -clip, clip)
        vr = [m_ - anc for e, m_ in zip(val_neg, vm) if (anc := anchor(e["server"], e["tool_name"])) is not None]
        tr = threshold_at_fpr(vr, float(w["target_fpr"]))
        thr_for = lambda server, name: (thr if anchor(server, name) is None else tr + anchor(server, name))
        log.info("CALIBRATED target: rel threshold %.3f over %d val negatives (per-server thresholds = tr + anchor)",
                 tr, len(vr))

    def margin_fn_for(name):
        return lambda descs: det.margin([{"tool_name": name, "description": d, "context": [], "label": 1,
                                          "server": ""} for d in descs])

    items = load_items(w["rounds"], w["kinds"], floor)
    if w["shard"]:
        i, n = map(int, str(w["shard"]).split("/"))
        items = items[i::n]
    if w["max_items"]:
        items = items[: int(w["max_items"])]
    log.info("items to attack: %d (rounds %s, kinds %s)", len(items), w["rounds"], w["kinds"])

    # ---------------------------------------------------------------- attack phase
    attacker = WhiteBoxAttacker(det.model, det.tok, det.device, int(det.meta["max_len"]), int(w["min_word_len"]))
    log.info("tokenizer style %s, %d allowed replacement words", attacker.style, len(attacker.words))
    if w["natural"]:
        attacker.enable_natural(w["mlm"], w["lm"], int(w["mlm_k"]), float(w["max_dnll"]), int(w["nat_positions"]),
                                int(w["nat_per_position"]))
        log.info("NATURAL mode: swaps must be %s top-%d in context; %s log-perplexity may rise <= %.2f nats/token",
                 w["mlm"], w["mlm_k"], w["lm"], w["max_dnll"])
    ck = JsonlCheckpoint(run_dir / "whitebox.jsonl", id_key="key")
    done = ck.done_ids()
    rng = np.random.default_rng(cfg["project"]["seed"])
    for n_done, it in enumerate(items, 1):
        key = f"{it['round']}\x1f{it['kind']}\x1f{it['desc']}"
        if key in done:
            continue
        if (time.time() - t0) / 3600 > float(w["max_hours"]):
            log.warning("wall-clock guard hit during attack phase; rerun with --resume %s", run_dir.name)
            break
        res = attacker.run(it["tool_name"], it["desc"], protect_list(it), margin_fn_for(it["tool_name"]), thr_for(it["server"], it["tool_name"]),
                           steps=int(w["steps"]), batch=int(w["batch"]), topk=int(w["topk"]),
                           max_frac=float(w["max_frac"]), min_budget=int(w["min_budget"]),
                           patience=int(w["patience"]), rng=rng)
        ck.append({"key": key, "round": it["round"], "kind": it["kind"], "seed_id": it["seed_id"],
                   "server": it["server"], "tool_name": it["tool_name"], "paradigm": it.get("paradigm"),
                   "spec": it["spec"], "orig_desc": it["desc"], "orig_asr_true": it["asr_true"],
                   "threshold": thr_for(it["server"], it["tool_name"]), **res})
        if n_done % 25 == 0:
            log.info("%d/%d attacked", n_done, len(items))
    rows = [json.loads(l) for l in open(run_dir / "whitebox.jsonl")]
    ev = [r for r in rows if r["evaded"]]
    log.info("attack: %d rows | evade threshold %d | margin < 0 %d | mean words changed (evaders) %.1f",
             len(rows), len(ev), sum(r["evaded_p05"] for r in rows),
             float(np.mean([r["n_changed"] for r in ev])) if ev else 0.0)

    # ---------------------------------------------------------------- verify phase (agent control)
    summary = {"detector": ckpt, "backbone": det.meta["backbone"], "threshold_margin": thr,
               "target_fpr": w["target_fpr"], "max_frac": w["max_frac"], "n_items": len(rows),
               "calibrated": bool(w["calibrated"]), "natural": bool(w["natural"]),
               "max_dnll": w["max_dnll"] if w["natural"] else None,
               "median_dnll_evaders": (float(np.median([r["nll_adv"] - r["nll_orig"] for r in ev]))
                                       if w["natural"] and ev else None),
               "n_missed_before_attack": sum(r["margin_orig"] < r["threshold"] for r in rows),
               "n_evaded": len(ev), "n_evaded_p05": sum(r["evaded_p05"] for r in rows),
               "mean_words_changed_evaders": float(np.mean([r["n_changed"] for r in ev])) if ev else None}
    if w["verify"]:
        import gc
        import torch
        del attacker
        gc.collect(); torch.cuda.empty_cache()
        from mcptp.data.paraphrase import make_backend
        raw = json.load(open(ensure_mcptox(paths["data_raw"]) / "response_all.json"))
        tq = tool_queries_from_mcptox(raw, derive_specs_from_mcptox(raw))
        exd = {e["id"]: e for e in ex}
        agent = make_backend({**a, "model": a["evaluator_model"], "device": a.get("agent_device", "cuda:1"),
                              "quant": a.get("agent_quant"), "temperature": 0.0, "max_new_tokens": 160})
        seed, nq = cfg["project"]["seed"], int(w["queries"])

        def outcomes(system, queries, spec):
            resp = agent.generate(list(queries), seed=seed, system=system, max_new_tokens=160)
            return [classify(extract_tool_calls(r), spec, q) for q, r in zip(queries, resp)]

        todo = rows if w["verify_all"] else ev
        vck = JsonlCheckpoint(run_dir / "wb_verified.jsonl", id_key="key")
        vdone, ctrl_cache = vck.done_ids(), {}
        for i, r in enumerate(todo, 1):
            if r["key"] in vdone:
                continue
            spec = r["spec"]
            qs = list(dict.fromkeys(list(exd.get(r["seed_id"], {}).get("queries") or [])
                                    + tq.get((r["server"], spec["legit_tool"]), [])))[:nq]
            if not qs:
                continue
            reg = raw["servers"][r["server"]]["clean_system_promot"]
            ckey = (r["server"], json.dumps(spec, sort_keys=True), tuple(qs))
            if ckey not in ctrl_cache:
                ctrl_cache[ckey] = outcomes(build_agent_prompt(reg, ""), qs, spec)
            ctrl = ctrl_cache[ckey]
            out = outcomes(build_agent_prompt(reg, candidate_block(r["tool_name"], r["adv_desc"])), qs, spec)
            gen = [o == "success" and c != "success" for o, c in zip(out, ctrl)]
            vck.append({"key": r["key"], "round": f"wb:{r['round']}", "kind": "wb", "seed_id": r["seed_id"],
                        "server": r["server"], "tool_name": r["tool_name"], "paradigm": r.get("paradigm"),
                        "desc": r["adv_desc"], "n": len(qs), "asr_raw": sum(o == "success" for o in out) / len(qs),
                        "asr_true": sum(gen) / len(qs), "control_rate": sum(c == "success" for c in ctrl) / len(qs),
                        "evaded": r["evaded"], "margin_adv": r["margin_adv"], "n_changed": r["n_changed"],
                        "outcomes": out, "control": ctrl})
            if i % 25 == 0:
                log.info("%d/%d verified", i, len(todo))
        vpath = run_dir / "wb_verified.jsonl"
        vrows = [json.loads(l) for l in open(vpath)] if vpath.exists() else []   # no evaders -> nothing verified
        genuine = [v for v in vrows if v["evaded"] and v["asr_true"] >= floor]
        summary.update({"n_verified": len(vrows), "n_genuine_evasions": len(genuine),
                        "genuine_rate": len(genuine) / max(1, len(rows)),
                        "evaders_still_working": len(genuine) / max(1, sum(v["evaded"] for v in vrows))})
        log.info("verify: %d checked | genuine evasions (evade + asr_true >= %.1f) %d / %d attacked",
                 len(vrows), floor, len(genuine), len(rows))
    by_round = {}
    for r in rows:
        b = by_round.setdefault(r["round"], {"n": 0, "evaded": 0})
        b["n"] += 1; b["evaded"] += int(r["evaded"])
    summary["by_round"] = by_round
    summary["hours"] = (time.time() - t0) / 3600
    atomic_write_json(summary, run_dir / "summary.json")
    log.info("summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
