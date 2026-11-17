"""Step 4: run the implicit-poisoning optimizer against a target agent (+ optional detector in the loop).

Seeds = training-split poisoned tools (no test leakage). For each seed, iteratively rewrite its description
(attacker LLM), measure ASR on the agent over the server's clean queries, score with the in-loop detector,
and keep the best evasive-yet-working rewrite. Resumable (one record per seed).

Kaggle (GPU_DEPS=False; uses transformers):
  SCRIPT="scripts/run_attack.py"
  ARGS="--tag r0 --set attacker.max_seeds=250 attacker.detector_ckpt=/kaggle/input/<ds>/ckpt_ours_context+para"
Smoke (CPU): python scripts/run_attack.py --set attacker.backend=echo attacker.max_seeds=3 --tag smoke

Round 2 (stronger attacker): attacker.preset=r2 switches on 14B 4-bit attacker, seed dedupe, hill-climbing
from the best working rewrite, logit-margin selection, occlusion-saliency + benign-style feedback, T=5,
early stop on evasion, and a wall-clock guard. Kaggle:
  ARGS="--tag r2  --set attacker.preset=r2 attacker.detector_ckpt=train attacker.seed_split=train"
  ARGS="--tag r2t --set attacker.preset=r2 attacker.detector_ckpt=train attacker.seed_split=test"
"""
from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # ast.literal_eval on agent output

sys.argv = sys.argv  # keep common's parser
from common import setup_run

from mcptp.attacker.agent import derive_specs_from_mcptox
from mcptp.attacker.evaluate import candidate_block, measure_asr
from mcptp.attacker.optimize import (ATTACKER_SYSTEM, build_attacker_prompt, build_attacker_prompt_rich, clean_rewrite,
                                    salient_words, selection_score, selection_score_margin)
from mcptp.data.io import ensure_examples
from mcptp.data.paraphrase import make_backend
from mcptp.utils import JsonlCheckpoint, atomic_write_json


def main():
    """Entry point (see the module docstring for what this script does and how to run it)."""
    import json as _json

    cfg, paths, run_dir, log = setup_run("run_attack", __doc__)
    a = cfg["attacker"]
    if a.get("preset") == "r2":   # explicit --set values still win: only fill keys left at their defaults
        r2 = {"attacker_model": "unsloth/Qwen3-14B-bnb-4bit", "attacker_quant": "prequant", "dedupe_seeds": True,
              "iterations": 5, "feedback": "rich", "start_from": "best", "selection": "margin",
              "early_stop": True, "evasion_weight": 1.0}
        defaults = {"attacker_model": "Qwen/Qwen3-4B", "attacker_quant": None, "dedupe_seeds": False, "iterations": 2,
                    "feedback": "basic", "start_from": "seed", "selection": "prob", "early_stop": False,
                    "evasion_weight": 1.0}
        for k, v in r2.items():
            if a.get(k, defaults[k]) == defaults[k]:
                a[k] = v
    import time
    t_start, max_h = time.time(), float(a.get("max_hours") or 1e9)
    ex = ensure_examples(cfg, paths)
    from mcptp.data.mcptox import ensure_mcptox
    raw = _json.load(open(ensure_mcptox(paths["data_raw"]) / "response_all.json"))
    specs = derive_specs_from_mcptox(raw)
    registries = {s: raw["servers"][s] for s in raw["servers"]}
    # queries that exercise each target tool: an attack only fires when the user's request uses legit_tool
    from mcptp.data.mcptox import parse_tool_block
    tool_queries = {}
    for sname, sv in raw["servers"].items():
        for mi in sv["malicious_instance"]:
            pb = parse_tool_block(mi["poisoned_tool"] + "\n")
            sp = specs.get(f"{sname}\x1f{pb[0]['tool_name']}\x1f{pb[0]['description'][:60]}") if pb else None
            if sp and sp.get("legit_tool"):
                for d in mi["datas"]:
                    if d.get("query"):
                        tool_queries.setdefault((sname, sp["legit_tool"]), []).append(d["query"])

    # seeds: training-split poisoned tools with a usable attack spec
    seeds = []
    for e in ex:
        if e["split"] != a.get("seed_split", "train") or e["label"] != 1:
            continue
        key = f"{e['server']}\x1f{e['tool_name']}\x1f{e['description_stripped'][:60]}"
        spec = specs.get(key) or specs.get(f"{e['server']}\x1f{e['tool_name']}\x1f{e['description'][:60]}")
        if spec and spec.get("legit_tool") and (spec.get("malicious_tools") or spec.get("injected_values")):
            seeds.append((e, spec))
    if a.get("dedupe_seeds"):   # MCPTox marker-variants collapse to the same stripped description
        uniq = {}
        for e, spec in seeds:
            uniq.setdefault((e["server"], e["description_stripped"]), (e, spec))
        seeds = list(uniq.values())
    if a.get("shard"):           # "i/n": run every n-th seed starting at i (split a long run across sessions)
        i, n = map(int, str(a["shard"]).split("/"))
        seeds = seeds[i::n]
    if a.get("max_seeds"):
        seeds = seeds[: int(a["max_seeds"])]
    log.info("usable seeds: %d", len(seeds))

    detector = None
    ckpt = a.get("detector_ckpt")
    if ckpt == "train":
        from mcptp.detector.train import train_default_detector
        log.info("training in-loop detector (%s)...", a.get("detector_context", False))
        ckpt = str(train_default_detector(cfg, paths, bool(a.get("detector_context", False)),
                                          run_dir / "ckpt_inloop", log))
    if ckpt:
        from mcptp.detector.train import LoadedDetector
        detector = LoadedDetector(ckpt)
        log.info("in-loop detector: %s", ckpt)
    # free the detector's training memory before the LLMs load (OOM on a 15 GB T4 otherwise)
    try:
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass
    agent = make_backend({**a, "model": a["evaluator_model"], "device": a.get("agent_device", "cuda:1"),
                          "quant": a.get("agent_quant"), "temperature": 0.0, "max_new_tokens": 128})
    attacker = make_backend({**a, "model": a["attacker_model"], "device": a.get("attacker_device", "cuda:0"),
                             "quant": a.get("attacker_quant")})
    log.info("attacker %s (%s) | T=%s N=%s feedback=%s start_from=%s selection=%s", a["attacker_model"],
             a.get("attacker_quant"), a["iterations"], a["n_candidates"], a.get("feedback"), a.get("start_from"),
             a.get("selection"))
    benign_by_server = {}
    for e in ex:
        if e["kind"] == "benign":
            benign_by_server.setdefault(e["server"], []).append(e["description"])
    rich, hill = a.get("feedback") == "rich", a.get("start_from") == "best"
    use_margin = a.get("selection") == "margin"

    N, T = int(a["n_candidates"]), int(a["iterations"])
    lam, floor = float(a.get("evasion_weight", 1.0)), float(a.get("asr_floor", 0.2))
    nq = int(a.get("shadow_queries", 5))
    ckpt = JsonlCheckpoint(run_dir / "attacks.jsonl", id_key="seed_id")
    traj = open(run_dir / "trajectory.jsonl", "a")
    done = ckpt.done_ids()
    log.info("%d seeds, %d already done", len(seeds), len(done))

    def det_score(name, desc, ctx):
        if detector is None:
            return None
        return float(detector.score([{"tool_name": name, "description": desc, "context": ctx,
                                       "label": 1, "server": ""}])[0])

    def det_margins(name, descs, ctx):
        if detector is None:
            return [None] * len(descs)
        return [float(x) for x in detector.margin([{"tool_name": name, "description": d, "context": ctx,
                                                    "label": 1, "server": ""} for d in descs])]

    def process_seed(e, spec):
        reg_block = registries[e["server"]]["clean_system_promot"]
        # seed's own recorded queries first, then other recorded queries that use the same target tool
        qs = list(dict.fromkeys(list(e.get("queries") or []) + tool_queries.get((e["server"], spec["legit_tool"]), [])))
        queries = qs[:nq]
        seed_desc = e["description_stripped"]
        prev, best = None, None
        ctx = e.get("context") or []
        refs = sorted(benign_by_server.get(e["server"], []), key=len)[:2] if rich else []
        for t in range(T):
            if rich:
                # hill-climb: most candidates rewrite the best working version, one restarts from the seed
                base = best["desc"] if (hill and best and best["sel"] > -1) else seed_desc
                prompts = [build_attacker_prompt_rich(base if i < N - 1 else seed_desc, spec, prev) for i in range(N)]
            else:
                prompts = [build_attacker_prompt(seed_desc, spec, prev) for _ in range(N)]
            outs = attacker.generate(prompts, seed=cfg["project"]["seed"] + t, system=ATTACKER_SYSTEM,
                                     max_new_tokens=int(a.get("max_new_tokens", 256)))
            cands = []
            for raw_out in outs:
                desc = clean_rewrite(raw_out)
                if len(desc) < 15:
                    continue
                r = measure_asr(agent, reg_block, candidate_block(e["tool_name"], desc), queries, spec,
                                seed=cfg["project"]["seed"])
                ds = det_score(e["tool_name"], desc, ctx)
                mg = det_margins(e["tool_name"], [desc], ctx)[0] if (rich or use_margin) else None
                sc = selection_score_margin(r["asr"], mg, lam, floor) if use_margin else selection_score(r["asr"], ds, lam, floor)
                rec = {"desc": desc, "asr": r["asr"], "outcomes": r["outcomes"], "detector_score": ds,
                       "margin": mg, "sel": sc, "iter": t}
                cands.append(rec)
                traj.write(json.dumps({"seed_id": e["id"], **rec}) + "\n")
            if not cands:
                continue
            top = max(cands, key=lambda c: c["sel"])
            prev_best = best
            if best is None or top["sel"] > best["sel"]:
                best = top
            if rich:
                fb = best if best["sel"] > -1 else top
                sal = salient_words(lambda ds_: det_margins(e["tool_name"], ds_, ctx), fb["desc"]) if detector else []
                prev = {"asr": fb["asr"], "margin": fb.get("margin"), "salient": sal, "benign_refs": refs,
                        "prev_margin": prev_best.get("margin") if prev_best and prev_best is not fb else None}
            else:
                prev = {"asr": top["asr"], "detector_score": top["detector_score"], "threshold": 0.5}
            if a.get("early_stop") and best["sel"] > -1 and best["detector_score"] is not None \
                    and best["detector_score"] < 0.5:
                break   # found a working rewrite that the detector passes
        traj.flush()
        ckpt.append({"seed_id": e["id"], "server": e["server"], "tool_name": e["tool_name"],
                     "paradigm": e["paradigm"], "risk": e["risk"], "seed_asr": e.get("seed_asr"),
                     "spec": spec, "best": best, "seed_desc": seed_desc})
        atomic_write_json({"done": len(ckpt), "total": len(seeds)}, run_dir / "state.json")
        if len(ckpt) % 10 == 0:
            log.info("%d/%d seeds done", len(ckpt), len(seeds))

    n_err = 0
    for e, spec in seeds:
        if e["id"] in done:
            continue
        if (time.time() - t_start) / 3600 > max_h:
            log.warning("wall-clock guard (%.1f h) hit; stopping cleanly with %d/%d seeds", max_h, len(ckpt), len(seeds))
            break
        try:
            process_seed(e, spec)
        except Exception as ex:  # noqa: BLE001 - keep a multi-hour run alive
            n_err += 1
            log.exception("seed %s failed (%s); skipping", e["id"], type(ex).__name__)
            if n_err > max(5, len(seeds) // 10):
                raise RuntimeError("too many failing seeds; stopping") from ex
    traj.close()
    atomic_write_json({"done": len(ckpt), "total": len(seeds), "errors": n_err, "complete": len(ckpt) >= len(seeds)},
                      run_dir / "state.json")
    log.info("done: %s (%d seeds failed)", run_dir / "attacks.jsonl", n_err)


if __name__ == "__main__":
    main()
