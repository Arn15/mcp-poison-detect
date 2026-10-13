"""Step 4: run the implicit-poisoning optimizer against a target agent (+ optional detector in the loop).

Seeds = training-split poisoned tools (no test leakage). For each seed, iteratively rewrite its description
(attacker LLM), measure ASR on the agent over the server's clean queries, score with the in-loop detector,
and keep the best evasive-yet-working rewrite. Resumable (one record per seed).

Kaggle (GPU_DEPS=False; uses transformers):
  SCRIPT="scripts/run_attack.py"
  ARGS="--tag r0 --set attacker.max_seeds=250 attacker.detector_ckpt=/kaggle/input/<ds>/ckpt_ours_context+para"
Smoke (CPU): python scripts/run_attack.py --set attacker.backend=echo attacker.max_seeds=3 --tag smoke
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
from mcptp.attacker.optimize import ATTACKER_SYSTEM, build_attacker_prompt, clean_rewrite, selection_score
from mcptp.data.io import ensure_examples
from mcptp.data.paraphrase import make_backend
from mcptp.utils import JsonlCheckpoint, atomic_write_json


def main():
    import json as _json

    cfg, paths, run_dir, log = setup_run("run_attack", __doc__)
    a = cfg["attacker"]
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
        if e["split"] != "train" or e["label"] != 1:
            continue
        key = f"{e['server']}\x1f{e['tool_name']}\x1f{e['description_stripped'][:60]}"
        spec = specs.get(key) or specs.get(f"{e['server']}\x1f{e['tool_name']}\x1f{e['description'][:60]}")
        if spec and spec.get("legit_tool") and (spec.get("malicious_tools") or spec.get("injected_values")):
            seeds.append((e, spec))
    if a.get("max_seeds"):
        seeds = seeds[: int(a["max_seeds"])]
    log.info("usable seeds: %d", len(seeds))

    agent = make_backend({**a, "model": a["evaluator_model"], "device": a.get("agent_device", "cuda:1"),
                          "quant": a.get("agent_quant"), "temperature": 0.0, "max_new_tokens": 128})
    attacker = make_backend({**a, "model": a["attacker_model"],
                                                                                         "device": a.get("attacker_device", "cuda:0")})
    detector = None
    if a.get("detector_ckpt"):
        from mcptp.detector.train import LoadedDetector
        detector = LoadedDetector(a["detector_ckpt"])
        log.info("in-loop detector: %s", a["detector_ckpt"])

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

    def process_seed(e, spec):
        reg_block = registries[e["server"]]["clean_system_promot"]
        # seed's own recorded queries first, then other recorded queries that use the same target tool
        qs = list(dict.fromkeys(list(e.get("queries") or []) + tool_queries.get((e["server"], spec["legit_tool"]), [])))
        queries = qs[:nq]
        seed_desc = e["description_stripped"]
        prev, best = None, None
        for t in range(T):
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
                ds = det_score(e["tool_name"], desc, e.get("context") or [])
                sc = selection_score(r["asr"], ds, lam, floor)
                rec = {"desc": desc, "asr": r["asr"], "outcomes": r["outcomes"], "detector_score": ds,
                       "sel": sc, "iter": t}
                cands.append(rec)
                traj.write(json.dumps({"seed_id": e["id"], **rec}) + "\n")
            if not cands:
                continue
            top = max(cands, key=lambda c: c["sel"])
            if best is None or top["sel"] > best["sel"]:
                best = top
            prev = {"asr": top["asr"], "detector_score": top["detector_score"], "threshold": 0.5}
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
        try:
            process_seed(e, spec)
        except Exception as ex:  # noqa: BLE001 - keep a multi-hour run alive
            n_err += 1
            log.exception("seed %s failed (%s); skipping", e["id"], type(ex).__name__)
            if n_err > max(5, len(seeds) // 10):
                raise RuntimeError("too many failing seeds; stopping") from ex
    traj.close()
    atomic_write_json({"done": len(ckpt), "total": len(seeds), "errors": n_err, "complete": True},
                      run_dir / "state.json")
    log.info("done: %s (%d seeds failed)", run_dir / "attacks.jsonl", n_err)


if __name__ == "__main__":
    main()
