"""Step 4.1 check: does our agent harness reproduce MCPTox's reported ASR?

Runs the target agent (evaluator_model) on the ORIGINAL poisoned descriptions over each server's clean
queries, measures ASR with our judge, and compares to MCPTox's recorded ASR for the same model family.
If our ASR tracks theirs (low mean abs error, positive correlation), the harness is trustworthy.

Kaggle: SCRIPT="scripts/verify_agent.py" ARGS="--tag verify --set attacker.verify_model=Qwen3-8b-No-Think attacker.max_seeds=60"
"""
from __future__ import annotations

import json

from common import setup_run

from mcptp.attacker.agent import derive_specs_from_mcptox, extract_tool_calls, classify
from mcptp.attacker.evaluate import candidate_block, measure_asr
from mcptp.data.mcptox import parse_tool_block
from mcptp.data.paraphrase import make_backend


def mcptox_asr(mi, model_key):
    """Recorded ASR for one instance for a given MCPTox model column."""
    labs = [(_d.get("label") or {}).get(model_key) for _d in mi["datas"]]
    labs = [l for l in labs if l and l != "None"]
    return (sum(l == "Success" for l in labs) / len(labs)) if labs else None


def main():
    import numpy as np

    cfg, paths, run_dir, log = setup_run("verify_agent", __doc__)
    a = cfg["attacker"]
    from mcptp.data.mcptox import ensure_mcptox
    raw = json.load(open(ensure_mcptox(paths["data_raw"]) / "response_all.json"))
    specs = derive_specs_from_mcptox(raw)
    model_key = a.get("verify_model", "Qwen3-8b-No-Think")
    agent = make_backend({**a, "model": a["evaluator_model"], "device": a.get("agent_device", "cuda:1"),
                          "quant": a.get("agent_quant"), "temperature": 0.0, "max_new_tokens": 128})

    rows = []
    insts = []
    for sname, sv in raw["servers"].items():
        for mi in sv["malicious_instance"]:
            insts.append((sname, sv, mi))
    if a.get("max_seeds"):
        insts = insts[:: max(1, len(insts) // int(a["max_seeds"]))][: int(a["max_seeds"])]
    log.info("verifying on %d instances vs MCPTox[%s]", len(insts), model_key)

    for sname, sv, mi in insts:
        parsed = parse_tool_block(mi["poisoned_tool"] + "\n")
        if not parsed:
            continue
        key = f"{sname}\x1f{parsed[0]['tool_name']}\x1f{parsed[0]['description'][:60]}"
        spec = specs.get(key)
        if not spec or not spec.get("legit_tool"):
            continue
        gold = mcptox_asr(mi, model_key)
        if gold is None:
            continue
        queries = (sv.get("clean_querys") or [])[: int(a.get("shadow_queries", 5))]
        r = measure_asr(agent, sv["clean_system_promot"],
                        candidate_block(parsed[0]["tool_name"], parsed[0]["description"]), queries, spec)
        rows.append({"server": sname, "tool": parsed[0]["tool_name"], "paradigm": spec["paradigm"],
                     "ours_asr": r["asr"], "mcptox_asr": gold, "n": r["n"], "outcomes": r["outcomes"]})

    (run_dir / "verify.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    o = np.array([x["ours_asr"] for x in rows]); m = np.array([x["mcptox_asr"] for x in rows])
    mae = float(np.mean(np.abs(o - m)))
    corr = float(np.corrcoef(o, m)[0, 1]) if len(o) > 2 and o.std() and m.std() else float("nan")
    summary = {"n": len(rows), "mae": round(mae, 3), "corr": round(corr, 3),
               "ours_mean": round(float(o.mean()), 3), "mcptox_mean": round(float(m.mean()), 3),
               "model_key": model_key, "evaluator": a["evaluator_model"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("ASR agreement: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
