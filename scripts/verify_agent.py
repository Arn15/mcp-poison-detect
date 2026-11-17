"""Step 4.1 check: does our agent harness reproduce MCPTox's recorded outcomes?

Like-for-like: for each sampled (instance, query) we use MCPTox's own recorded system prompt and query,
run our target agent (greedy), judge with our classifier, and compare to MCPTox's recorded label for the
same model. Sampling is stratified: all recorded successes for the model (that have a usable attack spec)
plus an equal number of recorded failures, so agreement on *success* is actually measurable.

Kaggle: SCRIPT="scripts/verify_agent.py" ARGS="--tag verify" GPU_DEPS=True
"""
from __future__ import annotations

import json
import random
import warnings

from common import setup_run

from mcptp.attacker.agent import classify, derive_specs_from_mcptox, extract_tool_calls
from mcptp.data.mcptox import ensure_mcptox, parse_tool_block
from mcptp.data.paraphrase import make_backend

warnings.filterwarnings("ignore", category=SyntaxWarning)


def main():
    """Entry point (see the module docstring for what this script does and how to run it)."""
    cfg, paths, run_dir, log = setup_run("verify_agent", __doc__)
    a = cfg["attacker"]
    raw = json.load(open(ensure_mcptox(paths["data_raw"]) / "response_all.json"))
    specs = derive_specs_from_mcptox(raw)
    model_key = a.get("verify_model", "Qwen3-8b-No-Think")

    pos, neg = [], []
    for sname, sv in raw["servers"].items():
        for mi in sv["malicious_instance"]:
            p = parse_tool_block(mi["poisoned_tool"] + "\n")
            if not p:
                continue
            spec = specs.get(f"{sname}\x1f{p[0]['tool_name']}\x1f{p[0]['description'][:60]}")
            if not (spec and spec.get("legit_tool") and (spec.get("malicious_tools") or spec.get("injected_values"))):
                continue
            for d in mi["datas"]:
                lab = (d.get("label") or {}).get(model_key)
                if not lab or lab == "None":
                    continue
                item = {"server": sname, "tool": p[0]["tool_name"], "paradigm": spec["paradigm"], "spec": spec,
                        "system": d["system"], "query": d["query"], "gold": lab}
                (pos if lab == "Success" else neg).append(item)
    rng = random.Random(cfg["project"]["seed"])
    k = min(len(pos), int(a.get("verify_n_pos", 60)))
    sample = rng.sample(pos, k) + rng.sample(neg, min(len(neg), k))
    log.info("verifying %d queries (%d recorded successes) vs MCPTox[%s]", len(sample), k, model_key)

    agent = make_backend({**a, "model": a["evaluator_model"], "device": a.get("agent_device", "cuda:1"),
                          "quant": a.get("agent_quant"), "temperature": 0.0, "max_new_tokens": 128})
    rows, bs = [], 4
    for i in range(0, len(sample), bs):
        chunk = sample[i:i + bs]
        # each item has its own system prompt, so generate one at a time within the chunk
        for it in chunk:
            resp = agent.generate([it["query"]], seed=cfg["project"]["seed"], system=it["system"], max_new_tokens=128)[0]
            ours = classify(extract_tool_calls(resp), it["spec"], it["query"])
            rows.append({k2: it[k2] for k2 in ("server", "tool", "paradigm", "query", "gold")} |
                        {"ours": ours, "response": resp[:400]})
        log.info("%d/%d", len(rows), len(sample))

    (run_dir / "verify.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    g = [r["gold"] == "Success" for r in rows]
    o = [r["ours"] == "success" for r in rows]
    tp = sum(x and y for x, y in zip(g, o)); fp = sum((not x) and y for x, y in zip(g, o))
    fn = sum(x and (not y) for x, y in zip(g, o)); tn = len(rows) - tp - fp - fn
    summary = {"n": len(rows), "gold_success": sum(g), "ours_success": sum(o),
               "agreement": round((tp + tn) / len(rows), 3),
               "success_precision": round(tp / (tp + fp), 3) if tp + fp else None,
               "success_recall": round(tp / (tp + fn), 3) if tp + fn else None,
               "tp": tp, "fp": fp, "fn": fn, "tn": tn, "model_key": model_key,
               "evaluator": a["evaluator_model"], "quant": a.get("agent_quant")}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("ASR agreement: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
