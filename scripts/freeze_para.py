"""Step 2b: turn raw paraphrase generations into frozen MCPTox-Para example files.

    python scripts/freeze_para.py --inputs path/to/generations.jsonl [more.jsonl ...] [--refreeze]

Keeps valid generations only, copies labels/metadata from the original example, recomputes registry
context from the paraphrased text, and writes (tracked in git, small):
  data/para/para_<version>.jsonl        paraphrased examples (split inherited from the original)
  data/splits/para_<version>.json       per-split SHA-256 + validity stats
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

args = sys.argv[1:]
refreeze = "--refreeze" in args
inputs = []
if "--inputs" in args:
    i = args.index("--inputs") + 1
    while i < len(args) and not args[i].startswith("--"):
        inputs.append(args[i]); i += 1
sys.argv = [sys.argv[0]] + [a for a in args if a not in inputs and a not in ("--inputs", "--refreeze")]

from common import setup_run  # noqa: E402

from mcptp.data.context import find_references  # noqa: E402
from mcptp.data.io import ensure_examples  # noqa: E402
from mcptp.data.mcptox import strip_injection_markers  # noqa: E402
from mcptp.utils.env import REPO_ROOT  # noqa: E402


def split_hash(recs):
    h = hashlib.sha256()
    for r in sorted(recs, key=lambda r: r["id"]):
        h.update(f"{r['id']}\x1f{r['label']}\x1f{r['description']}\n".encode())
    return h.hexdigest()


def main():
    if not inputs:
        raise SystemExit("pass --inputs <generations.jsonl> ...")
    cfg, paths, run_dir, log = setup_run("freeze_para", __doc__)
    version = cfg["para"]["version"]
    out_path = REPO_ROOT / "data" / "para" / f"para_{version}.jsonl"
    man_path = paths["splits"] / f"para_{version}.json"
    if man_path.exists() and not refreeze:
        raise SystemExit(f"{man_path} exists (frozen). Use --refreeze to replace it deliberately.")

    examples = {e["id"]: e for e in ensure_examples(cfg, paths)}
    registries = defaultdict(dict)
    for e in examples.values():
        if e["kind"] == "benign":
            registries[e["server"]][e["tool_name"]] = e["description"]

    gens = {}
    for path in inputs:
        for line in open(path):
            if line.strip():
                g = json.loads(line)
                gens[g["job"]] = g  # later files win
    stats = Counter()
    by_label = defaultdict(Counter)
    recs = []
    for g in sorted(gens.values(), key=lambda g: g["job"]):
        by_label[g["label"]]["total"] += 1
        if not g["valid"]:
            by_label[g["label"]]["invalid"] += 1
            for k in ("names_ok", "copy", "refusal"):
                if (k == "names_ok") != bool(g[k]):
                    stats[f"fail_{k}"] += 1
            continue
        o = examples[g["orig_id"]]
        r = {k: v for k, v in o.items() if k not in ("context", "referenced_tools", "ref_method")}
        refs, method = find_references(g["description"], registries[o["server"]],
                                       exclude=o["tool_name"] if o["kind"] == "benign" else None,
                                       top_k=cfg["context"]["top_k"])
        reg = registries[o["server"]]
        r.update({"id": g["id"], "orig_id": o["id"], "is_paraphrase": True, "para_style": g["style"],
                  "para_model": g["model"], "description": g["description"],
                  "description_stripped": strip_injection_markers(g["description"]),
                  "has_marker": False, "referenced_tools": refs, "ref_method": method,
                  "context": [{"tool_name": t, "description": reg[t]} for t in refs]})
        recs.append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    manifest = {"version": version, "inputs": [str(p) for p in inputs],
                "per_split": {s: {"n": sum(r["split"] == s for r in recs),
                                  "pos": sum(r["split"] == s and r["label"] == 1 for r in recs),
                                  "sha256": split_hash([r for r in recs if r["split"] == s])}
                              for s in ("train", "val", "test") if any(r["split"] == s for r in recs)},
                "validity": {str(k): dict(v) for k, v in by_label.items()}, "fail_reasons": dict(stats)}
    man_path.write_text(json.dumps(manifest, indent=1))
    (run_dir / "para_manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info(json.dumps(manifest, indent=1))
    for lab, c in by_label.items():
        rate = c["invalid"] / max(1, c["total"])
        if rate > 0.15:
            log.warning("label %s: %.0f%% of generations invalid -> check for selection bias", lab, 100 * rate)


if __name__ == "__main__":
    main()
