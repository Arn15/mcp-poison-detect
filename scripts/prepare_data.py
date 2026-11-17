"""Step 2: build detector examples from MCPTox and create / verify the frozen split.

    python scripts/prepare_data.py                 # build; freeze split if none exists, else verify it
    python scripts/prepare_data.py --semantic      # also use E5 semantic matching for registry context
    python scripts/prepare_data.py --refreeze      # deliberately replace the frozen split (avoid!)

Writes data/processed/examples.jsonl (gitignored, rebuilt deterministically), and
data/splits/split_<version>.json + data/splits/stats_<version>.json (tracked).
"""
from __future__ import annotations

import json
import sys
from collections import Counter

flags = {f for f in ("--semantic", "--refreeze") if f in sys.argv}
sys.argv = [a for a in sys.argv if a not in flags]

from common import setup_run  # noqa: E402

from mcptp.data.build import build_examples  # noqa: E402
from mcptp.data.io import annotate_splits  # noqa: E402
from mcptp.data.splits import (assign_splits, build_manifest, choose_server_split, load_manifest,  # noqa: E402
                               save_manifest, verify_manifest)


def main():
    """Entry point (see the module docstring for what this script does and how to run it)."""
    cfg, paths, run_dir, log = setup_run("prepare_data", __doc__)
    dcfg = cfg["data"]
    if "--semantic" in flags:
        from mcptp.data.context import make_e5_semantic
        cfg["context"]["semantic_match"] = True
        semantic = make_e5_semantic(cfg["context"]["embedder"])
    else:
        cfg["context"]["semantic_match"] = False
        semantic = None

    examples, _ = build_examples(cfg, paths["data_raw"], semantic=semantic)
    log.info("built %d examples", len(examples))

    version = dcfg["split_version"]
    mpath = paths["splits"] / f"split_{version}.json"
    if mpath.exists() and "--refreeze" not in flags:
        manifest = load_manifest(mpath)
        verify_manifest(examples, manifest)
        log.info("verified frozen split %s (test sha256 %s...)", mpath.name, manifest["test_sha256"][:12])
    else:
        sub = {k: dcfg[k] for k in ("mcptox_commit", "test_frac", "val_frac", "heldout_risk_families",
                                    "include_explicit", "hard_negatives_per_server")}
        sub["seed"] = cfg["project"]["seed"]
        server_split = choose_server_split(examples, dcfg["test_frac"], dcfg["val_frac"], cfg["project"]["seed"])
        ids = assign_splits(examples, server_split, dcfg["heldout_risk_families"])
        manifest = build_manifest(examples, server_split, ids, dcfg["heldout_risk_families"], sub, version)
        save_manifest(manifest, mpath, overwrite="--refreeze" in flags)
        log.warning("FROZE new split %s (test sha256 %s...). Commit data/splits/ now.", mpath.name,
                    manifest["test_sha256"][:12])

    annotate_splits(examples, manifest)

    out = paths["data_processed"] / "examples.jsonl"
    with open(out, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")

    # ---- stats (tracked, small) ----
    stats = {"version": version, "n_examples": len(examples), "per_split": {}}
    for s in ("train", "val", "test"):
        es = [e for e in examples if e["split"] == s]
        stats["per_split"][s] = {
            "servers": len(manifest["servers"][s]),
            "kind": dict(Counter(e["kind"] for e in es)),
            "paradigm": dict(Counter(e["paradigm"] for e in es if e["label"] == 1)),
            "hard_negatives": sum(e["is_hard_negative"] for e in es),
            "pos_with_marker": sum(e["has_marker"] for e in es if e["label"] == 1),
            "ref_method_pos": dict(Counter(e["ref_method"] for e in es if e["label"] == 1)),
            "ref_method_neg": dict(Counter(e["ref_method"] for e in es if e["label"] == 0)),
        }
    t = [e for e in examples if e["split"] == "test"]
    stats["test_subsets"] = {
        "heldout_server_pos": sum(e["label"] == 1 and e["heldout_server"] and not e["heldout_family"] for e in t),
        "heldout_family_pos": sum(e["heldout_family"] for e in t),
        "neg": sum(e["label"] == 0 for e in t),
    }
    (paths["splits"] / f"stats_{version}.json").write_text(json.dumps(stats, indent=1))
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    log.info(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
