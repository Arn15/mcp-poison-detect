"""Load processed examples with the frozen split enforced (rebuilding them deterministically if missing)."""
from __future__ import annotations

import json
from pathlib import Path

from .splits import load_manifest, split_of, verify_manifest


def annotate_splits(examples: list[dict], manifest: dict) -> list[dict]:
    where = split_of(manifest)
    test_servers = set(manifest["servers"]["test"])
    held = set(manifest["heldout_families"])
    for e in examples:
        e["split"] = where[e["id"]]
        e["heldout_server"] = e["server"] in test_servers
        e["heldout_family"] = e["label"] == 1 and e.get("risk") in held
    return examples


def ensure_examples(cfg: dict, paths: dict) -> list[dict]:
    """Return examples; on a fresh machine (e.g. Kaggle) rebuild them from MCPTox and verify the frozen hash."""
    version = cfg["data"]["split_version"]
    path = Path(paths["data_processed"]) / "examples.jsonl"
    if path.exists():
        return load_examples(paths["data_processed"], paths["splits"], version)
    from .build import build_examples
    c = {**cfg, "context": {**cfg["context"], "semantic_match": False}}
    examples, _ = build_examples(c, paths["data_raw"])
    manifest = load_manifest(Path(paths["splits"]) / f"split_{version}.json")
    verify_manifest(examples, manifest)
    annotate_splits(examples, manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    return examples


def load_examples(processed_dir, splits_dir, version: str = "v1") -> list[dict]:
    path = Path(processed_dir) / "examples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/prepare_data.py first")
    examples = [json.loads(line) for line in open(path)]
    verify_manifest(examples, load_manifest(Path(splits_dir) / f"split_{version}.json"))
    return examples


def load_para(splits_dir, version: str = "v1", split: str | None = None) -> list[dict]:
    """Load frozen MCPTox-Para examples (data/para/para_<version>.jsonl), verifying per-split hashes."""
    import hashlib

    from ..utils.env import REPO_ROOT

    path = REPO_ROOT / "data" / "para" / f"para_{version}.jsonl"
    man = json.loads((Path(splits_dir) / f"para_{version}.json").read_text())
    recs = [json.loads(line) for line in open(path)]
    for s, info in man["per_split"].items():
        h = hashlib.sha256()
        for r in sorted((r for r in recs if r["split"] == s), key=lambda r: r["id"]):
            h.update(f"{r['id']}\x1f{r['label']}\x1f{r['description']}\n".encode())
        if h.hexdigest() != info["sha256"]:
            raise ValueError(f"MCPTox-Para {s} hash mismatch")
    return [r for r in recs if split is None or r["split"] == split]
