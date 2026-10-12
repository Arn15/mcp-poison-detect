"""Load processed examples with the frozen split enforced."""
from __future__ import annotations

import json
from pathlib import Path

from .splits import load_manifest, verify_manifest


def load_examples(processed_dir, splits_dir, version: str = "v1") -> list[dict]:
    path = Path(processed_dir) / "examples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/prepare_data.py first")
    examples = [json.loads(line) for line in open(path)]
    verify_manifest(examples, load_manifest(Path(splits_dir) / f"split_{version}.json"))
    return examples
