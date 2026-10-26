"""CPU tests for the judge variants (no model load)."""
import json

from mcptp.baselines.llm_judge import build_messages, select_shots
from mcptp.utils.config import load_config
from mcptp.utils.env import REPO_ROOT


def _examples():
    return [json.loads(l) for l in open(REPO_ROOT / "data" / "processed" / "examples.jsonl")]


def _base_ids():
    ids = set()
    for f in (REPO_ROOT / "data" / "attacks").glob("*/attacks.jsonl"):
        ids |= {a["seed_id"] for a in map(json.loads, open(f)) if a}
    return ids


def test_judges_config_inherits_default():
    cfg = load_config(REPO_ROOT / "configs" / "judges.yaml")
    assert cfg["para"]["version"] and cfg["detector"]["backbone"]          # inherited
    assert [j["shots"] for j in cfg["score"]["judges"]] == [0, 6, 0, 6]


def test_shots_train_only_and_exclude_attack_bases():
    ex, base = _examples(), _base_ids()
    by_id = {e["id"]: e for e in ex}
    shots = select_shots(ex, 3, 3, exclude_ids=base, seed=42)
    assert len(shots) == 6 and sum(s["label"] for s in shots) == 3
    for s in shots:
        assert by_id[s["id"]]["split"] == "train" and s["id"] not in base
    assert any(by_id[s["id"]].get("is_hard_negative") for s in shots if s["label"] == 0)
    assert [s["label"] for s in shots] == [1, 0, 1, 0, 1, 0]
    assert shots == select_shots(ex, 3, 3, exclude_ids=base, seed=42)     # deterministic


def test_messages_layout():
    ex = _examples()
    shots = select_shots(ex, 1, 1, seed=0)
    m = build_messages({"tool_name": "t", "description": "d", "registry_names": ["a"]}, shots)
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert m[-1]["content"].count("Candidate tool: t") == 1
    assert build_messages({"tool_name": "t", "description": "d"})[1]["content"].startswith("Registered tools")
