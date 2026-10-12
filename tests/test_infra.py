import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcptp.utils import JsonlCheckpoint, atomic_write_json, load_config, read_json, set_seed  # noqa: E402
from mcptp.utils.config import apply_overrides  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    cfg = load_config(ROOT / "configs/default.yaml")
    for section in ["project", "paths", "data", "detector", "attacker", "eval", "checkpoint"]:
        assert section in cfg
    assert cfg["eval"]["target_fpr"] == 0.01


def test_overrides_parse_types():
    cfg = apply_overrides({"detector": {"lr": 1e-5}}, ["detector.lr=3e-5", "detector.use_context=false", "new.key=[a, b]"])
    assert cfg["detector"]["lr"] == 3e-5
    assert cfg["detector"]["use_context"] is False
    assert cfg["new"]["key"] == ["a", "b"]


def test_checkpoint_resume_and_partial_line(tmp_path):
    ck = JsonlCheckpoint(tmp_path / "x.jsonl")
    ck.append({"id": 1, "v": 0.1})
    ck.append({"id": 2, "v": 0.2})
    with open(ck.path, "a") as f:
        f.write('{"id": 3, "v":')  # simulated kill mid-write
    assert ck.done_ids() == {1, 2}


def test_atomic_json(tmp_path):
    p = tmp_path / "s.json"
    atomic_write_json({"a": 1}, p)
    assert read_json(p) == {"a": 1}
    assert read_json(tmp_path / "missing.json", default={}) == {}


def test_seed_reproducible():
    import numpy as np
    set_seed(0); a = np.random.rand(3)
    set_seed(0); b = np.random.rand(3)
    assert (a == b).all()
