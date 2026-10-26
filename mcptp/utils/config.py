"""YAML config loading with dotted-key CLI overrides (e.g. --set detector.lr=3e-5)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def _parse_value(raw: str) -> Any:
    # Reuse YAML's scalar parsing so "3e-5", "true", "[a, b]", "null" behave as expected.
    val = yaml.safe_load(raw)
    if isinstance(val, str):
        try:  # YAML 1.1 reads "3e-5" (no dot) as a string
            return float(val)
        except ValueError:
            pass
    return val


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = _parse_value(raw)
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml(path: Path) -> dict:
    """Load a YAML config; `_base: other.yaml` (relative to this file) is loaded first and deep-merged under it."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base", None)
    return _deep_merge(_load_yaml(Path(path).parent / base), cfg) if base else cfg


def load_config(path: str | Path = "configs/default.yaml", overrides: Iterable[str] | None = None) -> dict:
    return apply_overrides(_load_yaml(Path(path)), overrides or [])


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
