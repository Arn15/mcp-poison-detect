"""Resumable job state.

Long jobs (attacker loop, agent eval, LLM-judge baseline) write one JSON record per
finished item to an append-only JSONL file. On restart they call `done_ids()` and skip
what is already finished, so a Kaggle session timeout loses at most one item.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


def atomic_write_json(obj: Any, path: str | Path) -> None:
    """Write JSON so a crash mid-write never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file, or return default if it does not exist."""
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


class JsonlCheckpoint:
    """Append-only record store keyed by `id_key`."""

    def __init__(self, path: str | Path, id_key: str = "id"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.id_key = id_key

    def __iter__(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Last line may be partial if the session was killed mid-write; skip it.
                    continue

    def done_ids(self) -> set:
        return {r[self.id_key] for r in self if self.id_key in r}

    def append(self, record: dict) -> None:
        if self.id_key not in record:
            raise KeyError(f"record missing {self.id_key!r}")
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def records(self) -> list[dict]:
        return list(self)

    def __len__(self) -> int:
        return sum(1 for _ in self)
