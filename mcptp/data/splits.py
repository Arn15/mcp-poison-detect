"""Server-disjoint splits with held-out attack families, frozen with a SHA-256 manifest.

* Servers are partitioned into train / val / test, so every variant of an attack and every tool of a
  registry stays on one side (random splits inflate AUROC by up to 26 pts, MCPShield 2026).
* Held-out risk families: every poisoned example in these families goes to TEST (from any server) and
  never to train/val. Each test example carries `heldout_server` / `heldout_family` flags so results can
  be reported per subset.
* The manifest stores ids per split and a hash over (id, label, description) of the test set. Training
  code calls `verify_manifest` and refuses to run if the data drifted or a test id leaks into training.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "val", "test")


def choose_server_split(examples: list[dict], test_frac: float, val_frac: float, seed: int,
                        n_trials: int = 20000, required_test_paradigms=("Template-2", "Template-3")) -> dict:
    """Pick a server partition whose positive and negative shares are closest to the targets."""
    pos, neg, par = Counter(), Counter(), {}
    for e in examples:
        (pos if e["label"] == 1 else neg)[e["server"]] += 1
        if e["label"] == 1:
            par.setdefault(e["server"], set()).add(e["paradigm"])
    servers = sorted(set(pos) | set(neg))
    P, N = sum(pos.values()), sum(neg.values())
    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(n_trials):
        order = servers[:]
        rng.shuffle(order)
        assign, tp, tn, vp, vn = {}, 0, 0, 0, 0
        for s in order:
            if tp / P < test_frac:
                assign[s] = "test"; tp += pos[s]; tn += neg[s]
            elif vp / P < val_frac:
                assign[s] = "val"; vp += pos[s]; vn += neg[s]
            else:
                assign[s] = "train"
        test_par = set().union(*[par.get(s, set()) for s, v in assign.items() if v == "test"])
        if not set(required_test_paradigms) <= test_par:
            continue
        score = abs(tp / P - test_frac) + abs(tn / N - test_frac) + abs(vp / P - val_frac) + abs(vn / N - val_frac)
        if score < best_score:
            best, best_score = assign, score
    if best is None:
        raise RuntimeError("no valid server split found")
    return best


def assign_splits(examples: list[dict], server_split: dict, heldout_families: list[str]) -> dict[str, list[str]]:
    held = set(heldout_families)
    ids = {s: [] for s in SPLITS}
    for e in examples:
        split = server_split[e["server"]]
        fam = e["label"] == 1 and e.get("risk") in held
        if fam:
            split = "test"
        ids[split].append(e["id"])
    return {k: sorted(v) for k, v in ids.items()}


def test_hash(examples: list[dict], test_ids: list[str]) -> str:
    by_id = {e["id"]: e for e in examples}
    h = hashlib.sha256()
    for i in sorted(test_ids):
        e = by_id[i]
        h.update(f"{i}\x1f{e['label']}\x1f{e['description']}\n".encode())
    return h.hexdigest()


def build_manifest(examples, server_split, ids, heldout_families, cfg_subset: dict, version: str) -> dict:
    return {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg_subset,
        "heldout_families": sorted(heldout_families),
        "servers": {s: sorted(k for k, v in server_split.items() if v == s) for s in SPLITS},
        "counts": {s: {"n": len(ids[s]),
                       "pos": sum(1 for e in examples if e["id"] in set(ids[s]) and e["label"] == 1)} for s in SPLITS},
        "test_sha256": test_hash(examples, ids["test"]),
        "ids": ids,
    }


def save_manifest(manifest: dict, path: str | Path, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists: the test set is frozen. Pass --refreeze to replace it deliberately.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def verify_manifest(examples: list[dict], manifest: dict) -> None:
    """Raise if test data changed, ids are missing, or splits overlap."""
    by_id = {e["id"] for e in examples}
    ids = manifest["ids"]
    for s in SPLITS:
        missing = set(ids[s]) - by_id
        if missing:
            raise ValueError(f"{len(missing)} {s} ids missing from examples (data drift), e.g. {sorted(missing)[:3]}")
    for a in SPLITS:
        for b in SPLITS:
            if a < b and set(ids[a]) & set(ids[b]):
                raise ValueError(f"split overlap between {a} and {b}")
    if test_hash(examples, ids["test"]) != manifest["test_sha256"]:
        raise ValueError("test-set hash mismatch: test examples changed since freezing")


def split_of(manifest: dict) -> dict[str, str]:
    return {i: s for s in SPLITS for i in manifest["ids"][s]}
