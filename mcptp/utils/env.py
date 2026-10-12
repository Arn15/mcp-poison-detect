"""Detect the runtime (local / kaggle / colab) and resolve where data and outputs live.

Kaggle : repo in /kaggle/working/<repo>; outputs -> /kaggle/working/outputs (save as a Dataset version);
         read-only inputs (MCPTox, previous rounds) from /kaggle/input/<dataset>.
Colab  : outputs -> /content/drive/MyDrive/mcp-poison-detect/outputs so they survive disconnects.
Local  : everything relative to the repo root.
Override any of these with the env var MCPTP_ROOT.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def detect_runtime() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content/sample_data").exists():
        return "colab"
    return "local"


def storage_root(runtime: str | None = None) -> Path:
    if os.environ.get("MCPTP_ROOT"):
        return Path(os.environ["MCPTP_ROOT"])
    runtime = runtime or detect_runtime()
    if runtime == "kaggle":
        return Path("/kaggle/working")
    if runtime == "colab" and Path("/content/drive/MyDrive").exists():
        return Path("/content/drive/MyDrive/mcp-poison-detect")
    return REPO_ROOT


def resolve_paths(cfg: dict) -> dict:
    """Return {name: absolute Path} for cfg['paths'], creating directories.

    Tracked inputs (data/splits) always resolve inside the repo so the frozen test
    manifest is versioned; everything else resolves under the runtime storage root.
    """
    root = storage_root()
    out = {}
    for name, rel in cfg.get("paths", {}).items():
        base = REPO_ROOT if name == "splits" else root
        p = Path(rel) if Path(rel).is_absolute() else base / rel
        p.mkdir(parents=True, exist_ok=True)
        out[name] = p
    if "hf_cache" in out:
        os.environ.setdefault("HF_HOME", str(out["hf_cache"]))
    return out
