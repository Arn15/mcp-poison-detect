"""Logging and per-run output directories (outputs/<stage>/<timestamp>_<tag>/)."""
from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_logger(name: str = "mcptp", log_file: str | Path | None = None) -> logging.Logger:
    """Logger writing to stdout and optionally to a run.log file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def git_commit() -> str:
    """Short hash of the current git commit (recorded in every run's config)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def make_run_dir(outputs_root: str | Path, stage: str, tag: str = "", resume: str | None = None) -> Path:
    """Create outputs/<stage>/<YYYYmmdd-HHMMSS>[_tag]; pass `resume` (an existing dir name) to reuse one."""
    base = Path(outputs_root) / stage
    if resume:
        d = base / resume
        if not d.exists():
            raise FileNotFoundError(d)
        return d
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{tag}" if tag else "")
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    return d
