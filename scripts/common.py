"""Shared CLI boilerplate so every script takes the same flags:

    python scripts/<name>.py --config configs/default.yaml --set a.b=1 --tag pilot --resume <run_dir_name>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow running without pip install

from mcptp.utils import get_logger, load_config, make_run_dir, resolve_paths, save_config, set_seed  # noqa: E402
from mcptp.utils.env import detect_runtime  # noqa: E402
from mcptp.utils.logging import git_commit  # noqa: E402


def parse_args(description: str = "") -> argparse.Namespace:
    """Shared flags: --config, --set KEY=VALUE ..., --tag, --resume."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted-key config overrides")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--resume", default=None, help="existing run dir name under outputs/<stage>/ to continue")
    return ap.parse_args()


def setup_run(stage: str, description: str = ""):
    """Parse args, load config, seed, create run dir, snapshot config. Returns (cfg, paths, run_dir, log)."""
    args = parse_args(description)
    cfg = load_config(args.config, args.set)
    paths = resolve_paths(cfg)
    set_seed(cfg["project"]["seed"])
    run_dir = make_run_dir(paths["outputs"], stage, args.tag, args.resume)
    log = get_logger(stage, run_dir / "run.log")
    cfg["_run"] = {"stage": stage, "runtime": detect_runtime(), "git_commit": git_commit(), "run_dir": str(run_dir)}
    save_config(cfg, run_dir / "config.yaml")
    log.info("stage=%s runtime=%s commit=%s run_dir=%s", stage, cfg["_run"]["runtime"], cfg["_run"]["git_commit"], run_dir)
    return cfg, paths, run_dir, log
