"""Print runtime, GPU, package versions and resolved paths. Run first in every new Kaggle/Colab session."""
from __future__ import annotations

import importlib
import json
import platform

from common import setup_run

PKGS = ["torch", "transformers", "sentence_transformers", "datasets", "accelerate", "vllm", "sklearn", "pandas", "numpy"]


def main():
    cfg, paths, run_dir, log = setup_run("env_check", __doc__)
    report = {"python": platform.python_version(), "runtime": cfg["_run"]["runtime"], "packages": {}, "gpu": None,
              "paths": {k: str(v) for k, v in paths.items()}}
    for name in PKGS:
        try:
            report["packages"][name] = getattr(importlib.import_module(name), "__version__", "installed")
        except Exception:
            report["packages"][name] = None
    try:
        import torch

        if torch.cuda.is_available():
            report["gpu"] = [{"name": torch.cuda.get_device_name(i),
                              "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                             for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    (run_dir / "env.json").write_text(json.dumps(report, indent=2))
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
