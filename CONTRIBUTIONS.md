# Code authorship log

The course requires stating which code is my own. Update this file in the same commit as the code.

Legend: **Own** = written by me for this project · **Adapted** = written by me, following the cited reference's
design · **Library** = called via a third-party package, not reimplemented.

| File / module | Status | Function | Reference (if any) |
|---|---|---|---|
| `mcptp/utils/config.py` | Own | YAML config + dotted CLI overrides | — |
| `mcptp/utils/env.py` | Own | Runtime detection, storage paths for local/Kaggle/Colab | — |
| `mcptp/utils/seed.py` | Own | Seeding | — |
| `mcptp/utils/checkpoint.py` | Own | Resumable append-only JSONL store, atomic JSON | — |
| `mcptp/utils/logging.py` | Own | Logger, timestamped run dirs | — |
| `scripts/common.py` | Own | Shared CLI and run setup | — |
| `scripts/check_env.py` | Own | Environment report | — |
| `scripts/smoke_test.py` | Own | Infrastructure smoke test | — |
| `notebooks/*_runner.ipynb` | Own | Thin GPU runners | — |
| *(Step 2+)* `mcptp/data/…` | | | MCPTox format |
| *(Step 4)* `mcptp/attacker/…` | Adapted | MCP-ITP-style optimizer | Li et al., arXiv 2601.07395 |
| *(Step 5)* `mcptp/baselines/e5.py` | Adapted | MCP-Guard Stage II classifier | Xing et al., arXiv 2508.10991 |
| — | Library | DeBERTa-v3 / E5 weights, HF Transformers, vLLM, scikit-learn | — |
