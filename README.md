# Context-Aware, Adversarially Trained Detection of Implicit MCP Tool Poisoning

NCSU ECE 592 / CSC 591 (Advanced Topics in Deep Learning), Fall 2026 — Arnav Yennam.

A black-box, pre-registration detector that reads a candidate MCP tool's description **together with the
co-registered tools it references**, hardened by adversarial training against our own MCP-ITP-style
implicit-poisoning optimizer. Evaluated on MCPTox with server-disjoint splits, TPR@1%FPR, hard-negative FPR,
held-out attack families, and a final adaptive attack.

> Status: **Step 1 (infrastructure) complete.** Data pipeline, detector, attacker, baselines and eval land in later steps.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/default.yaml` | Single source of truth for every experiment setting |
| `mcptp/utils/` | Config loading + CLI overrides, runtime/path detection, seeding, resumable checkpoints, run dirs |
| `mcptp/data/` | MCPTox loading, hard negatives, registry context, server-disjoint splits *(Step 2)* |
| `mcptp/detector/` | Context-aware DeBERTa-v3 detector *(Steps 3, 7)* |
| `mcptp/attacker/` | MCP-ITP-style attacker / evaluator loop *(Step 4)* |
| `mcptp/baselines/` | B1 E5 classifier, B2 LLM judge, B4 MindGuard (optional) *(Step 5)* |
| `mcptp/eval/` | Metrics, bootstrap CIs, tables and plots *(Steps 3, 9)* |
| `scripts/` | One entry point per experiment; all share `scripts/common.py` flags |
| `notebooks/` | Thin Kaggle / Colab runners (pull repo → run one script) |
| `data/splits/` | Frozen test-set manifest (tracked in git) |
| `tests/` | CPU unit tests |

## Quick start

```bash
pip install -r requirements.txt && pip install -e . --no-deps
python -m pytest -q                      # unit tests (CPU)
python scripts/check_env.py              # runtime, GPU, package versions
python scripts/smoke_test.py --tag local # resumable-loop smoke test
```

Every script accepts the same flags:

```bash
python scripts/<script>.py --config configs/default.yaml \
    --set detector.use_context=false detector.lr=3e-5 \
    --tag b3_plain \
    --resume <existing_run_dir_name>
```

Each run writes to `outputs/<stage>/<timestamp>_<tag>/` with `config.yaml` (incl. git commit), `run.log`
and its results. On Kaggle this is `/kaggle/working/outputs`; on Colab it is Google Drive.

## Running on GPU (Kaggle / Colab)

See [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Code authorship

Required by the course: which code is my own and which is referenced. Kept up to date in
[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

## Data

MCPTox (Wang et al., AAAI 2026): https://github.com/zhiqiangwang4/MCPTox-Benchmark. Not redistributed here.
