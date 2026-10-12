# Context-Aware, Adversarially Trained Detection of Implicit MCP Tool Poisoning

NCSU ECE 592 / CSC 591 (Advanced Topics in Deep Learning), Fall 2026 — Arnav Yennam.

A black-box, pre-registration detector that reads a candidate MCP tool's description **together with the
co-registered tools it references**, hardened by adversarial training against our own MCP-ITP-style
implicit-poisoning optimizer. Evaluated on MCPTox with server-disjoint splits, TPR@1%FPR, hard-negative FPR,
held-out attack families, and a final adaptive attack.

> Status: **Steps 1-2 complete** (infrastructure; data pipeline + frozen split v1). Detector, attacker, baselines next.

## Data (split v1, frozen)

```bash
python scripts/prepare_data.py     # downloads MCPTox @ f85189f if missing, builds examples, verifies split_v1
python scripts/leakage_check.py    # shortcut baselines -> results/leakage_v1.json
```

| | train | val | test |
|---|---|---|---|
| servers | 31 | 4 | 10 |
| poisoned tools | 434 | 73 | 195 (57 from held-out risk families) |
| benign real tools | 257 | 42 | 63 |
| templated hard negatives | 248 | 32 | 80 |

**MCPTox-Para** (meaning-preserving paraphrases, because TF-IDF + LR already reaches TPR@1%FPR = 1.0 on the
original attacks; see `results/leakage_v1.json`):

```bash
python scripts/paraphrase.py --tag para_test                         # GPU (Kaggle T4), ~30-40 min for the test split
python scripts/freeze_para.py --inputs <run_dir>/generations.jsonl   # -> data/para/para_v1.jsonl (frozen)
```

Server-disjoint; risk families *Financial Loss* and *Message Hijacking* never appear in train/val.
The test set is pinned by a SHA-256 hash in `data/splits/split_v1.json`; loaders refuse drifted data.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/default.yaml` | Single source of truth for every experiment setting |
| `mcptp/utils/` | Config loading + CLI overrides, runtime/path detection, seeding, resumable checkpoints, run dirs |
| `mcptp/data/` | MCPTox loading, hard negatives, registry context, server-disjoint frozen splits |
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
