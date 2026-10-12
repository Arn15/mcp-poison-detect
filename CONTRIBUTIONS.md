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
| `mcptp/data/mcptox.py` | Own | Parse MCPTox raw JSON into benign / poison tables; injection-marker stripping; seed ASR | MCPTox data format |
| `mcptp/data/context.py` | Own | Registry-context retrieval (name variants + optional E5 semantic match); detector input pairs | — |
| `mcptp/data/hard_negatives.py` | Own | Benign hard negatives (precondition-on-self, cross-tool helper, mention) with destructive-tool filter | — |
| `mcptp/data/build.py` | Own | Assemble labeled examples with context | — |
| `mcptp/data/splits.py` | Own | Server-disjoint split search, held-out risk families, SHA-256 frozen manifest + verification | — |
| `mcptp/data/io.py` | Own | Load examples with the frozen split enforced | — |
| `mcptp/eval/metrics.py` | Own | TPR@FPR, val-fixed thresholds, AUROC, stratified bootstrap CIs | — |
| `scripts/prepare_data.py` | Own | Step 2 pipeline: build, freeze/verify split, stats | — |
| `scripts/leakage_check.py` | Own | TF-IDF/LR and rule baselines for shortcut detection | MCP-Poison-Bench concern |
| `mcptp/data/paraphrase.py` | Own | MCPTox-Para prompts, faithfulness checks (names, values, length, refusal), HF / echo backends | — |
| `scripts/paraphrase.py` | Own | Resumable batched paraphrase generation (GPU) | — |
| `scripts/freeze_para.py` | Own | Validate generations, rebuild context, freeze MCPTox-Para with per-split hashes | — |
| `tests/test_data.py` | Own | Unit tests for data, splits, metrics | — |
| `mcptp/detector/model.py` | Own | Plain / context-aware cross-encoder inputs, shuffled-context control | — |
| `mcptp/detector/train.py` | Own | AMP fine-tuning loop, context dropout + context swap augmentation, val selection with loss tie-break | — |
| `mcptp/eval/report.py` | Own | Subset evaluation (implicit, held-out family/server), val-fixed thresholds, hard-negative FPR | — |
| `scripts/train_detector.py` | Own | Variant sweep (B0, B3, context, +para), seeds, paired bootstrap, summary table | — |
| *(Step 4)* `mcptp/attacker/…` | Adapted | MCP-ITP-style optimizer | Li et al., arXiv 2601.07395 |
| *(Step 5)* `mcptp/baselines/e5.py` | Adapted | MCP-Guard Stage II classifier | Xing et al., arXiv 2508.10991 |
| — | Library | DeBERTa-v3 / E5 weights, HF Transformers, vLLM, scikit-learn | — |
