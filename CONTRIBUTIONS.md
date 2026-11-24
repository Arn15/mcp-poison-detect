# Code authorship

The course requires stating which code is my own. **All code in this repository was developed by me for this
project, with Claude (Anthropic) used as an AI coding assistant**; commits it helped write carry a
`Co-Authored-By: Claude` line. The research questions, threat model, experiment design, method choices (including
registry-relative calibration, the no-poison control and approved-only anchors) and the analysis of results are
my own.
Where a module re-implements a published method, it is marked *Adapted* and the paper is cited; no third-party
source code was used. Pretrained models and standard libraries are used through their packages (*Library*).

Legend: **Own** = my design, written for this project · **Adapted** = written for this project, following the cited paper's method · **Library** = a
third-party package or pretrained model, called but not re-implemented.

## Package `mcptp/`

| File | Status | Function | Reference |
|---|---|---|---|
| `mcptp/utils/config.py` | Own | YAML config with `_base` inheritance and dotted `--set KEY=VALUE` overrides | — |
| `mcptp/utils/env.py` | Own | Runtime detection (local / Kaggle / Colab), data and output paths | — |
| `mcptp/utils/seed.py` | Own | Seeding of python / numpy / torch | — |
| `mcptp/utils/checkpoint.py` | Own | Resumable append-only JSONL store (`JsonlCheckpoint`), atomic JSON writes | — |
| `mcptp/utils/logging.py` | Own | Logger, timestamped run directories, git commit capture | — |
| `mcptp/data/mcptox.py` | Own | Parse MCPTox raw JSON into benign / poisoned tool tables; injection-marker stripping | MCPTox data format |
| `mcptp/data/hard_negatives.py` | Own | Benign hard negatives that legitimately say "call X first" | — |
| `mcptp/data/context.py` | Own | Registry context: which registered tools a description refers to | — |
| `mcptp/data/build.py` | Own | Assemble labeled examples with context | — |
| `mcptp/data/splits.py` | Own | Server-disjoint split search, held-out risk families, SHA-256 frozen manifest | — |
| `mcptp/data/io.py` | Own | Load examples / MCPTox-Para with the frozen split enforced | — |
| `mcptp/data/paraphrase.py` | Own | MCPTox-Para prompts, faithfulness checks, generation backends | — |
| `mcptp/detector/model.py` | Own | Detector inputs (plain / with registry context), model construction | — |
| `mcptp/detector/train.py` | Own | AMP fine-tuning loop, context augmentation, validation selection, scoring, save / load | — |
| `mcptp/attacker/agent.py` | Own | MCPTox-protocol agent prompt, tool-call parser, rule-based success judge | MCPTox evaluation protocol |
| `mcptp/attacker/evaluate.py` | Own | Attack success rate of a candidate description on the target agent | — |
| `mcptp/attacker/optimize.py` | Adapted | Black-box implicit-poisoning optimizer (attacker LLM, feedback, evasion-aware selection) | Li et al., arXiv 2601.07395 (MCP-ITP) |
| `mcptp/attacker/whitebox.py` | Adapted | White-box word-substitution attack: gradient top-k (HotFlip) + exact scoring of sampled swaps (GCG), payload protection, word budget; natural mode with masked-LM candidates and a GPT-2 perplexity budget | Ebrahimi et al., 2018; Zou et al., 2023; Li et al., 2020 (BERT-Attack); Jin et al., 2020 (TextFooler) |
| `mcptp/baselines/llm_judge.py` | Adapted | LLM-as-judge defender (Qwen3-8B / 14B, zero / few-shot), Yes/No logit-margin score | AI-Infra-Guard audit-prompt style |
| `mcptp/eval/metrics.py` | Own | TPR@FPR, validation-fixed thresholds, AUROC, stratified bootstrap CIs | — |
| `mcptp/eval/report.py` | Own | Metrics on standard test subsets and hard negatives | — |
| `mcptp/eval/calibration.py` | **Own (method)** | Registry-relative threshold calibration: score minus the max score of the server's registered tools | — |

The E5 baseline (B1, MCP-Guard Stage II style) is the same detector code with `detector.backbone =
intfloat/multilingual-e5-base` (Adapted from Xing et al., arXiv 2508.10991; no separate module).

## Scripts `scripts/` (one entry point per experiment)

| File | Status | Step | Function |
|---|---|---|---|
| `common.py` | Own | — | Shared CLI flags and run setup for every script |
| `check_env.py`, `smoke_test.py` | Own | 1 | Environment report; resumable-loop smoke test |
| `prepare_data.py` | Own | 2 | Build examples, freeze / verify the split |
| `leakage_check.py` | Own | 2 | Shortcut baselines (TF-IDF + LR, keyword rules) on MCPTox |
| `paraphrase.py`, `freeze_para.py` | Own | 2b | Generate and freeze MCPTox-Para |
| `train_detector.py` | Own | 3 | Train / evaluate B0, plain and context-aware detectors |
| `verify_agent.py` | Own | 4 | Check the agent harness against MCPTox's recorded outcomes |
| `run_attack.py` | Own | 4 | Black-box attacker loop (resumable) |
| `score_attacks.py` | Own | 4 | Score an attack set against every defender |
| `verify_attacks.py` | **Own (method)** | 4 | No-poison control: agent with vs without the candidate tool |
| `score_robustness.py` | Own | 5 | Robustness comparison of all defenders incl. LLM judges |
| `calibrate.py` | Own | 5b | Registry-relative calibration study on stored scores |
| `attack_whitebox.py` | Own | 6, 11, 13 | White-box attack orchestration (global / calibrated threshold, natural mode) + agent control |
| `adv_train.py` | Adapted | 7 | Adversarial training on white-box rewrites of train seeds (Madry et al., 2018) |
| `defense_checks.py` | Own | 8 | GPT-2 fluency filter and calibration vs white-box evaders |
| `calibrate_adv.py` | Own | 9 | Final system: adversarial training × calibration |
| `verify_second_agent.py` | Own | 10 | Re-verify attacks on Llama-3.1-8B |
| `registry_contam.py` | **Own (method)** | 12 | Contaminated-registry test; approved-only / median anchors; anchor-ratchet attacker |
| `make_figures.py` | Own | — | All report figures from committed results |

## Other

| Path | Status | Function |
|---|---|---|
| `tests/*.py` | Own | CPU unit tests (data, splits, metrics, infra, judge, white-box constraints, calibration, contamination) |
| `notebooks/kaggle_runner.ipynb`, `colab_runner.ipynb` | Own | Generic GPU runner: clone repo, install, run one script |
| `notebooks/experiments/*.ipynb` | Own | The exact Kaggle runs behind each result (see `README.md`) |
| `configs/default.yaml` | Own | Every experiment setting |
| — | Library | DeBERTa-v3, multilingual-E5, Qwen3, Llama-3.1, distilroberta, GPT-2 weights; PyTorch, HF Transformers, bitsandbytes, scikit-learn, matplotlib |
| — | Data | MCPTox (Wang et al., AAAI 2026), not redistributed |
