# Results index

Every number in the report comes from a folder below. Each run folder holds the `config.yaml` it ran with (including
the git commit), its `run.log`, and its result files. Raw run outputs on Kaggle go to `outputs/<stage>/<timestamp>_<tag>/`
(git-ignored); the folders here are copies of those runs with the timestamp dropped. `scripts/make_figures.py` rebuilds
every report figure from these files.

**Reading order:** `leakage_v1.json` → `d1b_summary.md` → `attack_r*_summary.md` → `judges_summary.md` →
`ADV_SUMMARY.md` → `llama/SUMMARY.md` → `STEP12_SUMMARY.md` → `STEP13_SUMMARY.md`.

## Summaries (start here)

| File | Step | What it says |
|---|---|---|
| `leakage_v1.json` | 2 | TF-IDF + LR reaches TPR@1%FPR = 1.0 on original MCPTox: the benchmark is shortcut-solvable |
| `d1_pilot_summary.md`, `d1b_summary.md` | 3 | Plain vs context-aware DeBERTa on MCPTox-Para (d1b corrects d1); context gives no gain |
| `attack_r0/r1/r1t/r2_summary.md` | 4 | Black-box attack rounds; r2 adds the judge-artifact finding and the no-poison control |
| `robustness_v1.md`, `robust2_summary.md` | 5 | Early robustness tables (superseded by robust3 / robust5) |
| `robust3_summary.md` | 5 | Robustness on control-verified attacks (classifiers ≥ 0.99 at 1% FPR) |
| `judges_summary.md` | 5a | LLM-judge grid (Qwen3-8B/14B × zero/6-shot) vs classifiers — Fig. 2, 3 |
| `ADV_SUMMARY.md` | 6–11 | White-box attack, adversarial training (2 rounds), calibration, final system, adaptive attack |
| `llama/SUMMARY.md` | 10 | Re-verification of every attack on Llama-3.1-8B |
| `STEP12_SUMMARY.md` | 12 | Contaminated registry; E5 round 2 + final system — Fig. 6, 7 |
| `STEP13_SUMMARY.md` | 13 | Natural-text (fluency-constrained) white-box attack — Fig. 5, 8 |

## Run folders

| Folder | Script | Step | Content |
|---|---|---|---|
| `robust2`, `robust3`, `robust4` | `score_robustness.py` | 5 | Defenders vs attack sets (robust4's 14B judge rows superseded by robust6) |
| `robust5` | `score_robustness.py` | 5 | B0 TF-IDF, E5, DeBERTa, context model, 3 seeds; per-item `scores.jsonl` — Fig. 2, 3 |
| `robust6a`, `robust6b` | `score_robustness.py` | 5a | LLM judges (logit-margin scores); per-item `scores.jsonl` — Fig. 2, 3 |
| `calibration` | `calibrate.py` | 5b | Registry-relative calibration on robust5 scores (val→test FPR gap) |
| `wb1`, `wb1e5` | `attack_whitebox.py` | 6 | White-box attack, 15% word budget, on base DeBERTa / E5 — Fig. 5 |
| `adv1`, `adv1e5` | `adv_train.py` | 7 | Adversarial training round 1 (3 seeds) |
| `wb2_adv`, `wb2_adve5` | `attack_whitebox.py` | 7 | Re-attack of round-1 models, 15% budget |
| `wb3_base_b30_*`, `wb3_adv_b30_*` | `attack_whitebox.py` | 7b | 30% budget on base and round-1 models — Fig. 5 |
| `adv2_deb`, `adv2_e5` | `adv_train.py` | 7c / 12 | Adversarial training round 2 (3 seeds) |
| `wb4_adv2_b30_deb` | `attack_whitebox.py` | 7c | 30% re-attack of round-2 DeBERTa (0 evaders) — Fig. 5 |
| `def1` | `defense_checks.py` | 8 | GPT-2 fluency filter and calibration vs white-box evaders |
| `cal_adv_deb`, `cal_adv_e5` | `calibrate_adv.py` | 9 | Round-1 adversarial training × calibration |
| `cal_adv2_deb`, `cal_adv2_s43`, `cal_adv2_s44` | `calibrate_adv.py` | 9b / 11 | **Final system, DeBERTa**, seeds 42/43/44 — Fig. 6 |
| `cal_adv2_e5_s42/43/44` | `calibrate_adv.py` | 12 | **Final system, E5**, seeds 42/43/44 — Fig. 6 |
| `llama/llama_s0`, `llama/llama_s1` | `verify_second_agent.py` | 10 | Llama-3.1-8B re-verification (two shards) |
| `wb5_final_cal_b30`, `wb5_e5_final_cal_b30` | `attack_whitebox.py` | 11 / 12 | Adaptive attack on the calibrated final system — Fig. 5 |
| `contam_deb`, `contam_e5` | `registry_contam.py` | 12 | Contaminated-registry test, 3 seeds — Fig. 7 |
| `nat_base_*`, `nat_final_cal_*` | `attack_whitebox.py` (`natural=true`) | 13 | Natural-text attack on base and final systems — Fig. 5 |
| `fluency_nat_deb`, `fluency_nat_e5` | `defense_checks.py` (fluency) | 13 | GPT-2 log-perplexity of natural vs unconstrained evaders — Fig. 8 |

## File formats

- `whitebox.jsonl`: one row per attacked description (original, rewrite, margins, words changed, evaded).
- `wb_verified.jsonl` / `data/attacks/verified.jsonl`: agent outcomes with and without the candidate tool;
  `asr_true` counts only successes that do not happen without the tool. A **genuine evasion** = evades the detector
  and `asr_true ≥ 0.2`.
- `scores.jsonl`: `{defender, seed, set, id, score}` for every scored item.
- `summary.json`, `*_results.json`, `cal_adv.json`, `contam.json`: aggregate numbers behind each `.md` table.
