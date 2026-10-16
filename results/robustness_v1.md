# Robustness comparison v1 (Kaggle 20261016-143402_robust, commit 768e40f)

All trained defenders trained in-run (seed 42) on original + paraphrase train split. B2 = Qwen3-8B (4-bit) judge,
score = P(Yes). Thresholds fixed on validation negatives (orig + para val, n=138). *_neg / hardneg = FPR; others = detection.
Attack sets are de-duplicated by text: r0 = detector-blind (train seeds, 30 base attacks), r1 = adaptive vs plain
DeBERTa (train seeds, 28 bases), r1t = adaptive on held-out TEST seeds (4 bases). *_orig = same bases' MCPTox text.

### @ 1% val FPR
| defender | test_pos | test_neg | test_hardneg | r0_orig | r0_opt | r1_orig | r1_opt | r1t_orig | r1t_opt |
|---|---|---|---|---|---|---|---|---|---|
| B0_tfidf_lr | 0.835 | 0.018 | 0.020 | 1.000 | 0.720 | 1.000 | 0.693 | 1.000 | 0.971 |
| B1_e5+para | 1.000 | 0.069 | 0.045 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B3_plain_deberta+para | 1.000 | 0.122 | 0.053 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ours_context_deberta+para | 1.000 | 0.278 | 0.286 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B2_llm_judge | 0.651 | 0.000 | 0.000 | 0.600 | 0.106 | 0.679 | 0.153 | 1.000 | 0.800 |

### @ 5% val FPR
| defender | test_pos | test_neg | test_hardneg | r0_orig | r0_opt | r1_orig | r1_opt | r1t_orig | r1t_opt |
|---|---|---|---|---|---|---|---|---|---|
| B0_tfidf_lr | 0.982 | 0.082 | 0.053 | 1.000 | 0.957 | 1.000 | 0.970 | 1.000 | 1.000 |
| B1_e5+para | 1.000 | 0.094 | 0.045 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B3_plain_deberta+para | 1.000 | 0.143 | 0.069 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ours_context_deberta+para | 1.000 | 0.327 | 0.363 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B2_llm_judge | 0.905 | 0.087 | 0.106 | 0.833 | 0.317 | 0.821 | 0.371 | 1.000 | 1.000 |

n per set: val_neg 138, test_neg 392, test_hardneg 245, test_pos 571, r0_opt 161, r0_orig 30, r1_opt 202, r1_orig 28, r1t_opt 35, r1t_orig 4

## Findings
1. LLM judge is evaded: 0.60/0.68 on original attacks -> 0.11/0.15 on optimized rewrites (1% val FPR). Even the
   detector-blind round (r0) suffices: natural-language rewriting transfers to the judge.
2. Fine-tuned classifiers with paraphrase training (E5, DeBERTa) detect 100% of optimized rewrites, incl. held-out.
3. TF-IDF degrades (~0.70) -> paraphrase-robust semantics matter.
4. Registry context: no detection gain, 2-5x the false alarms (test FPR 0.28, hard-neg 0.29) -> negative result.
## Caveats
- Val-fixed thresholds transfer poorly across servers: realized test FPR differs per defender (judge 0.00 vs DeBERTa
  0.12). TODO: report detection at matched TEST FPR (needs per-example scores; add to score_robustness).
- r1t held-out evidence = 4 base attacks; single training seed (42); 4-bit agent; 4B attacker.
