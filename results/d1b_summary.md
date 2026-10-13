# D1b: corrected pilot (Kaggle run 20261013-005206_d1b, commit 6b70f94)

DeBERTa-v3-base, max_len 384, lr 2e-5, bs 16, 4 epochs, seeds 42/43/44 (seed-ensemble scores).
Val = original val + MCPTox-Para val (281). Val still saturated (AUROC 1.000 every epoch); the val-loss tie-break picked epochs 1-3.
Context model: context_dropout 0.3, context_swap 0.5. "+para" = training also on 862 train-split paraphrases.

| model | set | AUROC | TPR@1%FPR | TPR@5%FPR | implicit TPR@1% | held-out family TPR@1% | test FPR @ val thr 5% | HN-FPR @ val thr 5% | seed std TPR@1% |
|---|---|---|---|---|---|---|---|---|---|
| B0_tfidf_lr | orig_test | 1.000 [1.00, 1.00] | 1.000 [0.99, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.056 | 0.022 | 0.000 |
| B0_tfidf_lr | para_test | 0.983 [0.97, 0.99] | 0.590 [0.49, 0.76] | 0.923 [0.79, 0.97] | 0.585 | 0.565 | 0.096 | 0.071 | 0.000 |
| B3_plain | orig_test | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.196 | 0.124 | 0.000 |
| B3_plain | para_test | 0.999 [1.00, 1.00] | 0.981 [0.94, 1.00] | 0.997 [0.99, 1.00] | 0.994 | 0.954 | 0.193 | 0.186 | 0.008 |
| B3_plain+para | orig_test | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.161 | 0.079 | 0.000 |
| B3_plain+para | para_test | 1.000 [1.00, 1.00] | 1.000 [0.99, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.108 | 0.038 | 0.003 |
| ours_context | orig_test | 1.000 [1.00, 1.00] | 1.000 [0.97, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.266 | 0.270 | 0.000 |
| ours_context | para_test | 0.996 [0.99, 1.00] | 0.934 [0.90, 0.97] | 0.973 [0.95, 0.99] | 0.948 | 0.880 | 0.257 | 0.288 | 0.066 |
| ours_context+para | orig_test | 1.000 [1.00, 1.00] | 1.000 [0.97, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.364 | 0.371 | 0.000 |
| ours_context+para | para_test | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.349 | 0.404 | 0.004 |
| ours_context+para/SHUFFLED | orig_test | 1.000 [1.00, 1.00] | 1.000 [0.99, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.741 | 0.663 | 0.000 |
| ours_context+para/SHUFFLED | para_test | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.767 | 0.718 | 0.001 |
| ours_context/SHUFFLED | orig_test | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.538 | 0.461 | 0.002 |
| ours_context/SHUFFLED | para_test | 0.994 [0.99, 1.00] | 0.965 [0.92, 0.98] | 0.973 [0.95, 0.99] | 0.975 | 0.954 | 0.558 | 0.494 | 0.018 |

Key paired bootstrap results (seed-ensemble, a - b, 95% CI, P(diff<=0)):
- para_test: ours_context - B3_plain @1%: -0.048 [-0.072, -0.011], P=1.000 (context hurts without aug)
- para_test: ours_context+para - B3_plain+para @1%: +0.000 [+0.000, +0.005] (tie at ceiling)
- para_test: B3_plain+para - B3_plain @1%: +0.019 [+0.003, +0.053], P=0.009 (paraphrase aug helps)
- para_test: ours_context+para - ours_context @1%: +0.066 [+0.027, +0.098], P=0.000
- para_test: ours_context - B0 @1%: +0.343 [+0.175, +0.447], P=0.001
- orig_test: every model saturated (all diffs 0)
