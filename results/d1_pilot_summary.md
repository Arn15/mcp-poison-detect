# D1 pilot (Kaggle run 20261012-212809_d1_pilot, commit 5d80cd5)

DeBERTa-v3-base, max_len 384, lr 2e-5, bs 16, 4 epochs, seeds 42/43/44 (seed-ensemble scores), trained on original MCPTox train split.
Note: val AUROC = 1.000 at every epoch, so best-epoch selection kept epoch 0 (ties -> earliest).

| model | set | AUROC | TPR@1%FPR | TPR@5%FPR | implicit TPR@1% | held-out family TPR@1% | HN-FPR (val thr 5%) |
|---|---|---|---|---|---|---|---|
| B0_tfidf_lr | orig_test | 1.000 [1.00, 1.00] | 1.000 [0.99, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.067 |
| B0_tfidf_lr | para_test | 0.983 [0.97, 0.99] | 0.590 [0.49, 0.76] | 0.923 [0.79, 0.97] | 0.585 | 0.565 | 0.231 |
| B3_plain_deberta | orig_test | 1.000 [1.00, 1.00] | 1.000 [0.99, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.146 |
| B3_plain_deberta | para_test | 0.998 [1.00, 1.00] | 0.976 [0.88, 0.99] | 0.989 [0.98, 1.00] | 0.988 | 0.954 | 0.199 |
| ours_context_SHUFFLED | orig_test | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.573 |
| ours_context_SHUFFLED | para_test | 0.992 [0.99, 1.00] | 0.928 [0.76, 0.96] | 0.947 [0.92, 0.98] | 0.932 | 0.926 | 0.577 |
| ours_context_deberta | orig_test | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 0.360 |
| ours_context_deberta | para_test | 0.997 [0.99, 1.00] | 0.965 [0.92, 0.99] | 0.981 [0.97, 0.99] | 0.978 | 0.917 | 0.429 |

Paired bootstrap (seed-ensemble scores):

- para_test: ours - B3_plain_deberta @ TPR@1%: -0.011 [-0.040, +0.082], P(diff<=0)=0.620
- para_test: ours - B3_plain_deberta @ TPR@5%: -0.008 [-0.024, +0.008], P(diff<=0)=0.902
- para_test: ours - B0_tfidf_lr @ TPR@1%: +0.375 [+0.202, +0.471], P(diff<=0)=0.001
- para_test: ours - B0_tfidf_lr @ TPR@5%: +0.059 [+0.008, +0.197], P(diff<=0)=0.008
- para_test: ours - ours_context_SHUFFLED @ TPR@1%: +0.037 [-0.005, +0.176], P(diff<=0)=0.042
- para_test: ours - ours_context_SHUFFLED @ TPR@5%: +0.035 [+0.003, +0.056], P(diff<=0)=0.017
- orig_test: all differences 0 (every model saturated)
