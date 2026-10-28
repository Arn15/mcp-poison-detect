Trained classifiers: mean±std over seeds [42, 43, 44]; B0 and B2 are deterministic (single run).

### Val-fixed threshold @ 1% val FPR (deployable; realized FPR in *_neg columns)

| defender | test_pos | test_neg | test_hardneg | r0_orig | r0_opt | r1_orig | r1_opt | r1b_orig | r1b_opt | r1t_orig | r1t_opt | r2_orig | r2_opt | r2t_orig | r2t_opt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0_tfidf_lr | 0.835 | 0.018 | 0.020 | 1.000 | 0.978 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.928 | 1.000 | 0.857 |
| B2_judge_14b_fs | 0.718 | 0.013 | 0.016 | 0.565 | 0.391 | 0.571 | 0.355 | 0.500 | 0.319 | 0.500 | 0.750 | 0.636 | 0.179 | 1.000 | 0.619 |

### Matched threshold @ 1% TEST FPR (oracle; equal FPR for every defender)

| defender | test_pos | test_hardneg | r0_orig | r0_opt | r1_orig | r1_opt | r1b_orig | r1b_opt | r1t_orig | r1t_opt | r2_orig | r2_opt | r2t_orig | r2t_opt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0_tfidf_lr | 0.772 | 0.008 | 1.000 | 0.946 | 1.000 | 0.981 | 1.000 | 0.972 | 1.000 | 1.000 | 1.000 | 0.897 | 1.000 | 0.714 |
| B2_judge_14b_fs | 0.704 | 0.016 | 0.565 | 0.380 | 0.571 | 0.336 | 0.500 | 0.306 | 0.500 | 0.750 | 0.636 | 0.149 | 1.000 | 0.524 |

### Val-fixed threshold @ 5% val FPR (deployable; realized FPR in *_neg columns)

| defender | test_pos | test_neg | test_hardneg | r0_orig | r0_opt | r1_orig | r1_opt | r1b_orig | r1b_opt | r1t_orig | r1t_opt | r2_orig | r2_opt | r2t_orig | r2t_opt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0_tfidf_lr | 0.982 | 0.082 | 0.053 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.985 | 1.000 | 1.000 |
| B2_judge_14b_fs | 0.839 | 0.028 | 0.045 | 0.696 | 0.554 | 0.714 | 0.486 | 0.667 | 0.458 | 1.000 | 1.000 | 0.773 | 0.323 | 1.000 | 0.857 |

### Matched threshold @ 5% TEST FPR (oracle; equal FPR for every defender)

| defender | test_pos | test_hardneg | r0_orig | r0_opt | r1_orig | r1_opt | r1b_orig | r1b_opt | r1t_orig | r1t_opt | r2_orig | r2_opt | r2t_orig | r2t_opt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0_tfidf_lr | 0.956 | 0.033 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.974 | 1.000 | 1.000 |
| B2_judge_14b_fs | 0.928 | 0.073 | 0.913 | 0.761 | 0.952 | 0.776 | 0.944 | 0.722 | 1.000 | 1.000 | 1.000 | 0.610 | 1.000 | 1.000 |

### AUROC vs test negatives (threshold-free)

| defender | test_pos | test_pos_vs_hardneg | r0_orig | r0_opt | r1_orig | r1_opt | r1b_orig | r1b_opt | r1t_orig | r1t_opt | r2_orig | r2_opt | r2t_orig | r2t_opt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0_tfidf_lr | 0.991 | 0.993 | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 0.995 | 1.000 | 0.993 |
| B2_judge_14b_fs | 0.986 | 0.978 | 0.982 | 0.965 | 0.983 | 0.964 | 0.981 | 0.963 | 0.993 | 0.996 | 0.988 | 0.939 | 0.996 | 0.987 |

n per set: {"val_neg": 138, "test_neg": 392, "test_hardneg": 245, "test_pos": 571, "registry": 362, "r0_opt": 92, "r0_orig": 23, "r1_opt": 107, "r1_orig": 21, "r1b_opt": 72, "r1b_orig": 18, "r1t_opt": 8, "r1t_orig": 2, "r2_opt": 195, "r2_orig": 22, "r2t_opt": 21, "r2t_orig": 3}
