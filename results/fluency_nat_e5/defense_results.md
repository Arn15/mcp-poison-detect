# Defense checks against white-box evasions

Set sizes: {'val_neg': 138, 'test_neg': 392, 'test_hardneg': 245, 'test_pos': 571, 'blackbox_opt': 379, 'wb3_base_b30_e5_evaders': 311, 'wb3_base_b30_e5_genuine': 82, '20261116-220003_nat_base_e5_evaders': 36, '20261116-220003_nat_base_e5_genuine': 6, '20261116-230207_nat_final_cal_e5_evaders': 7, '20261116-230207_nat_final_cal_e5_genuine': 0}

## A. Fluency filter (gpt2 log-perplexity, threshold on val negatives)

| set | flagged @1% | flagged @5% | median NLL |
|---|---|---|---|
| test_neg | 0.003 | 0.003 | 4.16 |
| test_hardneg | 0.000 | 0.000 | 4.18 |
| test_pos | 0.000 | 0.000 | 3.94 |
| blackbox_opt | 0.000 | 0.000 | 4.07 |
| wb3_base_b30_e5_evaders | 0.000 | 0.000 | 6.43 |
| wb3_base_b30_e5_genuine | 0.000 | 0.000 | 6.43 |
| 20261116-220003_nat_base_e5_evaders | 0.000 | 0.000 | 4.36 |
| 20261116-220003_nat_base_e5_genuine | 0.000 | 0.000 | 4.50 |
| 20261116-230207_nat_final_cal_e5_evaders | 0.000 | 0.000 | 4.12 |

