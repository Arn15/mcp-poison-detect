# Defense checks against white-box evasions

Set sizes: {'val_neg': 138, 'test_neg': 392, 'test_hardneg': 245, 'test_pos': 571, 'blackbox_opt': 379, 'wb3_base_b30_deb_evaders': 252, 'wb3_base_b30_deb_genuine': 81, '20261116-220036_nat_base_deb_evaders': 47, '20261116-220036_nat_base_deb_genuine': 16, '20261116-231047_nat_final_cal_deb_evaders': 2, '20261116-231047_nat_final_cal_deb_genuine': 0}

## A. Fluency filter (gpt2 log-perplexity, threshold on val negatives)

| set | flagged @1% | flagged @5% | median NLL |
|---|---|---|---|
| test_neg | 0.003 | 0.003 | 4.16 |
| test_hardneg | 0.000 | 0.000 | 4.18 |
| test_pos | 0.000 | 0.000 | 3.94 |
| blackbox_opt | 0.000 | 0.000 | 4.07 |
| wb3_base_b30_deb_evaders | 0.000 | 0.000 | 6.13 |
| wb3_base_b30_deb_genuine | 0.000 | 0.000 | 6.08 |
| 20261116-220036_nat_base_deb_evaders | 0.000 | 0.000 | 4.45 |
| 20261116-220036_nat_base_deb_genuine | 0.000 | 0.000 | 4.53 |
| 20261116-231047_nat_final_cal_deb_evaders | 0.000 | 0.000 | 4.36 |

