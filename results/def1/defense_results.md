# Defense checks against white-box evasions

Set sizes: {'val_neg': 138, 'test_neg': 392, 'test_hardneg': 245, 'test_pos': 571, 'blackbox_opt': 379, 'wb1_evaders': 95, 'wb1_genuine': 48, 'wb1e5_evaders': 110, 'wb1e5_genuine': 40}

## A. Fluency filter (gpt2 log-perplexity, threshold on val negatives)

| set | flagged @1% | flagged @5% | median NLL |
|---|---|---|---|
| test_neg | 0.003 | 0.003 | 4.16 |
| test_hardneg | 0.000 | 0.000 | 4.18 |
| test_pos | 0.000 | 0.000 | 3.94 |
| blackbox_opt | 0.000 | 0.000 | 4.07 |
| wb1_evaders | 0.000 | 0.000 | 5.37 |
| wb1_genuine | 0.000 | 0.000 | 5.37 |
| wb1e5_evaders | 0.000 | 0.000 | 5.68 |
| wb1e5_genuine | 0.000 | 0.000 | 5.75 |

## B/C. Calibration on results/wb1 (microsoft/deberta-v3-base, twin detector)

| set @1% | global | rel_max | rel_max OR fluency | coverage |
|---|---|---|---|---|
| test_neg | 0.112 | 0.031 | 0.036 | 386/392 |
| test_hardneg | 0.049 | 0.012 | 0.012 | 245/245 |
| test_pos | 1.000 | 1.000 | 1.000 | 571/571 |
| blackbox_opt | 1.000 | 1.000 |  | 379/379 |
| evaders | 0.663 | 0.305 | 0.305 | 95/95 |
| genuine | 0.729 | 0.333 | 0.333 | 48/48 |

## B/C. Calibration on results/wb1e5 (intfloat/multilingual-e5-base, twin detector)

| set @1% | global | rel_max | rel_max OR fluency | coverage |
|---|---|---|---|---|
| test_neg | 0.041 | 0.041 | 0.043 | 386/392 |
| test_hardneg | 0.016 | 0.024 | 0.024 | 245/245 |
| test_pos | 1.000 | 1.000 | 1.000 | 571/571 |
| blackbox_opt | 1.000 | 1.000 |  | 379/379 |
| evaders | 0.373 | 0.100 | 0.100 | 110/110 |
| genuine | 0.375 | 0.050 | 0.050 | 40/40 |


## Reading (written after the run)
- **Fluency filter fails at deployable FPR.** White-box evaders have clearly higher GPT-2 log-perplexity (median 5.4-5.8 vs 4.2 for benign tools), but the 1%-FPR threshold on val negatives sits above every evader: short, jargon-heavy benign descriptions occupy the high-perplexity tail. Flags 0 evaders at both 1% and 5%. "Looks unnatural to a human" does not mean "separable by perplexity at low FPR".
- **White-box evasions are model-specific.** A re-trained twin (same backbone, data, seed; GPU nondeterminism only) already catches 73% (DeBERTa) / 38% (E5) of the genuine evasions at the global threshold. Suggests retraining / ensembling as a cheap partial defense; also means the calibration numbers here are on the twin, not the attacked model.
- **Registry-relative calibration is not a defense against white-box evasion.** On the twin it lowers false alarms (DeBERTa 11.2% -> 3.1%) but also lowers detection of the evaders (genuine 0.73 -> 0.33; E5 0.38 -> 0.05). Calibration fixes server shift; adversarial training (results/ADV_SUMMARY.md) is what removes the evasions.
