# Adversarial training vs the white-box attacker (Step 7)

Adversarial data: 700 white-box rewrites (wb1 + wb1e5) of TRAIN-split seeds, label 1. Held-out test seeds never trained on.
Re-attack: same white-box attack (15% word budget, 1%-val-FPR threshold) against the seed-42 ADV checkpoint.

| detector | white-box evaders (base -> adv) | genuine evasions (base -> adv) | test FPR @1% val (base -> adv, 3 seeds) | hard-neg FPR @1% (base -> adv) | held-out wb rewrites detected @1% (base -> adv) |
|---|---|---|---|---|---|
| DeBERTa-v3 + para | 95 -> **0** | 48 -> **0** | 0.117 -> 0.139 | 0.050 -> 0.068 | 0.77 -> 1.00 |
| E5 + para | 110 -> **1** | 40 -> **0** | 0.068 -> 0.060 | 0.033 -> 0.020 | 0.93 -> 1.00 |

- Detection on test positives and every black-box round stays 1.00 (E5 0.999) for both.
- E5's single remaining evader lost the attack effect on the agent (asr_true 0), so 0 genuine evasions.
- Below margin 0 (p < 0.5) after attack: DeBERTa 184 -> 11, E5 192 -> 8.
- Cost: DeBERTa's global-threshold FPR rises ~2 points (within ~1 std); E5's does not. Registry-relative calibration (results/calibration) addresses the global-threshold FPR gap separately.
- Caveats: one adversarial round; the re-attack uses the same budget as the attack that generated the training data (a larger budget, e.g. max_frac=0.30, is the obvious next adaptive test); held-out white-box set is small (29).
- Base FPR numbers differ from robust5 (0.106 / 0.071) because these are separate re-trainings; compare base vs adv within this table.
