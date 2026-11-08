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

## Stronger adaptive attack: 30% word budget, 100 steps (Step 7b, commit a81788c)

Same protocol; the ADV detector is re-created with seed 42 and attacked with twice the word budget the training rewrites used.
BASE is a normally trained twin attacked with the same budget.

| detector | evaders BASE -> ADV | genuine evasions BASE -> ADV | seeds with a genuine evasion BASE -> ADV |
|---|---|---|---|
| DeBERTa-v3 + para | 252 -> 30 | 81 -> **9** | 11 -> 3 |
| E5 + para | 311 -> 10 | 82 -> **0** | 12 -> 0 |

- At 30% the attack is much stronger on undefended models (genuine 48 -> 81 DeBERTa, 40 -> 82 E5; ~7.5 words changed).
- Adversarial training still removes ~89% (DeBERTa) and 100% (E5) of genuine evasions, although it only saw 15%-budget rewrites.
- DeBERTa's 9 residual genuine evasions come from 3 seeds, 7 of them held-out test seeds (r2t), and all rest on 1-2 agent queries; they need ~10 words changed.
- Takeaway: one round of adversarial training generalizes beyond its attack budget, but not fully for DeBERTa; a second round on 30%-budget rewrites is the natural next step.

## Final system: adversarial training + registry-relative calibration (Step 9, results/cal_adv_*; seed 42 twins, 1% val-FPR target)

| | DeBERTa base global | DeBERTa **adv + rel** | E5 base global | E5 **adv + rel** |
|---|---|---|---|---|
| FPR unseen-server benign (n=392) | 0.125 | **0.031** | 0.102 | **0.031** |
| FPR hard negatives (n=245) | 0.057 | **0.000** | 0.053 | **0.020** |
| test positives + all black-box rounds | 1.00 | **1.00** | 1.00 | **1.00** |
| white-box 15%, held-out seeds (genuine, n=20) | 1.00 | **1.00** | 1.00 | **1.00** |
| white-box 30% vs base (genuine, n=81 / 82) | 0.57 | **1.00** | 0.79 | **1.00** |
| white-box 30% vs ADV itself (genuine, n=9 / 0) | 0.00 | **0.00** (adv global 0.67) | — | — |

- The combination keeps every gain: false alarms ~3% (from 10-13%), hard negatives 0-2%, detection 1.00 on all non-adaptive and transferred white-box sets.
- Caveat (adaptive): the 9 DeBERTa evasions found against the ADV model at 30% budget still pass after calibration (the global threshold would have caught 6 of 9, at 12.8% FPR). Calibration trades some adaptive robustness for a 4x lower false-alarm rate; a second adversarial round at 30% is the fix.

## Second adversarial round, DeBERTa (Step 7c, commit 83de931; results/adv2_deb, results/wb4_adv2_b30_deb)

Round 2 adds the 30%-budget white-box rewrites of train seeds (from attacks on BASE and on the round-1 ADV model): 1,400 adversarial rewrites in total. Held-out test seeds never trained on.

| | base | round 1 | **round 2** |
|---|---|---|---|
| 30%-budget re-attack: evaders | 252 | 30 | **0** |
| 30%-budget re-attack: genuine evasions | 81 | 9 | **0** |
| held-out white-box rewrites detected (all 4 held-out sets) | 0.22-0.99 | 1.00 (15% sets) | **1.00** |
| test FPR @1% val, global threshold (3 seeds) | 0.107 | ~0.14 | **0.162** |
| hard-neg FPR, global threshold | 0.041 | ~0.07 | **0.099** |

- Two rounds remove all white-box evasions up to a 30% word budget; detection on test positives and all black-box rounds stays 1.00.
- Cost: global-threshold false alarms rise further (+5.5 points on unseen servers, +5.8 on hard negatives). Registry-relative calibration brought round 1 to 3.1%; round 2 + calibration has not been measured yet (one calibrate_adv run with the round-2 sources).

## Round 2 + calibration (Step 9b, results/cal_adv2_deb; seed 42, 1% val-FPR target) — FINAL SYSTEM (DeBERTa)

| | base global | base rel | round-2 adv global | **round-2 adv + rel** |
|---|---|---|---|---|
| FPR unseen-server benign (n=392) | 0.102 | 0.031 | 0.133 | **0.031** |
| FPR hard negatives (n=245) | 0.037 | 0.008 | 0.086 | **0.008** |
| test positives + all black-box rounds | 1.00 | 1.00 | 1.00 | **1.00** |
| held-out white-box genuine, 15% (n=43) | 0.67 | 0.47 | 1.00 | **1.00** |
| held-out white-box genuine, 30% (n=16) | 0.44 | 0.00 | 1.00 | **1.00** |

- Calibration removes round 2's false-alarm cost entirely (13.3% -> 3.1%, hard negatives 8.6% -> 0.8%) with no detection loss on any set, including held-out white-box evasions.
- Adaptive re-attack at 30% against the round-2 model (global threshold) found 0 evaders (results/wb4_adv2_b30_deb). An adaptive attack aimed at the calibrated score was not run; that and E5 round 2 are the remaining gaps.

## Adaptive attack on the final system + seed repeats (Step 11, commit 8743d9e; results/wb5_final_cal_b30, results/cal_adv2_s43, results/cal_adv2_s44)

White-box attack aimed at the CALIBRATED per-server threshold (rel threshold + server anchor) of round-2 DeBERTa, 30% budget, 100 steps:
**2 / 379 evade, 0 genuine evasions** (both evaders lose the attack effect on the agent). 37 / 379 fall below margin 0.

Final system over 3 seeds (42, 43, 44), 1% val-FPR target:

| | base, global threshold | **round-2 adv + calibration** |
|---|---|---|
| FPR unseen-server benign (n=392) | 0.104 ± 0.006 | **0.035 ± 0.004** |
| FPR hard negatives (n=245) | 0.039 ± 0.004 | **0.014 ± 0.005** |
| test positives (n=571) | 1.000 ± 0.000 | **1.000 ± 0.000** |
| held-out white-box genuine, 15% (n=43) | 0.558 ± 0.164 | **1.000 ± 0.000** |
| held-out white-box genuine, 30% (n=16) | 0.292 ± 0.206 | **1.000 ± 0.000** |

Black-box rounds: 1.00 for every seed and variant.
