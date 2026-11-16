# Step 12 — contaminated registry + E5 round 2 (commit 7ed37c0; results/contam_*, adv2_e5, wb5_e5_final_cal_b30, cal_adv2_e5_s4*)

All numbers: final system (round-2 adversarial training + registry-relative calibration), 1% val-FPR target, mean ± std over seeds 42–44.

## A. Naive contamination: K poisoned tools already in every server's registry

| anchor rule | K | DeBERTa FPR / test_pos / bb_opt / wb genuine | E5 FPR / test_pos / bb_opt / wb genuine |
|---|---|---|---|
| max (current) | 0 | 0.042 / 1.00 / 1.00 / 1.00 | 0.037 / 1.00 / 1.00 / 1.00 |
| max (current) | 1 | 0.000 / **0.27** / 0.52 / 0.22 | 0.001 / **0.20** / 0.40 / 0.35 |
| max (current) | 3 | 0.000 / **0.07** / 0.11 / 0.01 | 0.000 / **0.04** / 0.24 / 0.23 |
| approved-only | 0–3 | 0.065 / **1.00** / 1.00 / 1.00 | 0.049 / **1.00** / 1.00 / 1.00 |
| median | 3 | 0.045 / 0.99 / 0.96 / 1.00 | 0.039 / 0.99 / 0.93 / 1.00 |

- One poisoned tool in the registry breaks max-anchor calibration (detection 1.00 -> 0.20-0.27).
- Approved-only anchors fully fix it: 0.0% (DeBERTa) / 0.1% (E5) of planted poisoned tools would pass registration,
  so they never enter the anchor. Cost: clean FPR 4.2% -> 6.5% (DeBERTa), 3.7% -> 4.9% (E5), because flagged benign
  tools are also excluded. Median is a partial fix (bb_opt drops to 0.93-0.96 at K=3).

## B. Adaptive anchor ratchet (k stepping-stone tools that each pass the check)

- No test positive, black-box rewrite or held-out white-box evasion passes after 10 plants, with or without a cap (all 1.00).
- Reason: a plant must pass, so it can raise the anchor by at most tr (0.09-0.16 logits; 1.05 for E5 seed 44), while
  poisoned tools sit ~20+ logits above benign anchors (global threshold ≈ -10, poisoned margins near +16) -> on the
  order of 100+ plants per server. Caveat: the plant pool is real texts; a white-box attacker could tune plants to the
  limit, but the per-step bound tr still holds.
- Anchor cap: not needed; cap = val max costs FPR 4.2% -> 13.3%, cap = val max + 2 is free (4.3%).

## E5 round 2 (adv_train, 3 seeds, 1,400 rewrites) — global threshold
- Held-out white-box sets: 0.41-0.94 -> **1.00** (all four); genuine held-out 0.65 -> 1.00. Detection elsewhere 1.00.
- Cost: test FPR 4.8% -> 10.2%, hard-neg 2.0% -> 4.5% (global threshold).

## E5 final system (calibrate_adv, 3 seeds)
| | base global | base rel | adv2 global | **adv2 + rel** |
|---|---|---|---|---|
| FPR unseen servers (n=392) | 0.074 ± 0.022 | 0.034 ± 0.012 | 0.074 ± 0.014 | **0.030 ± 0.005** |
| FPR hard negatives (n=245) | 0.035 ± 0.007 | 0.018 ± 0.005 | 0.030 ± 0.020 | **0.004 ± 0.003** |
| test_pos + all black-box rounds | 0.999-1.00 | 0.999-1.00 | 0.999-1.00 | **0.999-1.00** |
| wb30 genuine (n=82) / held-out (n=18) | 0.65 / 0.52 | 0.35 / 0.28 | 1.00 / 1.00 | **1.00 / 1.00** |
| wb15 held-out genuine (n=38) | 0.76 | 0.61 | 1.00 | **1.00** |

## Adaptive white-box attack on the calibrated E5 final system (30% budget, 100 steps)
- 9 / 379 evade (13 words changed on average), **0 genuine** (none still hijacks the agent). Same as DeBERTa (2 evade, 0 genuine).

## Takeaways for the report
1. Both backbones now have the full pipeline: round-2 adversarial training + calibration -> ~3% FPR, 0.4-1.4% hard-neg,
   detection 1.00, 0 genuine evasions under an adaptive attack on the calibrated threshold.
2. Calibration's trust assumption is real (one poisoned registry tool breaks it); anchoring on approved tools only
   removes the failure at a ~1-2 point FPR cost. Recommend approved-only anchors as the deployed rule.
3. The incremental "ratchet" attack is impractical because each accepted tool moves the anchor by at most tr.
