# Robustness v2: matched FPR, AUROC, 3 seeds (run 20261016-171821_robust2)

Full tables are in results/robust2/robustness.md. Per-example scores are in results/robust2/scores.jsonl.
Trained classifiers (B1/B3/ours) are mean±std over seeds 42/43/44. B0 and B2 are deterministic and run once.

## Headline (detection on optimized rewrites, matched 1% TEST FPR)

| defender | r0_opt | r1_opt | r1t_opt (held-out) | AUROC r0_opt / r1_opt |
|---|---|---|---|---|
| B0 TF-IDF+LR | 0.689 | 0.673 | 0.971 | 0.987 / 0.986 |
| B1 E5+para | 1.000±0.000 | 1.000±0.000 | 0.981±0.027 | 1.000 / 1.000 |
| B3 DeBERTa+para | 0.994±0.005 | 0.995±0.004 | 1.000±0.000 | 1.000 / 1.000 |
| ours context DeBERTa+para | 0.994±0.005 | 0.995±0.004 | 1.000±0.000 | 1.000 / 1.000 |
| B2 LLM judge (Qwen3-8B) | 0.168 | 0.208 | 0.857 | 0.619 / 0.650 |

On the same seeds' original descriptions the judge scores 0.60/0.68 (AUROC 0.90).

## Findings
1. **The judge evasion holds up under every analysis.** At equal FPR the judge's detection falls from 0.60 to 0.17 (r0) and from 0.68 to 0.21 (r1). Its AUROC falls from 0.90 to 0.62–0.65, close to chance, so the drop is not a threshold artifact.
2. **The fine-tuned classifiers hold.** They detect ≥0.98 of rewrites at matched 1% FPR and have AUROC ≈1.00 on every set. Variance across seeds is small.
3. **The context false-alarm result is a calibration problem, not a ranking problem.** With val-fixed thresholds the context model's realized test FPR is 0.23±0.03 (plain DeBERTa: 0.09). At matched FPR the two models are indistinguishable (hard-negative FPR 0.001 vs 0.003). Context information neither helps nor hurts ranking, but it makes val→test threshold transfer worse.
4. **Val-fixed thresholds transfer poorly for every trained model.** The 1% val target gives 7–23% realized test FPR. The deployable numbers need this caveat, or a larger or server-matched calibration set.
5. **TF-IDF is partly evaded** (0.69/0.67 at matched FPR), yet its AUROC stays at 0.99. Optimized rewrites stay separable from benign descriptions; they just sit closer to the boundary.

## Caveats
- The held-out set r1t is small (4 base attacks, 35 rewrites).
- The classifiers are not evaded at all, so adversarial training currently has nothing to fix. A stronger attacker run is needed to test it.
