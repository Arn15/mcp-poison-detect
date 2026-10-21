# Robustness v3: control-verified attacks (run 20261021-162357_robust3, commit 0c33bc1)

This is the decisive table and supersedes robust2. Attack sets contain only rewrites that pass the no-poison
control (scripts/verify_attacks.py). Trained classifiers are mean±std over seeds 42/43/44. Full tables are in
results/robust3/robustness.md, and per-example scores in results/robust3/scores.jsonl.

## Detection at matched 1% TEST FPR (opt = optimized rewrites; orig = the same bases' original descriptions)

| defender | r0 opt | r1 opt | r1b opt | r2 opt (14B attacker) | r1t opt (held-out) | r2t opt (held-out) |
|---|---|---|---|---|---|---|
| n | 92 | 107 | 72 | 195 | 8 | 21 |
| B0 TF-IDF+LR | 0.946 | 0.981 | 0.972 | 0.897 | 1.000 | 0.714 |
| B1 E5+para | 0.996±0.005 | 1.000 | 1.000 | 0.991±0.005 | 0.917±0.118 | 1.000 |
| B3 DeBERTa+para | 1.000 | 1.000 | 1.000 | 0.998±0.002 | 1.000 | 1.000 |
| ours context DeBERTa+para | 1.000 | 1.000 | 1.000 | 0.991±0.006 | 1.000 | 1.000 |
| B2 LLM judge (Qwen3-8B) | 0.120 (orig 0.478) | 0.243 (0.571) | 0.208 (0.556) | 0.205 (0.455) | 0.625 (1.000) | 0.952 (0.667) |

The judge's AUROC on optimized rewrites vs test negatives is 0.524 (r0), 0.554 (r1), 0.514 (r1b) and 0.463 (r2),
which is chance level. On the originals it is 0.85–0.88. Every classifier has AUROC ≥0.997 on every set.

## Findings
1. **The judge evasion is confirmed on verified attacks and is stronger than in v2.** On train-seed rounds the
   judge falls from about 0.5 to 0.12–0.24 at equal FPR, and its AUROC drops to chance. The v2 numbers were
   diluted by payload-free artifacts, not inflated by them. Against the r2 attacker, a 14B model, the judge does
   worse than chance (AUROC 0.46).
2. **The fine-tuned classifiers are not evaded.** They detect ≥0.99 on every train-seed round. They have zero
   genuine detector-passing rewrites across about 500 verified attacks, including a 14B attacker that sees
   saliency feedback.
3. **TF-IDF degrades under the stronger attacker:** 0.95–0.98 on r0/r1, 0.90 on r2 and 0.71 on held-out r2t.
   Its AUROC stays at 0.99, so the drop is a threshold effect.
4. **Context gives no benefit,** confirmed again: at matched FPR it equals plain DeBERTa, while its val-fixed
   realized FPR is 0.24 (plain 0.11).
5. **Held-out caveat.** On the held-out seeds (r1t n=8, r2t n=21, from 2–6 bases) the judge mostly holds (0.63
   and 0.95), so its evasion is shown on train-seed rounds. The held-out sets are small and drawn from the
   held-out risk families. The report should state this rather than generalize.
