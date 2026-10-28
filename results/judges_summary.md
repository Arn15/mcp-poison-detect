# LLM-judge grid (robust6a/6b, commit 4df78d6) — supersedes robust4 for the 14B judges

Judges score the Yes/No logit margin (robust4 used sigmoid(margin), which saturated to 1.0 for the 14B judges and
made their 1%-FPR point undefined). 8B results are identical in robust4 and robust6a (same ranking; deterministic).
Few-shot demos: 6 train-split tools (3 implicit poisonings from distinct paradigms, 3 benign incl. hard negatives),
attack-base seeds excluded; same demo ids in robust4 and robust6 (judge_shots.json).
Classifier rows for comparison: robust5 (mean of 3 seeds). Matched threshold = equal realized test FPR.

## Detection on control-verified OPTIMIZED rewrites (original descriptions of the same bases in parentheses)

| defender | FPR | r0 | r1 | r1b | r2 (14B attacker) | r1t held-out (n=8) | r2t held-out (n=21) |
|---|---|---|---|---|---|---|---|
| 8B zero-shot | 1% | 0.12 (0.48) | 0.24 (0.57) | 0.21 (0.56) | 0.21 (0.45) | 0.63 | 0.95 |
| 8B 6-shot | 1% | 0.30 (0.74) | 0.36 (0.71) | 0.33 (0.67) | 0.26 (0.73) | 0.25 | 0.95 |
| 14B zero-shot | 1% | 0.19 (0.30) | 0.15 (0.33) | 0.14 (0.33) | 0.20 (0.55) | 0.63 | 0.33 |
| 14B 6-shot | 1% | 0.38 (0.57) | 0.34 (0.57) | 0.31 (0.50) | 0.15 (0.64) | 0.75 | 0.52 |
| 14B 6-shot | 5% | 0.76 (0.91) | 0.78 (0.95) | 0.72 (0.94) | 0.61 (1.00) | 1.00 | 1.00 |
| TF-IDF + LR | 1% | 0.95 | 0.98 | 0.97 | 0.90 | 1.00 | 0.71 |
| E5 / DeBERTa / context (fine-tuned) | 1% | ≥0.99 | 1.00 | 1.00 | ≥0.99 | 1.00 | 1.00 |

AUROC on optimized rewrites vs test negatives: 8B zs 0.46-0.55 (chance), 8B 6-shot 0.79-0.90, 14B zs 0.86-0.87,
14B 6-shot 0.94-0.97; fine-tuned classifiers ≥0.997.

## Findings
1. Scale and few-shot examples make the judge a much better *ranker* (AUROC 0.5 -> 0.96), but none of the four judges
   is usable at a deployment-relevant 1% FPR: the best (14B 6-shot) catches 15-38% of optimized attacks and only
   50-64% of the unmodified originals. At 5% FPR it reaches 61-78%. Fine-tuned classifiers catch ≥0.99 at 1%.
2. The optimizer's rewrites transfer to every judge even though no attack was optimized against a judge
   (r0 detector-blind; r1/r2 optimized against DeBERTa): detection drops from orig to opt for all judges, most
   for the 14B attacker's r2 (14B 6-shot 0.64 -> 0.15).
3. Held-out seeds: the 8B judges still mostly catch r2t (0.95), but the 14B judges do not (0.33 / 0.52), so judge
   evasion is no longer confined to the train-seed rounds. Held-out sets remain small (8 and 21).
4. Cost: on one T4, 14B 6-shot took 9.3 h for ~1,700 items vs 51 min for 8B 6-shot (pre-quantized 14B checkpoint;
   the fp16-compute override did not speed it up).
