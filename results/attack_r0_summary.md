# Attack round 0 (detector-blind) — results

Run: Kaggle 20261013-192044_r0 (commit dc2d95e). Target agent Qwen3-8B (4-bit, greedy). Attacker Qwen3-4B.
220 training-split implicit seeds, N=5 rewrites x T=2 iterations, no detector in the loop.

## Attack generation
- 54/220 seeds (25%) produced a working rewrite (ASR>=0.2 on Qwen3-8B); 44 reached ASR 1.0 on their query.
- By paradigm: Template-2 29/81, Template-3 25/139.
- Fair baseline: on these same seeds MCPTox records ~23% with >=1 success on Qwen3-8b-No-Think. So round 0 did
  NOT raise ASR vs MCPTox; it preserved effectiveness while producing natural-language, marker-free text.
- Qwen3-8B (no-think) is near the most robust MCPTox target (implicit ASR 6% overall vs Phi-4 72%, Qwen3-8B-think 22%).

## Detection (score_attacks, Kaggle 20261014-215006_r0, detectors trained in-run)
Detection rate on the 54 working attacks, at the val-fixed threshold; "original" = same seeds' MCPTox text.
| detector | original @1% | optimized @1% | mean score orig -> opt |
|---|---|---|---|
| B0 TF-IDF+LR | 1.00 | 1.00 | - |
| B3 plain DeBERTa +para | 1.00 | 1.00 | 0.9999 -> 0.995 |
| ours context +para | 1.00 | 1.00 | 0.9999 -> 0.981 |

## Takeaways
- Natural-language rewriting ALONE evades nothing (even TF-IDF): the rewrite still states the cross-tool action,
  which is the detectable content. Marker removal barely moves scores.
- Context go/no-go is still undecided: both detectors catch 100%, no gap.
- Round 0 is detector-blind by design. The real test is the ADAPTIVE attack (detector score fed back as
  feedback), which is round 1.
