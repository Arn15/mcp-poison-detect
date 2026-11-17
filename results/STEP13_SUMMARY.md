# Step 13 — natural-text white-box attack (commit afadd7a; results/nat_*, results/fluency_nat_*)

Same white-box attack as Steps 6/11 (30% word budget, 100 steps, payload protected, agent no-poison check) plus two
naturalness constraints: every swap must be one of distilroberta-base's top-50 predictions in context, and the
description's GPT-2 log-perplexity may rise by at most 0.25 nats/token. 379 control-verified attacks, seed 42.

| target | evaders | genuine evasions (evade + still hijack agent) | words changed | median GPT-2 NLL of evaders |
|---|---|---|---|---|
| DeBERTa base, unconstrained 30% (wb3_base_b30_deb) | 252 | 81 | ~7.5 | 6.13 |
| DeBERTa base, **natural** | 47 | **16** (6 seeds) | 7.7 | **4.45** |
| DeBERTa **final system** (round-2 adv + calibrated), natural | 2 | **0** | 8.5 | 4.36 |
| E5 base, unconstrained 30% (wb3_base_b30_e5) | 311 | 82 | ~7.5 | 6.43 |
| E5 base, **natural** | 36 | **6** (4 seeds) | 8.1 | **4.36** |
| E5 **final system**, natural | 7 | **0** | 11.6 | 4.12 |

Reference: benign test tools median NLL 4.16; the GPT-2 fluency filter (threshold on val negatives) flags 0% of
any evader set at 1% or 5% FPR, natural or not.

- Fluent evasions exist against undefended classifiers: 16 (DeBERTa) / 6 (E5) genuine evasions whose text is as
  fluent as benign tools (NLL 4.4-4.5 vs 4.2), ~8 contextually plausible word swaps. The constraint cuts the attack
  ~5-14x (81 -> 16, 82 -> 6) but does not eliminate it.
- The final system stops them: 2 / 7 natural evaders against the calibrated round-2 models, 0 genuine (none still
  hijacks the agent) — although adversarial training only ever saw unconstrained (unnatural) rewrites.
- Qualitative: successful natural swaps soften the precondition wording ("Before" -> "Easily", "call" -> "include",
  "ensure" -> "where") while the agent still follows the injected cross-tool instruction.
- Caveats: one seed per target; 11/16 and 5/6 genuine verdicts rest on 1-2 agent queries; naturalness measured by
  automatic proxies (MLM plausibility, GPT-2 NLL), not human judgement.
