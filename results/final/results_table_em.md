Experiment design sha256 637c056365be91a25d07155b63dcfe6ec2ac226afea2b970b229a0a3a2b427e5. Code commit e67c5e925702324c4eee1c375b873779e17f5f55 (pre-test-eval).

Accuracy (em, secondary, same decisions rescored) with bootstrap 95 percent CI, REAL data, test split, 300 episodes of 100 questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.

| Policy | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Always Tier 0 | 0.111 (0.108 to 0.114) | 0.111 (0.108 to 0.114) | 0.111 (0.108 to 0.114) |
| Always Tier 1 | 0.133 (0.130 to 0.136) | 0.339 (0.334 to 0.345) | 0.339 (0.334 to 0.345) |
| Always Tier 2 | 0.073 (0.070 to 0.076) | 0.231 (0.226 to 0.236) | 0.411 (0.406 to 0.417) |
| Random, matched spend | 0.115 (0.111 to 0.119) | 0.242 (0.237 to 0.247) | 0.420 (0.415 to 0.426) |
| Fixed threshold (ReDAct) | 0.086 (0.082 to 0.091) | 0.214 (0.209 to 0.218) | 0.222 (0.218 to 0.226) |
| Threshold tuned per budget | 0.110 (0.105 to 0.114) | 0.290 (0.285 to 0.295) | 0.429 (0.424 to 0.434) |
| Three-tier threshold tuned per budget | 0.174 (0.170 to 0.178) | 0.373 (0.368 to 0.379) | 0.463 (0.458 to 0.468) |
| FrugalGPT cascade | 0.174 (0.170 to 0.178) | 0.370 (0.365 to 0.375) | 0.451 (0.446 to 0.457) |
| RouteLLM-style | 0.132 (0.128 to 0.136) | 0.306 (0.301 to 0.311) | 0.440 (0.435 to 0.446) |
| Slack schedule | 0.175 (0.170 to 0.179) | 0.381 (0.376 to 0.387) | 0.472 (0.466 to 0.477) |
| Learned router | 0.161 (0.156 to 0.165) | 0.379 (0.373 to 0.384) | 0.472 (0.466 to 0.478) |
| DP oracle | 0.191 (0.187 to 0.196) | 0.383 (0.378 to 0.389) | 0.468 (0.463 to 0.474) |
| Clarify every question (blind) | 0.005 (0.004 to 0.006) | 0.162 (0.159 to 0.165) | 0.446 (0.440 to 0.450) |
| Ambiguity oracle (clarify iff ambiguous) | 0.223 (0.219 to 0.226) | 0.360 (0.355 to 0.366) | 0.521 (0.515 to 0.526) |
| Hindsight oracle | 0.362 (0.356 to 0.367) | 0.583 (0.578 to 0.588) | 0.594 (0.589 to 0.600) |
| Best simple with Tier 1, picked on tune | 0.174 (0.170 to 0.178), FrugalGPT cascade | 0.373 (0.368 to 0.379), Three-tier threshold tuned per budget | 0.463 (0.458 to 0.468), Three-tier threshold tuned per budget |
