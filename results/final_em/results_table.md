Experiment design sha256 637c056365be91a25d07155b63dcfe6ec2ac226afea2b970b229a0a3a2b427e5. Code commit 8aa7dca4c20ff1d450b7098f89c534eaac1c219b. Was c0b4fa5 before the history rewrite, see DECISIONS.md 118 to 121.

Accuracy (em) with bootstrap 95 percent CI, REAL data, test split, 300 episodes of 100 questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.

| Policy | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Always Tier 0 | 0.111 (0.108 to 0.114) | 0.111 (0.108 to 0.114) | 0.111 (0.108 to 0.114) |
| Always Tier 1 | 0.133 (0.130 to 0.136) | 0.339 (0.334 to 0.345) | 0.339 (0.334 to 0.345) |
| Always Tier 2 | 0.073 (0.070 to 0.076) | 0.231 (0.226 to 0.236) | 0.411 (0.406 to 0.417) |
| Random, matched spend | 0.112 (0.108 to 0.116) | 0.242 (0.237 to 0.247) | 0.420 (0.415 to 0.426) |
| Fixed threshold (ReDAct) | 0.084 (0.080 to 0.088) | 0.210 (0.206 to 0.214) | 0.217 (0.213 to 0.222) |
| Threshold tuned per budget | 0.111 (0.108 to 0.115) | 0.286 (0.281 to 0.292) | 0.429 (0.424 to 0.434) |
| Three-tier threshold tuned per budget | 0.170 (0.166 to 0.174) | 0.373 (0.367 to 0.378) | 0.463 (0.458 to 0.468) |
| FrugalGPT cascade | 0.170 (0.166 to 0.174) | 0.373 (0.367 to 0.380) | 0.454 (0.448 to 0.459) |
| RouteLLM-style | 0.134 (0.130 to 0.137) | 0.311 (0.306 to 0.316) | 0.440 (0.435 to 0.446) |
| Slack schedule | 0.175 (0.170 to 0.179) | 0.379 (0.373 to 0.384) | 0.470 (0.464 to 0.475) |
| Learned router | 0.174 (0.170 to 0.177) | 0.387 (0.382 to 0.393) | 0.477 (0.471 to 0.482) |
| DP oracle | 0.193 (0.189 to 0.197) | 0.385 (0.380 to 0.391) | 0.471 (0.466 to 0.476) |
| Clarify every question (blind) | 0.006 (0.006 to 0.007) | 0.166 (0.163 to 0.169) | 0.447 (0.442 to 0.453) |
| Ambiguity oracle (clarify iff ambiguous) | 0.220 (0.216 to 0.224) | 0.359 (0.354 to 0.365) | 0.517 (0.511 to 0.522) |
| Hindsight oracle | 0.367 (0.362 to 0.373) | 0.630 (0.625 to 0.635) | 0.645 (0.639 to 0.650) |
| Best simple with Tier 1, picked on tune | 0.170 (0.166 to 0.174), FrugalGPT cascade | 0.373 (0.367 to 0.378), Three-tier threshold tuned per budget | 0.454 (0.448 to 0.459), FrugalGPT cascade |

DP minus each reference, accuracy points, paired bootstrap 95 percent CI over the same episodes.

| DP against | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Fixed two-tier threshold (pre-registered reference) | +10.9 (+10.5 to +11.5) | +17.6 (+17.0 to +18.1) | +25.3 (+24.8 to +25.9) |
| Three-tier threshold tuned per budget | +2.3 (+2.1 to +2.6) | +1.3 (+1.0 to +1.6) | +0.8 (+0.4 to +1.1) |
| Best simple with Tier 1, picked on tune | +2.3 (+2.1 to +2.6) | +1.3 (+1.0 to +1.6) | +1.7 (+1.3 to +2.1) |
