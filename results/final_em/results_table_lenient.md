Experiment design sha256 637c056365be91a25d07155b63dcfe6ec2ac226afea2b970b229a0a3a2b427e5. Code commit 8aa7dca4c20ff1d450b7098f89c534eaac1c219b. Was c0b4fa5 before the history rewrite, see DECISIONS.md 118 to 121.

Accuracy (lenient, secondary, same decisions rescored) with bootstrap 95 percent CI, REAL data, test split, 300 episodes of 100 questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.

| Policy | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Always Tier 0 | 0.133 (0.129 to 0.136) | 0.133 (0.129 to 0.136) | 0.133 (0.129 to 0.136) |
| Always Tier 1 | 0.163 (0.159 to 0.166) | 0.413 (0.408 to 0.419) | 0.413 (0.408 to 0.419) |
| Always Tier 2 | 0.089 (0.086 to 0.092) | 0.270 (0.265 to 0.275) | 0.467 (0.462 to 0.472) |
| Random, matched spend | 0.134 (0.130 to 0.138) | 0.282 (0.277 to 0.288) | 0.477 (0.472 to 0.482) |
| Fixed threshold (ReDAct) | 0.098 (0.093 to 0.102) | 0.238 (0.234 to 0.242) | 0.246 (0.242 to 0.250) |
| Threshold tuned per budget | 0.133 (0.129 to 0.137) | 0.327 (0.321 to 0.332) | 0.484 (0.479 to 0.489) |
| Three-tier threshold tuned per budget | 0.199 (0.195 to 0.203) | 0.445 (0.439 to 0.450) | 0.533 (0.528 to 0.538) |
| FrugalGPT cascade | 0.199 (0.195 to 0.203) | 0.440 (0.434 to 0.446) | 0.515 (0.510 to 0.520) |
| RouteLLM-style | 0.152 (0.149 to 0.156) | 0.355 (0.350 to 0.360) | 0.501 (0.496 to 0.506) |
| Slack schedule | 0.205 (0.201 to 0.210) | 0.453 (0.447 to 0.458) | 0.545 (0.539 to 0.550) |
| Learned router | 0.209 (0.206 to 0.213) | 0.467 (0.462 to 0.473) | 0.555 (0.550 to 0.561) |
| DP oracle | 0.224 (0.220 to 0.228) | 0.460 (0.455 to 0.466) | 0.542 (0.537 to 0.548) |
| Clarify every question (blind) | 0.007 (0.007 to 0.008) | 0.190 (0.187 to 0.193) | 0.520 (0.514 to 0.525) |
| Ambiguity oracle (clarify iff ambiguous) | 0.251 (0.247 to 0.256) | 0.418 (0.412 to 0.424) | 0.591 (0.586 to 0.597) |
| Hindsight oracle | 0.367 (0.362 to 0.373) | 0.630 (0.625 to 0.635) | 0.645 (0.639 to 0.650) |
| Best simple with Tier 1, picked on tune | 0.199 (0.195 to 0.203), FrugalGPT cascade | 0.445 (0.439 to 0.450), Three-tier threshold tuned per budget | 0.515 (0.510 to 0.520), FrugalGPT cascade |
