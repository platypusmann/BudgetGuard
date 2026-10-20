Experiment design sha256 637c056365be91a25d07155b63dcfe6ec2ac226afea2b970b229a0a3a2b427e5. Code commit e67c5e925702324c4eee1c375b873779e17f5f55 (pre-test-eval).

Accuracy (lenient_plus, secondary, same decisions rescored) with bootstrap 95 percent CI, REAL data, test split, 300 episodes of 100 questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.

| Policy | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Always Tier 0 | 0.133 (0.130 to 0.137) | 0.133 (0.130 to 0.137) | 0.133 (0.130 to 0.137) |
| Always Tier 1 | 0.164 (0.161 to 0.167) | 0.417 (0.411 to 0.422) | 0.417 (0.411 to 0.422) |
| Always Tier 2 | 0.089 (0.087 to 0.092) | 0.272 (0.266 to 0.277) | 0.470 (0.464 to 0.475) |
| Random, matched spend | 0.139 (0.135 to 0.143) | 0.283 (0.279 to 0.289) | 0.480 (0.474 to 0.485) |
| Fixed threshold (ReDAct) | 0.101 (0.096 to 0.106) | 0.243 (0.239 to 0.248) | 0.252 (0.248 to 0.257) |
| Threshold tuned per budget | 0.133 (0.128 to 0.138) | 0.331 (0.326 to 0.337) | 0.487 (0.482 to 0.492) |
| Three-tier threshold tuned per budget | 0.205 (0.201 to 0.209) | 0.450 (0.445 to 0.456) | 0.539 (0.534 to 0.544) |
| FrugalGPT cascade | 0.205 (0.201 to 0.209) | 0.445 (0.439 to 0.450) | 0.518 (0.513 to 0.523) |
| RouteLLM-style | 0.152 (0.147 to 0.156) | 0.353 (0.347 to 0.358) | 0.505 (0.500 to 0.511) |
| Slack schedule | 0.204 (0.200 to 0.209) | 0.458 (0.453 to 0.464) | 0.555 (0.549 to 0.560) |
| Learned router | 0.191 (0.186 to 0.195) | 0.459 (0.453 to 0.464) | 0.557 (0.551 to 0.562) |
| DP oracle | 0.220 (0.216 to 0.224) | 0.463 (0.457 to 0.469) | 0.556 (0.551 to 0.561) |
| Clarify every question (blind) | 0.007 (0.006 to 0.007) | 0.196 (0.194 to 0.199) | 0.524 (0.518 to 0.529) |
| Ambiguity oracle (clarify iff ambiguous) | 0.258 (0.254 to 0.262) | 0.424 (0.418 to 0.430) | 0.604 (0.598 to 0.609) |
| Hindsight oracle | 0.435 (0.430 to 0.441) | 0.693 (0.688 to 0.698) | 0.707 (0.701 to 0.712) |
| Best simple with Tier 1, picked on tune | 0.205 (0.201 to 0.209), FrugalGPT cascade | 0.450 (0.445 to 0.456), Three-tier threshold tuned per budget | 0.539 (0.534 to 0.544), Three-tier threshold tuned per budget |
