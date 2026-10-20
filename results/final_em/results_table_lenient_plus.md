Experiment design sha256 637c056365be91a25d07155b63dcfe6ec2ac226afea2b970b229a0a3a2b427e5. Code commit c0b4fa5c294ec97fde6c3e34721a5940ae1aa373.

Accuracy (lenient_plus, secondary, same decisions rescored) with bootstrap 95 percent CI, REAL data, test split, 300 episodes of 100 questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.

| Policy | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Always Tier 0 | 0.133 (0.130 to 0.137) | 0.133 (0.130 to 0.137) | 0.133 (0.130 to 0.137) |
| Always Tier 1 | 0.164 (0.161 to 0.167) | 0.417 (0.411 to 0.422) | 0.417 (0.411 to 0.422) |
| Always Tier 2 | 0.089 (0.087 to 0.092) | 0.272 (0.266 to 0.277) | 0.470 (0.464 to 0.475) |
| Random, matched spend | 0.135 (0.131 to 0.139) | 0.284 (0.279 to 0.289) | 0.480 (0.474 to 0.485) |
| Fixed threshold (ReDAct) | 0.098 (0.093 to 0.103) | 0.239 (0.235 to 0.243) | 0.247 (0.242 to 0.251) |
| Threshold tuned per budget | 0.134 (0.130 to 0.137) | 0.327 (0.322 to 0.333) | 0.487 (0.482 to 0.492) |
| Three-tier threshold tuned per budget | 0.200 (0.195 to 0.204) | 0.449 (0.443 to 0.455) | 0.539 (0.533 to 0.544) |
| FrugalGPT cascade | 0.200 (0.195 to 0.204) | 0.443 (0.437 to 0.449) | 0.519 (0.514 to 0.524) |
| RouteLLM-style | 0.152 (0.149 to 0.156) | 0.356 (0.351 to 0.361) | 0.505 (0.500 to 0.511) |
| Slack schedule | 0.206 (0.201 to 0.210) | 0.457 (0.451 to 0.463) | 0.550 (0.544 to 0.556) |
| Learned router | 0.210 (0.207 to 0.214) | 0.471 (0.466 to 0.477) | 0.561 (0.555 to 0.566) |
| DP oracle | 0.225 (0.220 to 0.229) | 0.465 (0.459 to 0.470) | 0.548 (0.542 to 0.553) |
| Clarify every question (blind) | 0.007 (0.007 to 0.008) | 0.191 (0.188 to 0.194) | 0.525 (0.520 to 0.530) |
| Ambiguity oracle (clarify iff ambiguous) | 0.254 (0.249 to 0.258) | 0.423 (0.417 to 0.428) | 0.599 (0.592 to 0.604) |
| Hindsight oracle | 0.367 (0.362 to 0.373) | 0.630 (0.625 to 0.635) | 0.645 (0.639 to 0.650) |
| Best simple with Tier 1, picked on tune | 0.200 (0.195 to 0.204), FrugalGPT cascade | 0.449 (0.443 to 0.455), Three-tier threshold tuned per budget | 0.519 (0.514 to 0.524), FrugalGPT cascade |
