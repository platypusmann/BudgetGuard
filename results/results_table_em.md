Experiment design sha256 637c056365be91a25d07155b63dcfe6ec2ac226afea2b970b229a0a3a2b427e5.

Accuracy (em, secondary, same decisions rescored) with bootstrap 95 percent CI, SYNTHETIC data, test split, 300 episodes of 100 questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.

| Policy | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Always Tier 0 | 0.108 (0.105 to 0.111) | 0.108 (0.105 to 0.111) | 0.108 (0.105 to 0.111) |
| Always Tier 1 | 0.151 (0.148 to 0.154) | 0.377 (0.371 to 0.383) | 0.377 (0.371 to 0.383) |
| Always Tier 2 | 0.151 (0.148 to 0.154) | 0.285 (0.280 to 0.290) | 0.422 (0.416 to 0.428) |
| Random, matched spend | 0.150 (0.146 to 0.153) | 0.277 (0.272 to 0.283) | 0.430 (0.425 to 0.437) |
| Fixed threshold (ReDAct) | 0.081 (0.078 to 0.084) | 0.183 (0.179 to 0.187) | 0.187 (0.183 to 0.191) |
| Threshold tuned per budget | 0.158 (0.154 to 0.161) | 0.292 (0.287 to 0.297) | 0.431 (0.425 to 0.436) |
| FrugalGPT cascade | 0.158 (0.154 to 0.161) | 0.393 (0.388 to 0.399) | 0.446 (0.440 to 0.451) |
| RouteLLM-style | 0.176 (0.172 to 0.180) | 0.307 (0.301 to 0.313) | 0.438 (0.432 to 0.444) |
| Slack schedule | 0.166 (0.162 to 0.170) | 0.398 (0.393 to 0.404) | 0.449 (0.443 to 0.454) |
| Learned router | 0.159 (0.155 to 0.163) | 0.398 (0.392 to 0.404) | 0.452 (0.446 to 0.458) |
| DP oracle | 0.166 (0.162 to 0.170) | 0.401 (0.396 to 0.407) | 0.457 (0.451 to 0.462) |
| Clarify every question (blind) | 0.005 (0.004 to 0.005) | 0.150 (0.147 to 0.154) | 0.416 (0.411 to 0.422) |
| Ambiguity oracle (clarify iff ambiguous) | 0.202 (0.198 to 0.205) | 0.389 (0.384 to 0.395) | 0.492 (0.486 to 0.498) |
| Hindsight oracle | 0.413 (0.409 to 0.417) | 0.582 (0.577 to 0.588) | 0.583 (0.577 to 0.588) |
