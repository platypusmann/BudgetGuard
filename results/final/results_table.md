Experiment design sha256 637c056365be91a25d07155b63dcfe6ec2ac226afea2b970b229a0a3a2b427e5. Code commit a6175ad2ed1d74664057dc9b93b86c3227fbbceb (pre-test-eval). Was e67c5e9 before the history rewrite, see DECISIONS.md 118 to 121.

Accuracy (lenient) with bootstrap 95 percent CI, REAL data, test split, 300 episodes of 100 questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.

| Policy | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Always Tier 0 | 0.133 (0.129 to 0.136) | 0.133 (0.129 to 0.136) | 0.133 (0.129 to 0.136) |
| Always Tier 1 | 0.163 (0.159 to 0.166) | 0.413 (0.408 to 0.419) | 0.413 (0.408 to 0.419) |
| Always Tier 2 | 0.089 (0.086 to 0.092) | 0.270 (0.265 to 0.275) | 0.467 (0.462 to 0.472) |
| Random, matched spend | 0.138 (0.134 to 0.142) | 0.282 (0.277 to 0.287) | 0.477 (0.472 to 0.482) |
| Fixed threshold (ReDAct) | 0.101 (0.096 to 0.106) | 0.243 (0.238 to 0.247) | 0.251 (0.247 to 0.256) |
| Threshold tuned per budget | 0.133 (0.128 to 0.137) | 0.331 (0.326 to 0.336) | 0.484 (0.479 to 0.489) |
| Three-tier threshold tuned per budget | 0.205 (0.200 to 0.209) | 0.446 (0.440 to 0.452) | 0.533 (0.528 to 0.538) |
| FrugalGPT cascade | 0.205 (0.200 to 0.209) | 0.441 (0.435 to 0.446) | 0.514 (0.509 to 0.519) |
| RouteLLM-style | 0.151 (0.147 to 0.156) | 0.352 (0.347 to 0.357) | 0.501 (0.496 to 0.506) |
| Slack schedule | 0.204 (0.199 to 0.208) | 0.454 (0.448 to 0.459) | 0.549 (0.543 to 0.554) |
| Learned router | 0.190 (0.185 to 0.194) | 0.454 (0.448 to 0.460) | 0.551 (0.545 to 0.557) |
| DP oracle | 0.219 (0.215 to 0.224) | 0.458 (0.453 to 0.464) | 0.551 (0.545 to 0.556) |
| Clarify every question (blind) | 0.006 (0.006 to 0.007) | 0.194 (0.191 to 0.197) | 0.518 (0.513 to 0.524) |
| Ambiguity oracle (clarify iff ambiguous) | 0.256 (0.251 to 0.260) | 0.420 (0.414 to 0.426) | 0.597 (0.591 to 0.603) |
| Hindsight oracle | 0.435 (0.430 to 0.441) | 0.693 (0.688 to 0.698) | 0.707 (0.701 to 0.712) |
| Best simple with Tier 1, picked on tune | 0.205 (0.200 to 0.209), FrugalGPT cascade | 0.446 (0.440 to 0.452), Three-tier threshold tuned per budget | 0.533 (0.528 to 0.538), Three-tier threshold tuned per budget |

DP minus each reference, accuracy points, paired bootstrap 95 percent CI over the same episodes.

| DP against | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Fixed two-tier threshold (pre-registered reference) | +11.9 (+11.4 to +12.4) | +21.6 (+20.9 to +22.1) | +29.9 (+29.3 to +30.5) |
| Three-tier threshold tuned per budget | +1.5 (+1.2 to +1.7) | +1.2 (+0.9 to +1.5) | +1.8 (+1.4 to +2.1) |
| Best simple with Tier 1, picked on tune | +1.5 (+1.2 to +1.7) | +1.2 (+0.9 to +1.5) | +1.8 (+1.4 to +2.1) |
