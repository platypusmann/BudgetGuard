Experiment design sha256 637c056365be91a25d07155b63dcfe6ec2ac226afea2b970b229a0a3a2b427e5. Code commit c4af881f1e8803adafb974c31fca14c9f958af0f (pre-test-eval), with uncommitted code changes.

Accuracy (lenient) with bootstrap 95 percent CI, SYNTHETIC data, test split, 300 episodes of 100 questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.

| Policy | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Always Tier 0 | 0.129 (0.125 to 0.133) | 0.129 (0.125 to 0.133) | 0.129 (0.125 to 0.133) |
| Always Tier 1 | 0.179 (0.176 to 0.182) | 0.445 (0.439 to 0.451) | 0.445 (0.439 to 0.451) |
| Always Tier 2 | 0.179 (0.176 to 0.182) | 0.338 (0.333 to 0.344) | 0.500 (0.494 to 0.505) |
| Random, matched spend | 0.178 (0.174 to 0.182) | 0.329 (0.323 to 0.334) | 0.512 (0.506 to 0.518) |
| Fixed threshold (ReDAct) | 0.097 (0.094 to 0.100) | 0.219 (0.215 to 0.224) | 0.223 (0.219 to 0.228) |
| Threshold tuned per budget | 0.189 (0.185 to 0.193) | 0.346 (0.341 to 0.352) | 0.514 (0.508 to 0.520) |
| Three-tier threshold tuned per budget | 0.189 (0.185 to 0.193) | 0.458 (0.452 to 0.464) | 0.521 (0.516 to 0.527) |
| FrugalGPT cascade | 0.189 (0.185 to 0.193) | 0.464 (0.458 to 0.470) | 0.528 (0.523 to 0.534) |
| RouteLLM-style | 0.208 (0.205 to 0.212) | 0.363 (0.357 to 0.368) | 0.521 (0.515 to 0.526) |
| Slack schedule | 0.197 (0.192 to 0.201) | 0.469 (0.463 to 0.475) | 0.531 (0.525 to 0.536) |
| Learned router | 0.189 (0.185 to 0.193) | 0.469 (0.463 to 0.474) | 0.531 (0.526 to 0.537) |
| DP oracle | 0.198 (0.194 to 0.202) | 0.472 (0.466 to 0.478) | 0.535 (0.530 to 0.541) |
| Clarify every question (blind) | 0.006 (0.006 to 0.007) | 0.179 (0.176 to 0.183) | 0.491 (0.486 to 0.497) |
| Ambiguity oracle (clarify iff ambiguous) | 0.239 (0.235 to 0.243) | 0.461 (0.455 to 0.467) | 0.587 (0.581 to 0.592) |
| Hindsight oracle | 0.482 (0.478 to 0.486) | 0.685 (0.679 to 0.690) | 0.685 (0.679 to 0.691) |
| Best simple with Tier 1, picked on tune | 0.189 (0.185 to 0.193), FrugalGPT cascade | 0.458 (0.452 to 0.464), Three-tier threshold tuned per budget | 0.528 (0.523 to 0.534), FrugalGPT cascade |

DP minus each reference, accuracy points, paired bootstrap 95 percent CI over the same episodes.

| DP against | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Fixed two-tier threshold (pre-registered reference) | +10.1 (+9.8 to +10.5) | +25.3 (+24.6 to +25.9) | +31.2 (+30.6 to +31.8) |
| Three-tier threshold tuned per budget | +0.9 (+0.6 to +1.2) | +1.4 (+1.2 to +1.6) | +1.4 (+1.1 to +1.7) |
| Best simple with Tier 1, picked on tune | +0.9 (+0.6 to +1.2) | +1.4 (+1.2 to +1.6) | +0.7 (+0.4 to +1.0) |
