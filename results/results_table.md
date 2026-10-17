Accuracy (lenient) with bootstrap 95 percent CI, SYNTHETIC data, test split, 300 episodes of 100 questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.

| Policy | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Always Tier 0 | 0.178 (0.173 to 0.182) | 0.178 (0.173 to 0.182) | 0.178 (0.173 to 0.182) |
| Always Tier 1 | 0.118 (0.115 to 0.121) | 0.283 (0.278 to 0.287) | 0.283 (0.278 to 0.287) |
| Always Tier 2 | 0.118 (0.115 to 0.121) | 0.245 (0.240 to 0.249) | 0.380 (0.376 to 0.385) |
| Random, matched spend | 0.187 (0.183 to 0.191) | 0.267 (0.263 to 0.272) | 0.390 (0.385 to 0.395) |
| Fixed threshold (ReDAct) | 0.101 (0.097 to 0.104) | 0.218 (0.213 to 0.222) | 0.222 (0.217 to 0.227) |
| Threshold tuned per budget | 0.184 (0.180 to 0.188) | 0.267 (0.263 to 0.272) | 0.428 (0.423 to 0.432) |
| FrugalGPT cascade | 0.185 (0.181 to 0.190) | 0.299 (0.295 to 0.304) | 0.438 (0.432 to 0.443) |
| RouteLLM-style | 0.206 (0.201 to 0.210) | 0.277 (0.272 to 0.281) | 0.404 (0.399 to 0.409) |
| Slack schedule | 0.202 (0.198 to 0.206) | 0.331 (0.327 to 0.335) | 0.416 (0.410 to 0.422) |
| Learned router | 0.198 (0.194 to 0.202) | 0.346 (0.342 to 0.351) | 0.441 (0.437 to 0.446) |
| DP oracle | 0.198 (0.193 to 0.202) | 0.344 (0.339 to 0.348) | 0.442 (0.437 to 0.446) |
| Hindsight oracle | 0.374 (0.369 to 0.378) | 0.672 (0.668 to 0.676) | 0.729 (0.725 to 0.734) |
