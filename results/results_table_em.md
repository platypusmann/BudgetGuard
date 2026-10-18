Accuracy (em, secondary, same decisions rescored) with bootstrap 95 percent CI, SYNTHETIC data, test split, 300 episodes of 100 questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.

| Policy | budget 0.02 u2 | budget 0.35 u2 | budget 1 u2 |
|---|---|---|---|
| Always Tier 0 | 0.144 (0.140 to 0.148) | 0.144 (0.140 to 0.148) | 0.144 (0.140 to 0.148) |
| Always Tier 1 | 0.099 (0.096 to 0.102) | 0.238 (0.234 to 0.243) | 0.238 (0.234 to 0.243) |
| Always Tier 2 | 0.099 (0.096 to 0.102) | 0.201 (0.197 to 0.205) | 0.307 (0.303 to 0.311) |
| Random, matched spend | 0.153 (0.149 to 0.157) | 0.217 (0.213 to 0.221) | 0.316 (0.311 to 0.321) |
| Fixed threshold (ReDAct) | 0.083 (0.079 to 0.086) | 0.171 (0.167 to 0.175) | 0.174 (0.170 to 0.178) |
| Threshold tuned per budget | 0.150 (0.146 to 0.154) | 0.216 (0.212 to 0.220) | 0.345 (0.340 to 0.350) |
| FrugalGPT cascade | 0.151 (0.148 to 0.155) | 0.242 (0.237 to 0.246) | 0.350 (0.345 to 0.355) |
| RouteLLM-style | 0.167 (0.163 to 0.172) | 0.224 (0.220 to 0.229) | 0.331 (0.327 to 0.336) |
| Slack schedule | 0.161 (0.157 to 0.164) | 0.276 (0.271 to 0.280) | 0.359 (0.354 to 0.364) |
| Learned router | 0.162 (0.158 to 0.166) | 0.286 (0.282 to 0.291) | 0.365 (0.360 to 0.370) |
| DP oracle | 0.166 (0.163 to 0.170) | 0.274 (0.270 to 0.279) | 0.361 (0.356 to 0.366) |
| Clarify every question (blind) | 0.004 (0.003 to 0.004) | 0.142 (0.139 to 0.145) | 0.325 (0.321 to 0.330) |
| Ambiguity oracle (clarify iff ambiguous) | 0.149 (0.145 to 0.152) | 0.297 (0.292 to 0.302) | 0.403 (0.398 to 0.408) |
| Hindsight oracle | 0.312 (0.308 to 0.317) | 0.563 (0.558 to 0.568) | 0.603 (0.598 to 0.609) |
