# Budget Guard

A router that decides, question by question, whether to answer with a cheap model, escalate to a stronger one, or ask the user a clarifying question, while never spending more than a hard dollar cap over a stream of questions.

The results below are on SYNTHETIC data. The pipeline, policies and tests are real, but the real model answers haven't been collected yet. RUNBOOK.md has the steps, and the real run costs under $4.

## Why this exists

Most routing and cascade work treats cost as something to minimize on average. Production systems usually have a cap instead: a per-request ceiling, a per-batch budget, a monthly limit that must not be crossed. A router that's right on average can still blow through the cap on a bad run of hard questions.

This project extends my SIT320 research proposal (in `research/poc`). There, in a toy model, a policy that conditions on remaining budget, solved exactly by dynamic programming, beat a ReDAct-style fixed uncertainty threshold under a hard per-task cap, with zero violations. Budget Guard takes that idea to real questions, real model tiers and real prices, using AmbigNQ, where about half the questions are ambiguous, so asking the user is sometimes the right move.

## How it works

Three tiers answer every question once, and the answers are cached:

| Tier | Model | Price per 1M tokens (in / out) |
|---|---|---|
| 0 | Qwen3-4B-Instruct on Kaggle | free, modelled at 0.02 / 0.08, swept over 0, 0.02/0.08 and 0.05/0.20 |
| 1 | gpt-6-luna, reasoning effort none | 0.10 / 0.50 |
| 2 | gpt-6-sol, reasoning effort none | 2.00 / 10.00 |

An episode is a stream of K = 100 questions with a hard budget B. For each question a policy can probe Tier 0 (pay for it and see its confidence), answer with any tier, clarify (pay a fee, then a tier answers the intended reading), or skip. Before any action the runner computes its worst-case cost (input tokens times input price plus max output tokens times output price, plus the fee) and masks it if that would break the cap. Then it charges the actual cost from the cache. Actual is always at most the worst case, so a masked policy can't go over budget. The tests check this for every policy at every budget, fee and price.

Policies compared:

- Always Tier 0, 1 or 2, the floors and ceilings
- Random at matched spend, the signal-blind control
- Fixed threshold, calibrated once (the ReDAct reference)
- Threshold tuned per budget
- FrugalGPT-style confidence cascade with a clarify gate
- RouteLLM-style router, question embedding plus logistic regression, no probe
- Exact DP over (questions left, budget left, signal bin), the proposal's oracle with additive reward
- Slack threshold schedule fitted to the DP, the proposal's interpretable method
- Learned router, behaviour cloned from the DP, with remaining budget per remaining question as an input
- Hindsight oracle, which knows every answer, as the upper bound
- Two clarify references built on the same DP: clarify every question (blind), and an ambiguity oracle that clarifies exactly when the question is ambiguous

The DP and the learned router also get a free ambiguity score from the question text alone (hand features plus a sentence embedding, fitted on calib), which they can use before paying for any call.

Everything is fitted on calib, tuned on tune, and the test split is only read in the final evaluate step. Every entry point verifies the frozen split hashes first and asserts that no test question reaches fitting data. DECISIONS.md logs every choice along the way.

## Results (SYNTHETIC)

![Cost versus accuracy, synthetic](results/figures/headline_cost_vs_accuracy.png)

Accuracy (lenient match, pre-registered) on the test split, 300 episodes of 100 questions over three stream seeds, clarify fee equal to one mean Tier 2 call, Tier 0 at 0.02/0.08. Budget is per question in u2, the mean Tier 2 call cost (about $0.00015). Brackets are bootstrap 95 percent CIs over episodes. Violation rate is 0 in every row.

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
| Slack schedule | 0.196 (0.192 to 0.200) | 0.328 (0.324 to 0.333) | 0.435 (0.430 to 0.440) |
| Learned router | 0.202 (0.197 to 0.206) | 0.342 (0.337 to 0.347) | 0.439 (0.434 to 0.444) |
| DP oracle | 0.202 (0.198 to 0.206) | 0.330 (0.325 to 0.334) | 0.433 (0.427 to 0.438) |
| Clarify every question (blind) | 0.005 (0.004 to 0.005) | 0.174 (0.171 to 0.177) | 0.392 (0.387 to 0.397) |
| Ambiguity oracle (clarify iff ambiguous) | 0.183 (0.179 to 0.187) | 0.358 (0.353 to 0.363) | 0.485 (0.480 to 0.489) |
| Hindsight oracle | 0.374 (0.369 to 0.378) | 0.672 (0.668 to 0.676) | 0.729 (0.725 to 0.734) |

What the synthetic run says, for what it's worth before real data:

- The toy's story holds. The fixed threshold fails in both directions. At tight budgets it spends early and runs dry (0.101, worse than always Tier 0). At loose budgets it can't use the money (stuck at 0.222).
- Budget-aware policies win in the middle of the range. At 0.35 u2 the DP beats the threshold tuned for that exact budget by 6.2 points (paired 95 percent CI 5.7 to 6.7) and the fixed threshold by 11.2 points (10.6 to 11.8).
- Averaged over the 8 budgets, the learned router closes 102 percent of the gap between the fixed threshold and the DP (it edges the DP), the slack schedule 98 percent, the cascade 91 and the tuned threshold 61. That ordering is close to the toy's (tuned threshold 61, schedule 93).
- The guard matters. Run without masking, the fixed and tuned thresholds go over the cap in 100 percent of episodes at budgets up to 0.2 u2, with p95 spend up to 24 times the budget.
- Clarify is used with good precision but modest recall. The DP clarifies 13 percent of questions at 0.35 u2, and 81 percent of those really are ambiguous, but it only catches 23 percent of the ambiguous ones, because the fee competes with Tier 2 calls.
- Knowing ambiguity exactly is worth a few points. The ambiguity oracle beats the DP by 2.8 points at 0.35 u2 and 5.2 at 1 u2. On a scale where blind clarify-all is 0 and the oracle is 1, the DP sits at 0.85 and the learned router at 0.92 at 0.35 u2 (`results/clarify_reference.csv`).

Other figures in `results/figures`: accuracy by budget, violations and p95 spend without the guard, the clarify fee and Tier 0 price sweeps, and Tier 2 use by remaining slack. Full numbers are in `results/summary.csv` and `results/paired.csv`.

## Reproduce for $0

Python 3.11 or newer, 8 GB RAM, no GPU.

```
pip install -r requirements.txt
python scripts/bg.py test          # offline test suite, about 30 seconds
python scripts/bg.py synthetic     # synthetic world, grading, features, full evaluation, about 4 minutes
```

Once the real cache is in `data/cache`, rebuilding every number and figure from it costs nothing and needs no network:

```
python scripts/bg.py real          # grade, features, evaluate on data/, writes results/
```

Each stage also runs alone (`synth`, `grade`, `features`, `evaluate`, `report`). `python scripts/bg.py --help` lists them, and there's a Makefile wrapper if you have make.

## Real run cost

Worst case, using the same formula `cache_api` enforces (estimated input tokens, every answer at the 24-token max), scaled to AmbigNQ's roughly 12,000 questions and 17,700 prompts:

| Step | Pricing | Worst case |
|---|---|---|
| Pilot, 300 questions (about 450 prompts), Tier 1 and 2 | standard (sync) | about $0.18 |
| Full run, every prompt, Tier 1 and 2 | batch, 50 percent off | about $3.45 more |
| Tier 0 on Kaggle | free GPU | $0 |

So the pilot stays under $0.20 and everything together under $4, before any answer comes in short. `cache_api estimate` prints the exact figure once the real splits exist, and every paid command refuses to run past its cap. RUNBOOK.md walks through it.

## Layout

```
src/budget_guard/
  split.py        frozen calib, tune, test and pilot splits from AmbigNQ, plus prompts.jsonl
  cache_api.py    resumable, budget-guarded API cache, sync and Batch modes
  prompts.py      the one prompt every tier uses
  grade.py        SQuAD normalization, EM, token F1, lenient match
  synthetic.py    synthetic splits and caches through the real code paths
  features.py     per-pid feature table and cached question embeddings
  signals.py      calibrated Tier 0 confidence and ambiguity belief, fitted on calib
  episodes.py     environment, stream sampler, hard-cap runner
  policies/       every routing policy
  evaluate.py     sweeps, bootstrap CIs, paired differences
  report.py       figures and the results table
notebooks/kaggle_tier0.py   Tier 0 on a Kaggle T4, checkpointed and resumable
scripts/bg.py               one command per stage
tests/                      offline pytest suite
research/poc/               the SIT320 proposal, comparison note and toy notebook
```

## Limitations

- These numbers are synthetic. The generator is tuned to land near realistic accuracies (about 25, 40 and 55 percent on clear prompts) and realistic signal quality (AUROC about 0.73 for Tier 0 confidence), but the shape of the results depends on modelling choices I made, such as difficulty shared across tiers. Treat it as a test that the machinery works and a forecast to check, not a finding.
- This is a reframing of the SIT320 formulation, not the formulation itself. The proposal is about multi-step tool-using agents, where one failed step ends the task, the reward is a product over steps, and clarifying helps every remaining step. Here each question is single-turn, the reward is the number correct over a stream (additive), and a clarification only helps the question it was asked on. The remaining budget plays the same role in the state, and the DP, the monotonicity lemma and the hard cap carry over, but the dependence between steps, which is what made the toy interesting, is gone. Irreversible actions and the post-hoc abstention failure mode aren't tested at all.
- That reframing puts this close to TREACLE, which also learns a budget-constrained policy over a stream of independent questions with remaining budget in the state. The differences are the clarify action priced against escalation, the hard per-episode cap enforced by worst-case masking rather than an average, and the exact DP and slack-schedule baselines from the proposal. That's a modest step from TREACLE, not a new problem. The multi-step agent setting from the proposal is still the open part.
- Detecting ambiguity is the weak link on real data. On the 300-question pilot, Tier 0 self consistency and cross-tier agreement carry no ambiguity signal (AUROC about 0.5), and the free text detector reaches 0.65 on real tune. The synthetic world assumed more (0.81 for the Tier 0 belief), so expect the clarify gains above to shrink on real data.
- The DP is exact for its own model (Tier 0 signal in 24 bins, mean costs), not for the world. The cascade, which also sees Tier 1's confidence, ties or slightly beats it at the two loosest budgets, and the learned router matches or slightly beats it at most budgets.
- The learned router is behaviour cloning only. The PPO-Lagrangian fine-tune from the proposal isn't implemented, and with masking already enforcing the cap it would only change how well the router spends, not whether it obeys the cap.
- Clarification is simulated as a perfect user. The answer to a clarifying question is always the hash-chosen intended reading, at a fixed fee. Real users are slower, sometimes vague, and the fee is a modelling choice, which is why it's swept.
- Unlike the toy, the hard-cap DP doesn't sit below a soft frontier here. The soft frontier is the two-tier fixed threshold with no cap, which is a weaker rule than the three-tier DP, so the toy's "price of the guarantee" can't be read off this figure. A fair version would need an uncapped DP, which isn't built yet.
- Tier 0's price is a modelling choice since Kaggle is free. The sweep shows how much it matters. Evaluation uses standard API prices, because a live router can't wait for batch results.
- One sample per tier at temperature 0, on one dataset, one prompt. The lenient metric was pre-registered, and EM is the robustness check (`--metric em`).
- The RouteLLM-style baseline uses a hashing embedding by default so it runs offline. Pass `--embed minilm` for MiniLM, which downloads the model once.
