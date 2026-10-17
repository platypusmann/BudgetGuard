# Decisions

Every choice I made while building this out, with the reason. Newest sections go at the bottom of each part. Frozen decisions from the brief (dataset, splits, tiers, prices, prompt, hard cap rule) are not repeated here unless I had to interpret them.

## Setup

1. Pinned JSONL line endings to LF in `common.JsonlAppender` and in the batch input writer. On Windows, text mode wrote CRLF, so split files built on the laptop would hash differently from the same splits built on Linux or Kaggle, and `split.verify()` would break the moment git or another machine touched them. No real splits existed yet, so nothing frozen changed. Added `.gitattributes` so git never rewrites bytes under `data/`.

2. Fixed one flaky line in the smoke test. It printed the first cache row, but with threaded sync calls the first row can be one of the deliberately failed calls, which has no `answer` key. It now picks the first good row.

3. Synthetic outputs live in `data/synthetic/`, which is gitignored and rebuilt in a couple of minutes. Real cache files under `data/cache/` stay tracked so the $0 reproduction works from a clone.

## Grading (pre-registered)

4. Primary metric is lenient match. I'm fixing this before any real answers exist. The reason is that the prompt asks for a short answer but chat models still add harmless words ("The Eiffel Tower in Paris"), and EM marks those wrong for reasons that have nothing to do with routing. The length cap of len(gold) + 4 tokens stops a model from passing by listing several guesses. EM is reported as the robustness check, and evaluate takes `--metric em` to rerun everything on EM.

5. Lenient means the normalized gold appears as a contiguous run of tokens inside the normalized prediction. Plain substring matching would let gold "1" match "1990", so tokens it is.

6. Normalization is SQuAD's (lowercase, drop punctuation, drop a/an/the, squash whitespace) with one change. I also strip unicode punctuation, so "1939–1945" and "1939-1945" come out the same. AmbigNQ golds contain en dashes and curly quotes, and the models output a mix.

7. An answer that normalizes to the empty string is never correct, under any metric. Raw SQuAD gives EM 1 when both sides are empty, which would let "the" match a gold of "a".

8. Every pid is graded against the qid's `gold` field, which for ambiguous questions is the hash-chosen intended interpretation. So answering the raw question only scores when the model answered the reading the user meant, which is the frozen rule.

9. Duplicate cache rows for a pid (an error row then a retry, say) resolve to the last non-error row, the same rule `cache_api.done_ids` uses.

## Synthetic cache

10. The synthetic world goes through the real code. Raw AmbigNQ-shaped rows are fed to `split.build`, so the fixture has the real file format, intent hashing and MANIFEST. Tier 1 and 2 rows are made by building a fake chat completion body and calling `cache_api.cache_row`, so their schema is the real one by construction. Tier 0 rows copy the Kaggle notebook dict key for key, and a test checks the keys against the notebook source.

11. Answers are strings, not labels. grade.py decides correctness, so the synthetic pipeline exercises grading too. Correct answers are sometimes phrased loosely ("The X", "X (Y)", "It was X"), which pass lenient but fail EM, so the two metrics differ by a realistic margin.

12. Latent model: each question has a difficulty z, and tier k succeeds on a clear prompt with probability sigmoid(a_k - z + noise). a_k is solved by bisection so clear-prompt accuracy is about 25, 40 and 55 percent. Measured after grading on the default world it's 27, 41 and 55. On the raw prompt of an ambiguous question the model first picks a reading uniformly, and only the intended one scores, which puts raw ambiguous accuracy at about 10, 15 and 22.

13. Logprob confidence is a noisy, overconfident view of the same latent skill, and it stays high when the model confidently answers the wrong reading. Tier 0 self consistency samples split across readings on ambiguous raw prompts. The resulting signals are informative but far from perfect. Raw p_first has AUROC about 0.71 for Tier 0 correctness, and the calibrated p0 about 0.73. Ambiguity is easier, at about 0.67 from 1 - sc_agree alone and 0.81 for the calibrated belief, which also sees question length and the time-word flag. That's in the range published work reports for token-probability confidence on open QA, and it keeps the routing problem from being trivial. My first pass had Tier 0 at 0.68, below that range, so I cut the confidence noise.

14. Default synthetic size is 3,000 train and 1,000 validation questions, so the whole pipeline runs in a few minutes on the laptop. About half the questions are ambiguous, close to AmbigNQ's rate.

15. A small share (0.5 percent) of API rows get a preceding error row, like a failed call and a good retry, to exercise the dedupe rule. Pilot prompts are written as sync rows and the rest as batch rows, matching the real run plan.

## Features

16. features.py keeps one row per pid with only what policies and grading need: question metadata, per-tier token counts, the cost as charged, logprob features, grades, Tier 0 self consistency and cross-tier agreement. No answer strings. It's about 1.5 MB on the synthetic world and should stay under 10 MB on the real one.

17. Cross-tier agreement (agree_01, agree_02, agree_12) is in the table, but a policy may only read agree_0k after it has paid for tier k. The runner enforces this by masking every signal of an unrevealed tier.

18. Embeddings: MiniLM (all-MiniLM-L6-v2) when `--embed minilm` is passed, which needs sentence-transformers and a one-time model download. The default is a deterministic word and bigram hashing embedding (512 dims, sklearn HashingVectorizer), because the brief says everything must run offline and tests can't download a model. Neither one is fitted, so neither can leak test information. The RouteLLM-style baseline works with either.

19. Cost columns stay float64 (prices around 1e-6 dollars lose digits in float32), everything else float32.

## Episodes and the hard cap

20. Per question the runner works in stages. A policy can probe a tier on the raw prompt (pay for it, see its signals, decide again) or take a final action: answer with tier k (T0, T1, T2), clarify and then have tier k answer the clarified prompt (C0, C1, C2), or SKIP. This one interface covers single-shot routers, the DP (probe Tier 0, then pick), and a FrugalGPT cascade (probe 0, maybe probe 1, then answer).

21. Worst-case cost of an action is input tokens times input price plus max output tokens times output price, plus the fee for a clarify, as the brief says. I use the prompt's real input token count from the cache, since a deployed router can tokenize the prompt before the call. Actual cost can then never exceed worst case, so a masked policy can't break the cap. Env asserts this for every prompt.

22. A masked action is downgraded along a fixed ladder, Ck to Tk, Tk to Tk-1, down to SKIP, and if some tier was already paid for on this question, the runner answers with the best one instead of skipping. This is the same graceful downgrade the SIT320 toy runner used. It's applied to every policy, so budget-blind policies get the cap enforced on them too. Their failure mode shows up as running dry and skipping questions late in the stream, not as violations.

23. Violation rate is the share of episodes where total spend exceeds B. It's zero by construction for masked runs, and a test checks it for every policy. To show what the guard is worth, evaluate also runs the budget-blind policies with masked=False and reports their violation rate and p95 spend.

24. Clarify on a question that isn't ambiguous still costs the fee, then the tier answers the raw prompt again. If that tier was already probed, its answer is reused at no extra cost.

25. Leak check on worst-case costs. The clarified prompt only exists for ambiguous questions, so showing its length to a policy would leak the label. Policies see a clarify worst case built from the raw prompt's length, and the runner masks with the true one. Signals of a tier are blanked until that tier is paid for.

26. Costs are recomputed from cached token counts at one price sheet, standard API prices by default, rather than taken from the usd field. Two reasons: the Tier 0 price sweep needs recomputation anyway, and the cache mixes sync-priced pilot rows with batch-priced rows, which would make the same call cost different amounts depending on whether its question was in the pilot. A live router can't wait a day for batch results, so standard prices are the honest default. `batch_prices: true` in configs/experiment.yaml switches.

27. Budgets and clarify fees are quoted in u2, the mean Tier 2 raw-prompt cost on calib at the evaluation price sheet. The budget grid runs from 0.02 u2 per question (Tier 0 only) to 1.0 u2 (Tier 2 on everything, with room to clarify some). Dollar figures go into every CSV next to the u2 numbers.

28. Episodes are K = 100 distinct questions drawn without replacement from one split, seeded per (seed, split). Every policy sees exactly the same streams, which makes paired comparisons clean. Answers are cached at temperature 0, so stream order is the only randomness.

## Policies

29. Shared signals are fitted once on calib (signals.py): p0 = P(Tier 0 raw answer correct), p_amb = P(ambiguous), both from Tier 0 features plus question length and the time flag, and p1 = P(Tier 1 correct) from Tier 1 features and agreement with Tier 0. All are plain logistic regressions. On calib the values are out of fold (5 folds), so tables estimated on calib, like the DP's success rate per bin, aren't optimistic. Every policy gets the same signals, so differences come from decision rules, not information, as in the toy.

30. Fixed threshold (ReDAct reference): probe Tier 0, escalate to Tier 2 when 1 - p0 is above a threshold, clarify when p_amb is above another. Thresholds are calibrated once on calib so 20 percent of questions escalate and 15 percent clarify, then reused at every budget, fee and price. ReDAct targets a fixed number of large-model calls per episode, and 20 per 100 sits in the middle of my budget grid. Clarify plus escalate is C2, clarify alone is C0. It stays a two-model rule like ReDAct, so it never uses Tier 1.

31. Threshold tuned per budget: same rule, the two thresholds grid searched on tune streams for each budget (10 escalation rates times 6 clarify rates). It knows the budget at tuning time but never reads the remaining budget during an episode.

32. Random at matched spend: probe Tier 0, then clarify and escalate by coin flip. The clarify rate is copied from the tuned threshold at the same budget, and the escalation rate is bisected on tune until mean spend matches the tuned threshold's. Any gap to the tuned threshold is what the signal is worth at equal money.

33. FrugalGPT-style cascade: probe Tier 0, accept if p0 clears tau0, else pay for Tier 1, accept if p1 clears tau1, else Tier 2. A clarify gate sends high-p_amb questions to C2. The three thresholds are tuned on tune per budget (6 x 4 x 3 grid). FrugalGPT has no clarify, so the gate is my addition, and the grid includes "never clarify".

34. Exact DP: backward induction over (questions left, budget left on an 800-level grid, signal bin), with 8 quantile bins of p0 crossed with 3 of p_amb. The value is linearly interpolated between grid points, which keeps it monotone. Success rates per (bin, action) come from calib with 5 pseudo counts of shrinkage toward the action's overall rate. Costs in the recurrence are calib means, and feasibility uses mean worst cases while planning and each question's exact worst case at run time. The reward is additive, so V(0, b) = 0 and each answered question adds its success probability.

35. The DP can also answer directly without probing, using marginal success rates. My first version always probed, and at 0.05 u2 per question it lost to always-Tier-1, because probe plus Tier 1 didn't fit the budget. Adding direct answers fixed that, and the lemma still holds because every option feasible at budget b stays feasible at any larger b. Skipping the question is always an option too.

36. The DP is exact for its own model, not for the world. It only sees Tier 0 signals through 24 bins and plans with mean costs, so a policy with more information can beat it. On synthetic tune streams the FrugalGPT cascade, which also sees Tier 1's confidence, edges it at the loosest budgets. I kept the name "DP oracle" from the proposal, and the README says what it's an oracle for.

37. Learned router: behaviour cloning from the DP, as in proposal section 3.2. Training rows are the DP's decisions on calib streams at every budget in the grid, so one model covers all budgets. The inputs are the standardized Tier 0 features, p0, p_amb, log remaining budget per remaining question (in u2), remaining budget over B and remaining questions over K. Two model classes compete (multinomial logistic and a 32-unit MLP), and tune picks the winner. A second small logistic head imitates the DP's first move (probe, skip or answer directly) from the budget inputs alone. The PPO-Lagrangian fine-tune from the proposal is out of scope. It's a lot of machinery for a single-turn problem the DP already solves, and masking keeps the cap hard anyway. Listed under limitations.

38. Slack threshold schedule: for each slack bucket (remaining budget per remaining question in u2), a first move plus a small table of cuts: clarify when p_amb crosses a cut, then Tier 2 or Tier 1 when 1 - p0 crosses a cut, fitted separately for the clarify and no-clarify branches. It's fitted once across all budgets to the DP's decisions on tune streams, the same recipe as Experiment 4 in the toy. One change from the toy: each cut learns its direction. In the synthetic world, and plausibly on real data, Tier 2's gain over Tier 0 is largest on questions Tier 0 is fairly confident about, because hopeless questions stay hopeless for every tier. A cut that can only fire when uncertainty is high couldn't follow the DP, and agreement went from 69 to 74 percent with signed cuts. The fixed and tuned thresholds keep ReDAct's direction on purpose, since that's the design being tested.

39. RouteLLM-style: logistic regression on the question embedding predicts whether Tier 2 beats Tier 0 on the raw prompt, the RouteLLM win label, and a second one predicts ambiguity. It never probes, so it saves the Tier 0 cost but only sees the question. The two thresholds are tuned on tune per budget.

40. Hindsight oracle: knows every action's correctness and cost for every question in the episode. Reward is one per correct answer, so the optimum is to take the cheapest correct action per question and fill the budget with the cheapest questions first. That greedy is exactly optimal, no discretization needed. It doesn't pay for a probe.

41. Always Tier k has no clarify, on purpose. Those are single-action floors and ceilings. Every other policy can clarify.

42. Policies that tune per budget store parameters per budget and look up the nearest tuned budget (in log space) when run at another. evaluate always tunes at exactly the budgets it evaluates.

## Evaluation

43. Configs: the headline config (clarify fee 1x u2, Tier 0 at 0.02/0.08) runs the full budget grid. The fee sweep (0.5x, 2x) and the Tier 0 price sweep (0, 0.05/0.20) each run at three budgets (0.1, 0.35, 0.75 u2 per question) to keep the laptop run to minutes. Every config refits and retunes every policy, since the best threshold depends on the fee and prices.

44. Evaluation streams: 3 seeds times 100 episodes of 100 questions on the test split. Tuning uses 100 streams on tune (seed 1000), and the learned router's behaviour-cloning data is 60 DP rollouts per budget on calib (seed 2000). CIs are percentile bootstraps over the 300 episodes, 1000 resamples. Paired differences resample the per-episode accuracy difference, which works because every policy sees the same streams. I report differences against both the fixed threshold (as the brief asks) and the tuned threshold, since the tuned one is the fairer budget-blind baseline.

45. Clarify precision is P(ambiguous | clarified) and recall is P(clarified | ambiguous), pooled over every question visit in every episode, against `is_ambiguous`.

46. Budget-blind policies (always Tier k, both thresholds, random, the cascade and RouteLLM) are also run with masked=False, which shows what the cap is worth: their violation rate and p95 spend over B. Budget-aware policies only run masked.

47. The soft frontier is the fixed-threshold rule with its clarify threshold kept and the escalation threshold swept over the 10 rates, run with no cap at all. That's the toy's Experiment 2, the accuracy-only frontier cascade papers report.

48. Decision logs (every decision as a row) are written for the fixed threshold, tuned threshold, DP and learned router at the middle budget of the headline config. Logging every run would be tens of millions of rows. The runner can log any run through `RunResult.log_frame`.

49. `--eval-split tune` exists for development, so I could look at numbers without touching test. The README results come from the one test run.

50. Figures are static PNGs for the README. The seven routing policies get the first seven slots of a validated categorical palette, always in the same order, and floors, controls and bounds are gray. Every synthetic figure carries a red "SYNTHETIC DATA" stamp, and every CSV has a data column saying SYNTHETIC or REAL.
