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

## Scripts, docs and the real run

51. Stages run through `scripts/bg.py`, a small Python dispatcher, because Windows has no make by default. The Makefile is a thin wrapper over it for anyone who has make.

52. `scripts/hand_check.py` prints a random sample of graded pilot answers per tier plus health numbers (model returned, missing logprobs, max-token hits, spend), for the hand check step of the runbook. It reads the cache, so it costs nothing.

53. Real-run cost in the README comes from `cache_api.worst_case_cost` applied to the synthetic prompts (mean estimated input of about 74 tokens, which is close to real AmbigNQ question length), scaled to roughly 17,700 real prompts. Pilot worst case about $0.18 at standard prices, full batch about $3.45 more. The runbook makes `cache_api estimate` on the real splits the source of truth before any paid command.

54. The synthetic results are committed under `results/` so the README figure renders from a clone, and every artifact is stamped SYNTHETIC. The real run overwrites the same files, and the runbook says to swap the README numbers then.

55. The README says plainly that the soft frontier here is a weaker two-tier rule, so the hard-cap DP sits above it rather than below. The toy's "price of the guarantee" comparison needs an uncapped version of the DP, which isn't built.

56. Two features stayed out of scope: the PPO-Lagrangian fine-tune of the learned router, and a DP that can probe Tier 1 before deciding. Both are listed as limitations rather than half-built.

## Pilot hand check and lenient_plus (2026-10-19)

57. The pilot hand check came in at about 26 of 30 per tier for Tier 1 and Tier 2, against the 27 of 30 bar in the runbook. That count is Claude's read of the sampled rows, not mine. Every disagreement was a false negative, meaning lenient marked a right answer wrong. There were four patterns:
    - Date order or format: "1 July 2017" against gold "July 1, 2017".
    - Ordinals: "78th" against gold "78".
    - Under-specified dates: "1963" against gold "July 1, 1963". Arguable, and left as wrong.
    - Synonyms: "Plasma membrane" against gold "cell membrane". No string rule can fix this one.

58. The primary metric is unchanged. Lenient was pre-registered before any real answers existed, and it stays the metric every headline number is reported on.

59. lenient_plus is added as a secondary metric only. It's lenient after two more normalizations applied to both prediction and gold: ordinal suffixes are stripped from numbers, and complete day-month-year dates in either order are rewritten to one canonical token. It has no synonym handling, no looser containment and no partial date matching, so "1963" still fails against "July 1, 1963" and "plasma membrane" still fails against "cell membrane".

60. lenient_plus also accepts anything lenient accepts. Collapsing a full date hides its year, so without this rule "September 14, 1814" stopped matching a gold of "1814", which lenient accepts. On the pilot, the literal version flipped 5 lenient matches to wrong, which would have made the secondary metric stricter than the primary in places. The rule adds no new partial date matching.

61. lenient_plus was designed using calib pilot rows only. The test split was not read, graded or looked at. On the pilot it moves accuracy by very little: Tier 0 from 12.4 to 12.6 percent, Tier 1 from 43.2 to 43.7, Tier 2 from 51.4 to 51.9.

62. evaluate must report the headline under both metrics and flag any policy whose ranking differs between them. The same decisions are rescored under lenient_plus, without refitting, so any change comes from the grader alone.

63. Hand count, 2026-10-19, Ranveer. 30 pilot rows per tier, judged against the gold answers only, not real-world truth.
Agreement with the automated lenient grader: Tier 0 30/30, Tier 1 26/30, Tier 2 27/30
Grader said wrong, I say right: Tier 0 0, Tier 1 4, Tier 2 3
Grader said right, I say wrong: 0 on all tiers
Disagreement rows: Tier 1 #5 (date order), #17 (Fifth vs 5th), #26 (same four majors reworded), #30 (78th vs 78). Tier 2 #5 (78th vs 78), #13 (plasma membrane vs cell membrane), #20 (paraphrase).
Conventions: partial answers counted as wrong (Tier 0 #4, Tier 1 #8, Tier 2 #8). Answers that are right in the real world but differ from the gold counted as wrong. Examples are Tier 1 #1, #6, #9, #16, #18, #23 and Tier 2 #16, #19. These are gold noise or the ambiguity penalty, not grader errors.
Borderline calls were settled with Claude's input.
Compared with Claude Code's read of about 26/30 per tier: Tier 1 matches, Tier 2 is one better, Tier 0 is clean.
Bar of 27/30: met for Tier 0 and Tier 2, missed by one for Tier 1.
Outcome: primary metric unchanged, lenient_plus stays secondary, evaluate reports both. Date order and the two 78th rows should be caught by lenient_plus. The rest are out of its scope by design.

64. Fixed a bug found while running the pilot check. The synthetic marker test was `(data_dir / "SYNTHETIC").exists()`, and Windows paths ignore case, so `data/SYNTHETIC` matched the `data/synthetic/` folder. The real evaluation would have stamped its CSVs and figures SYNTHETIC. It now requires the marker to be a file, and a test covers it.

65. The pilot signal check (`python scripts/bg.py pilot-check`) reads pilot.jsonl, the cache rows of pilot pids and the graded files, nothing else. It deliberately doesn't go through features.py, which builds metadata rows for every split including test. It asserts that the pilot is a subset of calib and shares no qid with test. AUROC is reported raw, not flipped, so a feature that runs the other way (entropy, distinct answers) shows below 0.5. CIs are 2,000-resample percentile bootstraps over questions.

## Ambiguity detection and clarify references (2026-10-19)

66. Negative result first. On the 300 calib pilot questions, Tier 0 self consistency does not predict ambiguity: sc_agree AUROC 0.52 [0.46, 0.59], sc_distinct 0.48. The other Tier 0 features sit near 0.5 too. Cross-tier agreement on the raw prompt is no better, with Tier 1 and 2 answers matching 0.48 [0.42, 0.53] and Tier 0 matching either 0.51 [0.47, 0.56]. The only weak signal is Tier 2's own confidence, lp_mean 0.41 [0.34, 0.47], meaning slightly less sure on ambiguous questions, and that one costs a Tier 2 call. My synthetic world had assumed 0.67 for sc_agree, so the synthetic clarify results overstate how well a router can find ambiguous questions.

67. That's why the free text detector was added (ambiguity_text.py). The question text costs nothing and is known before any call, and AmbigNQ labels every question, so it trains on all 7,119 calib questions instead of the 300 pilot rows. Three versions, trained on calib and scored on tune (2,917 questions, 47.5 percent ambiguous): hand features 0.611 [0.592, 0.631], MiniLM embedding 0.636 [0.616, 0.656], both combined 0.648 [0.629, 0.668]. Modest, but clearly above chance and above every model-based signal on the pilot.

68. The detector uses logistic regression with C chosen by 5-fold CV on calib (AUROC scoring). The grid runs from 1e-5 to 10 because the first grid's lowest value was chosen, and a best value sitting at the grid edge suggests the edge is binding. Only calib and tune questions were embedded. The embedding cache is incremental per split, so test is embedded only at final evaluation.

69. Risk-coverage for Tier 0 on the pilot: no confident slice of Tier 0 answers reaches Tier 1. Sorted by lp_min, the top 5 percent (15 questions) are right 33 percent of the time against 47 percent for Tier 1 on the same questions, and the gap is clearer at 20 and 40 percent coverage (17 and 16 points, CIs exclude 0). sc_agree can't even make the slices: 44 percent of answers have all five samples agreeing, so every coverage up to 40 percent is the same tie group, which is right 21 percent of the time. Slices are tie-aware, so this doesn't depend on sort order. Tier 1 and 2 are compared on the same questions, not only overall, because a confident Tier 0 slice is also an easier one.

70. The free detector's score p_text is a signal that needs nothing, so a policy can read it at stage 0. The DP gets a text axis (bins of p_text) on top of the post-probe bins. The tables are shrunk toward the pooled tables so thin cells borrow strength, and the number of text bins is picked on tune from 1, 2 or 3, where 1 means the signal is ignored. The learned router gets p_text as an input to both its main model and its first-move head. The text detector in evaluate uses hand features plus whatever embedding the env has (hash on the synthetic world, MiniLM on real data with `--embed minilm`).

71. On the synthetic world, tune picked 3 text bins by half a point (0.328 against 0.323 with the signal ignored), but on the synthetic test streams the DP came out about one point lower at most budgets than the earlier run without the text axis. The learned router moved by less than half a point. I didn't switch the setting back, because that would be selecting on test, even synthetic test. The likely cause is thinner table cells (synthetic calib is only 2,062 questions, split three ways) plus a selection margin within tune noise. Real calib is 3.5 times larger. Worth watching on the real run.

72. Two clarify reference policies, both restricted versions of the same DP, so they differ from it only in the clarify rule and keep its budget-aware tier choice: clarify_all (clarify actions only, blind), and ambiguity_oracle (the true label as its text axis, clarify actions only on ambiguous questions and plain answers only on clear ones). Under a tight budget the runner can still downgrade an unaffordable clarify, so the oracle clarifies exactly when ambiguous whenever a clarify fits. At 0.02 u2 per question clarify_all is near 0, because almost nothing it is allowed to do fits.

73. evaluate places every policy between the references: position = (accuracy minus clarify_all) over (ambiguity_oracle minus clarify_all), written to `results/clarify_reference.csv`, undefined when the references are within half a point. Positions above 1 at tight budgets are expected, because forcing a clarify on every ambiguous question is a poor use of a small budget, and below 0 at loose budgets for policies that never clarify. On synthetic data at 0.35 u2, the DP sits at 0.85 and the learned router at 0.92, and perfect ambiguity knowledge is worth 2.8 points there and 5.2 at 1 u2.

74. Recommendation on the paid LLM ambiguity probe (asking a model whether the question is ambiguous): not yet as a full run, but a tiny paid pilot is cheap enough to settle it. Tier 1 batch on the 300 pilot questions would cost well under a cent. The case for it: clarifying is worth a lot on real data (Tier 2 gains 30 points on ambiguous pilot questions), and the free detector only reaches 0.65. The case against: the synthetic oracle headroom is only 3 to 5 points even with perfect detection, and in a live router the probe is an extra call on every question, about 0.07 u2 at Tier 1 standard prices, which is more than the whole budget at the tightest settings. Worth running if it beats the free detector clearly, say AUROC above 0.72 with the CI's lower end above 0.65 on the same pilot questions, compared against the detector's out-of-fold scores there. Otherwise drop it. It needs its own prompt module and cache file, so prompts.py and the existing caches stay untouched. Not run.

## Frozen experiment design (2026-10-19)

75. configs/experiment.yaml is frozen as of today, before any real cache has been evaluated. Its sha256, taken over LF-normalized bytes so a Windows checkout hashes the same as Linux, is

    637c056365be91a25d07155b63dcfe6ec2ac226afea2b970b229a0a3a2b427e5

    It is recorded in configs/experiment.sha256. It fixes the budget grid (0.02 to 1.0 u2 per question), stream length K = 100, evaluation stream seeds 0, 1 and 2 (100 streams each), tuning and fitting seeds, clarify fees of 0.5, 1 and 2 times the mean Tier 2 call cost, the Tier 0 nominal price sweep (0, 0.02/0.08 and 0.05/0.20), the DP grid and bins, and the fixed-threshold calibration rates.

76. The rule for the DP's number of text bins is frozen with it: candidates 1, 2 and 3, scored by mean DP accuracy over every tuning budget on the tune streams, starting from the fewest bins and moving up only when a setting beats the current choice by at least 0.5 accuracy points. It lives in design.choose_text_bins and has unit tests. Test is never consulted. Under this rule the earlier synthetic run would still have kept 3 bins, since its margin was 0.53 points.

77. evaluate.py prints the design hash in its header and writes it into every CSV, run_info.json and the README results table. On real data it refuses to run if the design doesn't match the lock, or if --quick overrides are in play. Synthetic runs still work with any design but carry "+quick" or "+unfrozen" in the stamp. Changing the design after this point needs a new lock and a DECISIONS entry saying why, written before the real evaluation.

78. evaluate.py checks cache completeness before fitting anything (completeness.py). On real data it refuses when any tier has a pid from prompts.jsonl with no row, a pid whose latest row is an error, or a good row with a prompt_version other than v1. It also reports rows for unknown pids and errors that a retry fixed. One interpretation: the brief said "has error rows", and I refuse only on unresolved errors. cache_api is append only, so a transient rate-limit row stays in the file after the retry succeeds, and refusing on it would block the real run for good after the runbook's own "just rerun it" advice. Run today on the real pilot-only cache, it refuses with 17,531 missing pids per tier, as it should.

## Synthetic generator recalibrated to the real pilot (2026-10-19)

79. The synthetic world is now a rehearsal of the real data, calibrated to the 300-question pilot. Measured on the synthetic calib split against the pilot (pilot in brackets): clear raw accuracy 0.19, 0.57, 0.65 (0.18, 0.57, 0.65), raw ambiguous 0.07, 0.25, 0.28 (0.06, 0.26, 0.27), clarified ambiguous 0.12, 0.42, 0.59 (0.11, 0.42, 0.58). Tier 1 and 2 both right on 39 percent of raw questions (42), Tier 2 right where Tier 1 is wrong 8 percent (6), the reverse 3 percent (1). Own-correctness AUROC from confidence about 0.69 to 0.70 on every tier (pilot 0.64 to 0.75). Tier 0 features predicting Tier 1 or Tier 2 correctness 0.51 to 0.52 (pilot 0.42 to 0.56). Ambiguity from any model signal 0.44 to 0.50 (pilot 0.41 to 0.55). Text detector 0.64 to 0.65 on synthetic tune (real 0.65). Full Tier 0 sample agreement on 39 percent of questions (44). The world is also real-sized now: 10,036 train and 2,002 validation questions.

80. How the latent model changed to get there: skill = a - L z - d is_ambiguous + N e per tier. Tier 1 and 2 load fully on the shared difficulty z, have little own noise (0.15), and share one success draw and one reading draw, which is what makes them agree as much as on the pilot. Tier 0 loads only 0.1 on z with its own draw, so its confidence says nothing about the paid tiers. d is solved per tier so clarified ambiguous accuracy matches, and the raw-prompt reading probability is the pilot's raw over clarified ratio. Confidence ignores the ambiguity penalty for Tiers 0 and 1 and half of it for Tier 2, matching the pilot, where only Tier 2 was slightly less sure on ambiguous questions. Self consistency samples no longer split across readings. Template ambiguity rates were narrowed (spread 0.9 around 0.475) so the text detector lands near 0.65. One honest gap: Tier 0 sc_agree predicts its own correctness at 0.60 against 0.68 on the pilot.

81. Result on the recalibrated synthetic test streams: the DP beats the tuned threshold by 12.6 points at 0.35 u2 (paired CI 12.1 to 13.1), more than the old world's 6.2, not less. Most of that is not budget-awareness. The threshold family is two-model (Tier 0 or Tier 2), as ReDAct is, and on pilot-shaped data Tier 1 is nearly as accurate as Tier 2 at a twentieth of the price. The fair comparison is against the best simple policy that can use Tier 1, which is within about 1 point of the DP at every budget (DP minus best simple, from 0.02 to 1 u2: -1.0, -0.1, +0.4, +1.2, +0.8, +0.2, 0.0, +0.7 points). That's the expected "much smaller" gain. The README now leads with it. A three-tier tuned threshold would be the fairer ReDAct-style baseline, and I haven't added one.

82. Under the frozen text-bin rule every config kept one bin. Tune scores for 1, 2 and 3 bins were within 0.4 points, so the free text score is unused by the DP on this world.

83. Bug found and fixed while rebuilding: the incremental embedding cache keyed vectors by qid only, and the real-sized synthetic world reuses qid names from the old one with different question text, so 4,000 questions got stale vectors. Vectors are now keyed by qid plus a fingerprint of the text, and a changed question is embedded again. On real data the qids and text are frozen, so it couldn't have bitten there.

84. Process slip, owned and undone: while refreshing the real MiniLM cache to the new index format I ran `features --embed minilm` on the real data dir, which embeds every split and writes a feature table, so it read the 2,002 test questions' text and metadata and computed their embeddings. Nothing was fitted, tuned, graded or evaluated on them, and nothing read the outputs. I deleted the real feature table and rebuilt the MiniLM cache from calib and tune only (10,036 questions, 0 test qids, checked). To keep it from happening again, features.py now refuses to run on real data without --include-test, and only `scripts/bg.py real`, the final evaluation step, passes it.

85. First paid run of the ambiguity probe failed on all 300 pilot prompts with a 400 from the API ("max_tokens or model output limit was reached"), because I had capped its reply at 2 tokens. Nothing was charged. The cap is now 16 (the answer calls use 24 and work), which raises the worst case from $0.0034 to $0.0055 at sync prices. The 300 error rows stay in `probe_tier1.jsonl`, and a successful retry supersedes them. A smoke test with a fake client couldn't catch this, since it doesn't enforce an API minimum.

86. Ambiguity probe pilot result (2026-10-19): DROP. Tier 1 asked whether each of the 300 pilot questions is ambiguous (prompt amb-v1, 128 ambiguous) scored AUROC 0.594 [0.539, 0.648]. The free text detector, out of fold on the same questions, scored 0.691 [0.629, 0.751], and the paired difference is -0.096 [-0.173, -0.020], so the probe is worse and the interval excludes zero. The keep rule needed at least 0.72 with the lower CI end above 0.65, and the probe is far from both. It cost $0.0030, and total recorded spend is now $0.0947. The probe is not used anywhere, the full run is unchanged, and it won't be run on more questions. I'm also not rewording the prompt and rerunning on the same 300 questions, since that would be tuning against the pilot on which the keep rule was set. A different prompt or model would be a new probe with a new pre-set rule. The text detector's 0.691 on the pilot sits above its 0.648 on the 2,917 tune questions, which is within what 300 questions can do, and the larger tune figure is the better estimate of it.
