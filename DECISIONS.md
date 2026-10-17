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

13. Logprob confidence is a noisy, overconfident view of the same latent skill, and it stays high when the model confidently answers the wrong reading. Tier 0 self consistency samples split across readings on ambiguous raw prompts. The resulting signals are informative but far from perfect: AUROC about 0.68 for Tier 0 correctness from p_first, and about 0.67 for ambiguity from 1 - sc_agree. That's roughly what published work reports for real uncertainty signals, and it keeps the routing problem from being trivial.

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
