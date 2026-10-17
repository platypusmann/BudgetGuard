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
