# Runbook for the real run

Your exact manual steps, in order. The total worst case is about $3.65, a pilot under $0.20 plus a batch run of about $3.45. Tier 0 is free on Kaggle. Nothing here runs by itself, and every paid command prints its worst case and refuses to pass its cap.

Run everything from the repo root. Commands are PowerShell. Always go through `python scripts/bg.py`, which puts `src` on the Python path for you. Calling `python -m budget_guard...` directly fails with ModuleNotFoundError unless you set `$env:PYTHONPATH = "src"` first.

## 0. One-time setup

1. `pip install -r requirements.txt`
2. `python scripts/bg.py test` and check that everything passes.
3. In the OpenAI dashboard, set a hard monthly spend limit of $10 on the project you'll use. That's the backstop behind the per-command caps.
4. Put the key in your shell session only, never in a file:
   `$env:OPENAI_API_KEY = "sk-..."`
   `.env.example` is only a reminder of the variable name. The code doesn't read `.env`.

## 1. Build the frozen splits (free, needs internet once)

```
python scripts/bg.py splits
```

This downloads AmbigNQ light from Hugging Face into `data/raw/` and writes `data/splits/`. Check the printed stats: `n_test` should be 2002, `n_pilot` 300, and `n_prompts` around 17,000 to 18,000. Then commit the splits so they're frozen:

```
git add data/splits
git commit -m "Freeze AmbigNQ splits"
```

Never pass `--force` after this point. `split.verify()` will refuse to tune or evaluate if a split file changes.

## 2. Upload prompts to Kaggle

1. On Kaggle, Datasets, New Dataset, upload `data/splits/prompts.jsonl` and name it `budget-guard-prompts`. Private is fine.
2. Create a notebook from `notebooks/kaggle_tier0.py` (paste the cells). Settings: Accelerator GPU T4 x2, Internet on, Persistence Files only. Add `budget-guard-prompts` as an input.
3. Optional: add a Kaggle secret `HF_TOKEN` and set `HF_MIRROR_REPO` to a private dataset name, so checkpoints survive a crashed session.

## 3. Tier 0 pilot on Kaggle (free)

1. Leave `PILOT_ONLY = True` in cell 1 and run all cells, using Save Version, then Save and Run All (Commit).
2. Cell 6 prints sample answers, the share hitting max tokens and mean self consistency. Max-token hits should be a few percent at most.
3. From the version's Output tab, download `tier0.jsonl` to `data/cache/tier0.jsonl`.

## 4. Estimate the pilot (free)

```
python scripts/bg.py cache estimate --tier tier1 --pilot
python scripts/bg.py cache estimate --tier tier2 --pilot
```

Expect about 450 prompts each, with a sync worst case of about $0.01 for Tier 1 and $0.17 for Tier 2. If either is far off, stop and check `prompts.jsonl`.

## 5. Pilot sync (about $0.18 worst case)

```
python scripts/bg.py cache sync --tier tier1 --pilot --cap 0.05
python scripts/bg.py cache sync --tier tier2 --pilot --cap 0.25
```

If a command dies or some calls fail, rerun the same line. It only calls what's missing and never pays twice. `data/cache/ledger.jsonl` keeps the running spend.

## 6. Hand check (free)

```
python scripts/bg.py grade
python scripts/hand_check.py --n 30
```

For each tier, look at:

- `models returned` shows the model you expect (gpt-6-luna and gpt-6-sol snapshots, Qwen for Tier 0), and the prompt version is v1.
- `logprobs missing` is 0 percent for Tier 1 and 2. If it isn't, reasoning effort none wasn't honoured, so stop and fix that before the full run.
- `hit max tokens` is low. Answers should be short phrases.
- Read the 30 sampled rows and mark each one right or wrong yourself. Your call should agree with the lenient column on at least 27 of 30. If lenient is clearly too strict or too loose, write down what went wrong in DECISIONS.md before going on. The primary metric is pre-registered, so don't switch it, but the mismatch belongs in the writeup.
- Rough accuracy on the pilot should land near 20 to 35 percent for Tier 0 and noticeably higher for Tier 2. Big surprises usually mean a prompt or parsing problem.

## 7. Full Tier 0 run on Kaggle (free)

1. Set `PILOT_ONLY = False` and commit the notebook again. To keep the pilot rows, attach the previous version's output as an input, and `restore_prior` merges it.
2. It's about 17,700 prompts with 5 samples each. If one session isn't enough (the notebook stops itself at 11 hours), commit again with the last output attached and it resumes. You can also split the work with `NUM_SHARDS = 2` and `SHARD_ID = 0` or `1` in two notebooks, then concatenate the two `tier0.jsonl` files.
3. Download the final `tier0.jsonl` to `data/cache/tier0.jsonl`, replacing the pilot file. It already contains the pilot rows.

## 8. Batch Tier 1 and Tier 2 (about $3.45 worst case)

```
python scripts/bg.py cache estimate --tier tier1
python scripts/bg.py cache estimate --tier tier2
```

Check the batch worst case: about $0.17 for Tier 1 and $3.3 for Tier 2. Pilot prompts are already cached, so they're skipped. Then:

```
python scripts/bg.py cache batch-submit --tier tier1 --cap 0.5
python scripts/bg.py cache batch-submit --tier tier2 --cap 4.0
```

Batches finish within 24 hours. Check on them, as often as you like, with

```
python scripts/bg.py cache batch-collect --tier tier1
python scripts/bg.py cache batch-collect --tier tier2
```

until both say "nothing to collect". Then run `estimate` again. Anything still missing (a failed line or an expired batch) shows up there. Submit again with the same `batch-submit` line, and only the missing prompts are sent.

## 9. Evaluate on the real cache (free, offline)

```
python scripts/bg.py real
```

This grades all three tiers, builds the feature table, fits on calib, tunes on tune, and evaluates once on test. It writes `results/` and `results/figures/` and takes a few minutes. For the RouteLLM baseline with MiniLM instead of the hashing embedding, use `python scripts/bg.py real --minilm` (it downloads the model once). Then:

1. Check that `results/run_info.json` says `"data": "REAL"` and that the figures no longer carry the SYNTHETIC stamp.
2. Copy `results/results_table.md` into the README results section, swap the synthetic bullet points for what the real run shows, and drop the SYNTHETIC wording.
3. Robustness check on EM: `python scripts/bg.py evaluate --metric em --out results/em`.
4. Commit `data/cache`, `results/` and the README. Anyone can then reproduce every number for $0.

Test is touched exactly once, in step 9. If you change anything after looking at test numbers, say so in DECISIONS.md.
