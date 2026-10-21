# Runbook for the real run

One checklist, in order. Tick each box as you go. Every command runs from `C:\BudgetGuard` in Git Bash, and every one is free unless its heading gives a price. Paid commands print their worst case first and refuse to go past their cap.

Total worst case for the whole real run is about $3.65: the pilot ($0.09 actually spent), an optional ambiguity probe (about $0.002) and the batch run (about $3.45). Tier 0 on Kaggle is free.

Always go through `python scripts/bg.py`, which puts `src` on the Python path. Calling `python -m budget_guard...` directly fails with ModuleNotFoundError.

## A. Done already (as of 2026-10-19)

- [x] `pip install -r requirements.txt`, and the OpenAI project has a $10 hard monthly limit
- [x] Splits built and frozen: `python scripts/bg.py splits`, then committed (`Freeze AmbigNQ splits`). 17,959 prompts, 2,002 test questions, 300 pilot questions
- [x] `prompts.jsonl` uploaded to Kaggle as the `budget-guard-prompts` dataset
- [x] Kaggle Tier 0 pilot run, `tier0.jsonl` saved as `data/cache/tier0.jsonl` (428 rows)
- [x] Pilot sync for Tier 1 and 2, $0.0917 spent
- [x] Pilot hand check and pilot signal checks (`python scripts/bg.py pilot-check`)
- [x] Experiment design frozen (`configs/experiment.sha256`, DECISIONS.md 75)

## B. Before the full run (free, except the optional probe)

- [x] Write your own hand count into DECISIONS.md entry 63 (Tier 0, 1 and 2, out of 30 each). `python scripts/hand_check.py --n 30` prints the rows again.

- [x] Run the tests. Everything should pass, including the check that the experiment design still matches its lock.
  ```
  python scripts/bg.py test
  ```

- [x] Optional, about $0.002: the LLM ambiguity probe on the 300 pilot questions. It has its own prompt and its own cache file (`data/cache/probe_tier1.jsonl`), so nothing frozen changes. Set the key for this terminal only, then:
  ```
  read -s -p "Paste key: " OPENAI_API_KEY; echo; export OPENAI_API_KEY
  python scripts/bg.py cache estimate --task ambiguity --tier tier1 --pilot
  python scripts/bg.py cache sync --task ambiguity --tier tier1 --pilot --cap 0.01
  python scripts/bg.py probe-compare
  ```
  The estimate should say 300 prompts and about $0.0055 worst case. `probe-compare` prints KEEP or DROP (keep only if AUROC is at least 0.72 and the CI's lower end is above 0.65, against the free text detector on the same questions). Write the result into DECISIONS.md either way. A KEEP doesn't change the full run below, because using the probe in the router would be a design change, which needs its own decision first.

## C. Full Tier 0 run on Kaggle (free, a few GPU hours)

- [x] In the Kaggle notebook's cell 1, set `PILOT_ONLY = False`. Keep everything else as it is.
- [x] Add the pilot version's output as an input to the notebook (Add Input, then your notebook's output), so `restore_prior` merges the 428 pilot rows instead of recomputing them.
- [x] Check Session options: Accelerator GPU T4 x2, Internet on.
- [x] Save Version, then Save and Run All (Commit). The notebook stops itself cleanly at 11 hours. If it stops before finishing, commit again with the latest output attached, and it carries on from where it stopped.
- [x] When the version finishes, open the notebook's page, then the Output tab, download `tier0.jsonl`, and replace your pilot copy:
  ```
  mv ~/Downloads/tier0.jsonl data/cache/tier0.jsonl
  ```
- [x] Check it. The tier0 line must say 0 missing, 0 unresolved errors and 0 wrong prompt version. Tier 1 and 2 will still show missing rows at this point, so the overall verdict says NOT COMPLETE. That's expected.
  ```
  python scripts/bg.py completeness
  ```

## D. Batch Tier 1 and Tier 2 (about $3.45 worst case)

- [x] Set the key in this terminal if it isn't set yet:
  ```
  read -s -p "Paste key: " OPENAI_API_KEY; echo; export OPENAI_API_KEY
  ```
- [x] Estimate. Expect about 17,500 missing prompts per tier, with a batch worst case of about $0.17 for Tier 1 and $3.3 for Tier 2. If either is far off, stop and ask.
  ```
  python scripts/bg.py cache estimate --tier tier1
  python scripts/bg.py cache estimate --tier tier2
  ```
- [x] Submit both:
  ```
  python scripts/bg.py cache batch-submit --tier tier1 --cap 0.5
  python scripts/bg.py cache batch-submit --tier tier2 --cap 4.0
  ```
- [x] Collect. Batches finish within 24 hours, and you can run this as often as you like. Repeat until both commands say "nothing to collect".
  ```
  python scripts/bg.py cache batch-collect --tier tier1
  python scripts/bg.py cache batch-collect --tier tier2
  ```
- [x] Estimate again. Both tiers should say 0 prompts missing. If not, run the same `batch-submit` line again (only the missing prompts are sent), then collect again.
- [x] Check the whole cache. It must end with COMPLETE. If it lists unresolved errors, resubmit as in the previous box. If it lists a wrong prompt version, stop and ask.
  ```
  python scripts/bg.py completeness
  ```
- [x] Check the spend: `tail -5 data/cache/ledger.jsonl`. The total so far should be about $1.5 to $3.6.
- [x] Commit the cache, so anyone can reproduce every number for $0:
  ```
  git add data/cache
  git commit -m "Real cache: Tier 0, 1 and 2 for every prompt"
  ```

## E. Final evaluation (free, reads test exactly once)

This is the only step that reads the test split. It grades all three tiers, builds the feature table and MiniLM embeddings (now including test), fits on calib, tunes on tune, then evaluates once on test. It refuses to run if the cache isn't complete or the design doesn't match its lock.

- [x] Dry run on tune passed, calib and tune only, no test row decoded. Its report, `results/dev_tune/DRY_RUN_REPORT.md`, is debugging only and stays on this machine, because `results/dev_tune` isn't in the repo.
- [x] Make sure you're on the frozen code. This must print `pre-test-eval`, and `git status --short src scripts configs tests notebooks` must print nothing:
  ```
  git describe --tags --exact-match HEAD
  git status --short src scripts configs tests notebooks
  ```
- [x] Run it. It takes about 10 to 20 minutes on the laptop.
  ```
  python scripts/bg.py real --minilm
  ```
- [x] Check the header it printed. The first line must say `code commit <hash>, tag pre-test-eval, clean`. Then `experiment design sha256 637c0563... frozen design`, and every tier COMPLETE in the cache completeness lines.
- [x] Check `results/final/run_info.json` says `"data": "REAL"` and `"design_frozen": true`, and that the figures in `results/final/figures` don't carry the SYNTHETIC stamp.
- [x] Read the printed ranking check (lenient against lenient_plus). Any RANKING CHANGES line goes into the write-up.
- [x] Pre-registered robustness check on EM, into its own folder:
  ```
  python scripts/bg.py evaluate --metric em --embed minilm --out results/final_em
  ```

## F. Write-up

- [ ] Copy `results/final/results_table.md` into the README results section and replace the synthetic bullet points with what the real run shows. Lead with the DP against the best simple policy that can use Tier 1 (DECISIONS.md 81), not against the two-model thresholds.
- [ ] Add a DECISIONS.md entry with the real headline numbers and anything that surprised you. If you changed anything after seeing test numbers, say so there.
- [ ] Commit:
  ```
  git add results/final results/final_em README.md DECISIONS.md
  git commit -m "Real results"
  ```

## If something goes wrong

- A paid command dies partway or some calls fail: rerun the exact same line. It only sends what's missing, so nothing is paid twice.
- `ModuleNotFoundError: budget_guard`: you called `python -m ...` directly. Use `python scripts/bg.py ...` instead.
- `features on real data reads the test split`: you ran `features` by hand. Only step E may do that, through `python scripts/bg.py real`.
- `refusing to evaluate an incomplete real cache`: run `python scripts/bg.py completeness` and fix what it lists before trying again.
- `refusing to evaluate real data on anything but the frozen design`: `configs/experiment.yaml` changed after it was frozen. Put it back (`git checkout configs/experiment.yaml`) unless you've decided on a design change and written it up first.
