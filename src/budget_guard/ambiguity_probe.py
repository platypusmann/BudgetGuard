"""Optional paid LLM ambiguity probe: ask Tier 1 whether a question is ambiguous.

Kept fully separate from answering, so nothing frozen moves:
  prompt   its own SYSTEM and user message here, never prompts.py. PROBE_VERSION is in every row
  cache    data/cache/probe_<tier>.jsonl, never the answer caches
  pids     <qid>::ambprobe, so probe rows can never be mistaken for answers

The model is asked for one word, Yes or No, with logprobs on. p_yes is the probability mass on
"yes" over "yes" plus "no" among the first token's top logprobs, so it is a graded score rather
than a hard label. It is None when neither word appears there.

Run through cache_api with --task ambiguity, pilot only for now (RUNBOOK, optional step):
  python scripts/bg.py cache estimate --task ambiguity --tier tier1 --pilot
  python scripts/bg.py cache sync --task ambiguity --tier tier1 --pilot --cap 0.01

Then compare against the free text detector on the same pilot questions:
  python scripts/bg.py probe-compare
Keep rule: probe AUROC at least 0.72 and the lower end of its 95 percent CI above 0.65.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

PROBE_VERSION = "amb-v1"
TASK = "ambiguity"
MAX_OUT = 2  # "Yes" or "No" is one token, one spare
KEEP_AUROC, KEEP_CI_LO = 0.72, 0.65

SYSTEM = (
    "You judge whether a trivia question is ambiguous. A question is ambiguous when reasonable "
    "readers could mean different things by it that have different correct answers, for example "
    "a different year, season, version, event, place or person. "
    "Reply with Yes or No only."
)


def user_message(question: str) -> str:
    return f"Question: {question.strip()}\nIs this question ambiguous?"


def messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(question)}]


def probe_pid(qid: str) -> str:
    return f"{qid}::ambprobe"


def est_in_tokens(text: str) -> int:
    """Same deliberately high estimate cache_api uses for answers."""
    return math.ceil((len(SYSTEM) + len(user_message(text))) / 4) + 16


def request_body(cfg: dict, text: str) -> dict:
    body = {"model": cfg["model"], "messages": messages(text), "max_completion_tokens": MAX_OUT,
            "reasoning_effort": cfg["reasoning_effort"]}
    if cfg["reasoning_effort"] == "none":
        body.update(temperature=0, logprobs=True, top_logprobs=cfg.get("top_logprobs", 5))
    return body


def p_yes(content: list[dict] | None) -> float | None:
    """P(yes) renormalized over yes and no, from the first token's top logprobs."""
    if not content:
        return None
    yes = no = 0.0
    for t in content[0].get("top_logprobs") or [{"token": content[0]["token"], "logprob": content[0]["logprob"]}]:
        word = t["token"].strip().lower()
        if word.startswith("yes"):
            yes += math.exp(t["logprob"])
        elif word.startswith("no"):
            no += math.exp(t["logprob"])
    return yes / (yes + no) if yes + no > 0 else None


def probe_row(tier: str, cfg: dict, p: dict, body: dict, batch: bool) -> dict:
    from .cache_api import actual_cost, now

    choice = body["choices"][0]
    usage = body.get("usage") or {}
    content = (choice.get("logprobs") or {}).get("content")
    return {
        "pid": p["pid"], "qid": p["qid"], "split": p["split"], "tier": tier, "model": cfg["model"],
        "model_returned": body.get("model"), "probe_version": PROBE_VERSION, "mode": "batch" if batch else "sync",
        "answer": (choice["message"].get("content") or "").strip(), "p_yes": p_yes(content),
        "finish_reason": choice.get("finish_reason", ""),
        "n_in": usage.get("prompt_tokens"), "n_out": usage.get("completion_tokens"),
        "usd": round(actual_cost(cfg, usage, batch), 8), "ts": now(), "error": None,
    }


# ---------------------------------------------------------------- comparison

def compare(data_dir: Path, out: Path, backend: str = "minilm", tier: str = "tier1", n_boot: int = 2000):
    """Probe against the free text detector on the same calib pilot questions. No test rows read.

    The text detector's scores are out of fold on calib (5 folds), so a pilot question is never
    scored by a model that trained on it, and the probe needs no training at all.
    """
    import pandas as pd

    from .ambiguity_text import TextAmbiguityDetector
    from .common import iter_jsonl
    from .features import build_embeddings, load_embeddings
    from .grade import latest_good_rows
    from .leakage import assert_no_test, verify_and_test_qids
    from .pilot_check import auroc, boot_auroc
    from .synthetic import is_synthetic

    data_dir, out = Path(data_dir), Path(out)
    test_qids = verify_and_test_qids(data_dir / "splits")
    calib = list(iter_jsonl(data_dir / "splits" / "calib.jsonl"))
    pilot = {r["qid"] for r in iter_jsonl(data_dir / "splits" / "pilot.jsonl")}
    assert pilot <= {r["qid"] for r in calib}, "pilot must come from calib"
    assert_no_test([r["qid"] for r in calib], test_qids, "calib")

    cache = data_dir / "cache" / f"probe_{tier}.jsonl"
    probe = {r["qid"]: r for r in latest_good_rows(cache)} if cache.exists() else {}
    if not probe:
        raise SystemExit(f"no probe rows in {cache}. Run the optional probe step in RUNBOOK.md first.")
    versions = {r.get("probe_version") for r in probe.values()}
    assert versions == {PROBE_VERSION}, f"probe rows from another prompt version: {versions}"

    build_embeddings(data_dir, backend, splits=("calib",))
    emb = load_embeddings(data_dir, backend)
    oof = TextAmbiguityDetector("combined").out_of_fold(
        [r["question"] for r in calib], np.vstack([emb[r["qid"]] for r in calib]),
        np.array([r["is_ambiguous"] for r in calib], bool))
    text = {r["qid"]: s for r, s in zip(calib, oof)}

    qs = [q for q in sorted(pilot) if q in probe and probe[q].get("p_yes") is not None]
    no_score = len([q for q in pilot if q in probe]) - len(qs)
    amb = {r["qid"]: r["is_ambiguous"] for r in calib}
    y = np.array([amb[q] for q in qs], bool)
    xp = np.array([probe[q]["p_yes"] for q in qs], float)
    xt = np.array([text[q] for q in qs], float)
    rows = []
    for name, x, seed in (("llm_probe", xp, 11), ("text_detector_oof", xt, 12)):
        lo, hi = boot_auroc(x, y, n_boot, seed)
        rows.append({"score": name, "n": len(qs), "n_ambiguous": int(y.sum()), "auroc": auroc(x, y),
                     "ci_lo": lo, "ci_hi": hi})
    # paired difference, the same resampled questions for both scores
    rng = np.random.default_rng(13)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(qs), len(qs))
        a, b = auroc(xp[i], y[i]), auroc(xt[i], y[i])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    d_lo, d_hi = (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))) if diffs else (np.nan, np.nan)
    pr = rows[0]
    keep = pr["auroc"] >= KEEP_AUROC and pr["ci_lo"] > KEEP_CI_LO
    rows.append({"score": "probe_minus_text", "n": len(qs), "n_ambiguous": int(y.sum()),
                 "auroc": pr["auroc"] - rows[1]["auroc"], "ci_lo": d_lo, "ci_hi": d_hi})
    df = pd.DataFrame(rows)
    df.insert(0, "data", "SYNTHETIC" if is_synthetic(data_dir) else "REAL")
    df["decision"] = "KEEP" if keep else "DROP"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ambiguity_probe_vs_text.csv"
    df.to_csv(path, index=False)
    spent = sum(r.get("usd") or 0 for r in probe.values())
    print(f"\nLLM ambiguity probe ({tier}, {PROBE_VERSION}) against the free text detector, "
          f"{len(qs)} calib pilot questions ({int(y.sum())} ambiguous), {df['data'].iloc[0]} data")
    if no_score:
        print(f"  {no_score} probe rows had neither Yes nor No in the top logprobs and were left out")
    for r in rows:
        label = {"llm_probe": "LLM probe", "text_detector_oof": "text detector (out of fold)",
                 "probe_minus_text": "probe minus text"}[r["score"]]
        print(f"  {label:28s} {r['auroc']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
    print(f"  keep rule: AUROC >= {KEEP_AUROC} and CI lower end > {KEEP_CI_LO}  ->  {'KEEP' if keep else 'DROP'}")
    print(f"  probe spend recorded in its rows: ${spent:.4f}")
    print(f"wrote {path}")
    return df


def main(argv=None) -> None:
    from .common import DATA, ROOT

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["compare"])
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--embed", choices=["minilm", "hash"], default="minilm")
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    compare(args.data_dir, args.out, args.embed, args.tier, args.n_boot)


if __name__ == "__main__":
    main()
