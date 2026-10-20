"""Print a random sample of graded pilot answers per tier for a manual check. Spends nothing.

  python scripts/hand_check.py              # 25 per tier from data/
  python scripts/hand_check.py --n 40 --data-dir data/synthetic

For each row, decide yourself whether the answer is right, then compare with the em and lenient
columns. Also prints the health numbers the runbook asks you to look at.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from budget_guard.common import DATA, iter_jsonl, iter_jsonl_where, only_splits  # noqa: E402
from budget_guard.grade import TIERS, latest_good_rows, load_graded  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import pandas as pd

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    # the pilot is drawn from calib, so nothing else is read, and other splits' prompts are never decoded
    rows = {r["qid"]: r for r in iter_jsonl(args.data_dir / "splits" / "calib.jsonl")}
    pilot = {p["pid"]: p["text"] for p in iter_jsonl_where(args.data_dir / "splits" / "prompts.jsonl", only_splits(["calib"]))
             if p["pilot"]}
    for tier in TIERS:
        cache = args.data_dir / "cache" / f"{tier}.jsonl"
        if not cache.exists():
            print(f"{tier}: no cache yet")
            continue
        raw = pd.DataFrame([{k: r.get(k) for k in ("pid", "answer", "finish_reason", "lp_mean", "model_returned",
                                                    "prompt_version", "usd")}
                            for r in latest_good_rows(cache, {p.split("::")[0] for p in pilot})
                            if r["pid"] in pilot])
        if raw.empty:
            print(f"{tier}: no pilot rows")
            continue
        g = load_graded(tier, args.data_dir)[["pid", "em", "lenient", "f1"]]
        d = raw.merge(g, on="pid", how="left")
        d["prompt"] = d["pid"].map(pilot)
        d["gold"] = d["pid"].str.split("::").str[0].map(lambda q: " | ".join(rows[q]["gold"][:3]))
        print(f"\n===== {tier}: {len(d)} pilot rows")
        print(f"models returned: {sorted(d['model_returned'].dropna().unique())}   prompt versions: "
              f"{sorted(d['prompt_version'].dropna().unique())}")
        print(f"hit max tokens: {(d['finish_reason'] == 'length').mean():.1%}   logprobs missing: "
              f"{d['lp_mean'].isna().mean():.1%}   em {d['em'].mean():.3f}   lenient {d['lenient'].mean():.3f}   "
              f"spent ${d['usd'].sum():.4f}")
        sample = d.sample(min(args.n, len(d)), random_state=args.seed)
        print(sample[["prompt", "gold", "answer", "em", "lenient"]].to_string(index=False))


if __name__ == "__main__":
    main()
