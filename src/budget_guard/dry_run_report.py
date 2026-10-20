"""Dry run report from an `evaluate --eval-split tune` output folder. Debugging only.

Tune is in sample for every threshold that was tuned on it, so these numbers are optimistic for the
tuned policies and are not results. Nothing here goes in the README.

  python scripts/bg.py dry-run-report --out results/dev_tune
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import ROOT

SIMPLE = ["always_tier1", "always_tier2", "cascade_frugal", "threshold_tuned", "routellm"]
AWARE = ["dp_oracle", "slack_schedule", "learned_router"]


def flags(s: pd.DataFrame, amb_rate: float, schedule: pd.DataFrame | None) -> list[str]:
    """Rule-based checks for policies that behave strangely. Each line says what tripped."""
    out = []
    m = s[s["masked"]]
    h = m[m["headline"]]
    for r in m[m["violation_rate"] > 0].itertuples():
        out.append(f"ERROR {r.policy} broke the cap under masking at {r.budget_u2_per_q:g} u2 (violation {r.violation_rate:.3f})")
    acc = h.pivot(index="budget_u2_per_q", columns="policy", values="accuracy")
    skip = h.pivot(index="budget_u2_per_q", columns="policy", values="skip_rate")
    others = acc.drop(columns=["hindsight_oracle"], errors="ignore")
    if "hindsight_oracle" in acc:
        bad = acc["hindsight_oracle"] < others.max(axis=1) - 1e-9
        for b in acc.index[bad]:
            out.append(f"ERROR hindsight oracle is not the best policy at {b:g} u2")
    for p in AWARE:
        if p == "dp_oracle" or p not in acc:
            continue
        for b in acc.index:
            gap = acc.at[b, p] - acc.at[b, "dp_oracle"]
            if gap < -0.03:
                out.append(f"{p} is {-gap * 100:.1f} points below the DP it imitates at {b:g} u2 "
                           f"(skips {skip.at[b, p]:.0%} of questions, DP skips {skip.at[b, 'dp_oracle']:.0%})")
    for p in acc.columns:
        diffs = acc[p].diff().dropna()
        for b, d in diffs[diffs < -0.01].items():
            out.append(f"{p} loses {-d * 100:.1f} points when the budget grows to {b:g} u2")
    clar = h[(h["clarify_rate"] > 0.05) & (h["clarify_precision"] < amb_rate)]
    for r in clar.itertuples():
        if r.policy == "clarify_all":
            continue
        out.append(f"{r.policy} clarifies with precision {r.clarify_precision:.2f} at {r.budget_u2_per_q:g} u2, "
                   f"below the {amb_rate:.2f} base rate, so worse than clarifying at random")
    if "clarify_all" in acc and "always_tier0" in acc:
        low = acc.index[acc["clarify_all"] < acc["always_tier0"]]
        if len(low):
            out.append(f"reference clarify_all is below always Tier 0 at {', '.join(f'{b:g}' for b in low)} u2, where "
                       "one clarify fee is more than the budget per question, so clarify positions mean little there")
    if schedule is not None and "dp_decisions" in schedule:
        empty = schedule[schedule["dp_decisions"] == 0]
        for r in empty.itertuples():
            out.append(f"slack schedule bucket from {r.slack_from_u2:g} u2 had no DP decisions to fit, it borrows a neighbour")
    return out


def report(out: Path) -> str:
    out = Path(out)
    s = pd.read_csv(out / "summary.csv")
    info = json.loads((out / "run_info.json").read_text())
    rank = pd.read_csv(out / "metric_ranking.csv") if (out / "metric_ranking.csv").exists() else None
    sched = pd.read_csv(out / "schedule.csv") if (out / "schedule.csv").exists() else None
    m, u = s[s["masked"]], s[~s["masked"]]
    h = m[m["headline"]]
    acc = h.pivot(index="budget_u2_per_q", columns="policy", values="accuracy")

    lines = ["# Dry run report (debugging only)", "",
             "DRY RUN on the tune split. Tune is in sample for every threshold tuned on it, so these numbers "
             "are optimistic for the tuned policies and are not results. Keep them out of the README.", ""]
    lines += [f"- Evaluated split: {', '.join(sorted(s['eval_split'].unique()))}. Mode: {info.get('mode', 'unknown')}",
              f"- Experiment design sha256: {info['design_sha256']} ({'frozen' if info.get('design_frozen') else 'NOT frozen'})",
              f"- Pools: {info['pool_sizes']}. Dropped questions: {info['dropped']}", ""]
    lines += ["## Cache completeness (calib and tune)", ""]
    for tier, r in info.get("cache_completeness", {}).items():
        lines.append(f"- {tier}: {r['rows']} pids, {r['missing']} missing, {r['unresolved']} unresolved errors, "
                     f"{r['prompt_version']} wrong prompt version")
    ok = all(r and r["missing"] == 0 and r["unresolved"] == 0 and r["prompt_version"] == 0
             for r in info.get("cache_completeness", {}).values())
    lines += ["", f"Result: {'COMPLETE' if ok else 'NOT COMPLETE'}", ""]

    lines += ["## Violation rate per policy, worst over every config and budget", "",
              "| Policy | masked | without the guard |", "|---|---|---|"]
    for p in sorted(s["policy"].unique()):
        mv = m[m["policy"] == p]["violation_rate"].max()
        uv = u[u["policy"] == p]["violation_rate"].max() if (u["policy"] == p).any() else float("nan")
        lines.append(f"| {p} | {mv:.3f} | {'not run' if pd.isna(uv) else f'{uv:.3f}'} |")
    lines += ["", f"Masked policies all at 0: {'yes' if (m['violation_rate'] == 0).all() else 'NO'}", ""]

    lines += ["## Ranking check, lenient against lenient_plus", ""]
    rc = info.get("ranking_check")
    if rank is None or rc in (None, "skipped"):
        lines.append("Did not run.")
    else:
        ch = rank[rank["rank_changed"]]
        lines.append(f"Ran on {rank['budget_u2_per_q'].nunique()} budgets. {len(ch)} rank changes.")
        for r in ch.itertuples():
            lines.append(f"- {r.budget_u2_per_q:g} u2: {r.policy} {r.rank_lenient} to {r.rank_lenient_plus} "
                         f"(lenient {r.accuracy_lenient:.4f}, lenient_plus {r.accuracy_lenient_plus:.4f})")
    lines.append("")

    lines += ["## DP against the best simple policy that can use Tier 1", "",
              "| Budget u2 | DP | best simple | which | DP minus best, points |", "|---|---|---|---|---|"]
    simple = [c for c in SIMPLE if c in acc]
    for b in acc.index:
        best = acc.loc[b, simple].max()
        which = acc.loc[b, simple].idxmax()
        lines.append(f"| {b:g} | {acc.at[b, 'dp_oracle']:.3f} | {best:.3f} | {which} | {(acc.at[b, 'dp_oracle'] - best) * 100:+.1f} |")
    lines.append("")

    amb_rate = info.get("eval_ambiguous_rate")
    if amb_rate is None:   # runs from before the field existed: the dev table holds calib and tune only
        f = pd.read_parquet(Path(info["data_dir"]) / "dev" / "features" / "features.parquet",
                            columns=["variant", "split", "is_ambiguous"])
        amb_rate = float(f[(f["variant"] == "orig") & (f["split"] == "tune")]["is_ambiguous"].mean())
    f = flags(s, amb_rate, sched)
    lines += ["## Policies that behave strangely (rule-based flags)", ""]
    lines += [f"- {x}" for x in f] if f else ["None tripped."]
    lines.append("")
    text = "\n".join(lines)
    (out / "DRY_RUN_REPORT.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "dev_tune")
    args = ap.parse_args(argv)
    print(report(args.out))


if __name__ == "__main__":
    main()
