"""Cache completeness check, run by evaluate.py before anything is fitted.

For each tier cache it checks, against every pid in prompts.jsonl:
  missing         a pid with no row at all
  unresolved      a pid whose latest row is an error, so the call never succeeded
  prompt_version  a good row whose prompt_version is not prompts.PROMPT_VERSION (v1)
  unknown         rows for pids that are not in prompts.jsonl (reported, not fatal)
  retried         pids with an error row followed by a good retry (reported, not fatal)

An error row that a later retry fixed is not a problem. cache_api is append only, so the failed
attempt stays in the file forever, and refusing on it would block the run after a single transient
rate limit. Only errors that were never retried successfully block evaluation.

On real data any missing, unresolved or wrong-version row stops evaluate.py.
"""
from __future__ import annotations

from pathlib import Path

from .common import iter_jsonl, iter_jsonl_where, only_qids, only_splits, read_qids
from .grade import TIERS
from .prompts import PROMPT_VERSION


def check_tier(cache: Path, expected: set[str], qids: set[str] | None = None) -> dict:
    latest: dict[str, dict] = {}
    had_error: set[str] = set()
    rows = iter_jsonl(cache) if qids is None else iter_jsonl_where(cache, only_qids(qids))
    for r in rows:
        pid = r.get("pid")
        if not pid:
            continue
        if r.get("error"):
            had_error.add(pid)
        latest[pid] = r
    good = {p for p, r in latest.items() if not r.get("error")}
    wrong_version = sorted(p for p in good if latest[p].get("prompt_version") != PROMPT_VERSION)
    return {
        "rows": len(latest),
        "missing": sorted(expected - set(latest)),
        "unresolved": sorted(p for p in expected & set(latest) if latest[p].get("error")),
        "prompt_version": wrong_version,
        "unknown": sorted(set(latest) - expected),
        "retried": sorted(had_error & good),
    }


def check_cache(data_dir: Path, splits: tuple[str, ...] | None = None) -> tuple[list[str], dict]:
    """Problems that must block a real evaluation, and the full per-tier report.

    With splits, only those questions are checked, and no other row is decoded.
    """
    data_dir = Path(data_dir)
    prompts = data_dir / "splits" / "prompts.jsonl"
    if splits is None:
        expected, qids = {p["pid"] for p in iter_jsonl(prompts)}, None
    else:
        expected = {p["pid"] for p in iter_jsonl_where(prompts, only_splits(splits))}
        qids = {q for s in splits for q in read_qids(data_dir / "splits" / f"{s}.jsonl")}
    problems, report = [], {}
    for tier in TIERS:
        cache = data_dir / "cache" / f"{tier}.jsonl"
        if not cache.exists():
            problems.append(f"{tier}: no cache file at {cache}")
            report[tier] = None
            continue
        r = check_tier(cache, expected, qids)
        report[tier] = {k: (len(v) if isinstance(v, list) else v) for k, v in r.items()}
        for key, what in (("missing", "pids with no row"), ("unresolved", "pids whose latest row is an error"),
                          ("prompt_version", f"good rows with prompt_version other than {PROMPT_VERSION}")):
            if r[key]:
                problems.append(f"{tier}: {len(r[key])} {what}, for example {r[key][:3]}")
    return problems, report


def describe(report: dict) -> str:
    lines = []
    for tier, r in report.items():
        if r is None:
            lines.append(f"  {tier}: no cache")
            continue
        lines.append(f"  {tier}: {r['rows']} pids cached, {r['missing']} missing, {r['unresolved']} unresolved errors, "
                     f"{r['prompt_version']} wrong prompt version, {r['retried']} fixed by a retry, "
                     f"{r['unknown']} not in prompts.jsonl")
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse

    from .common import DATA

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    args = ap.parse_args(argv)
    problems, report = check_cache(args.data_dir)
    print(describe(report))
    if problems:
        print("NOT COMPLETE:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print("COMPLETE: every pid in prompts.jsonl has a good v1 row in every tier cache")


if __name__ == "__main__":
    main()
