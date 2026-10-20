"""Pilot signal check on the calib pilot rows only. Spends nothing, reads no test rows.

Answers two questions before the full run, with honest uncertainty:
  1. Do the uncertainty features carry signal? AUROC with bootstrap 95 percent CIs for each
     feature predicting (a) whether that tier's raw answer is correct and (b) is_ambiguous.
     Tier 0: sc_agree, sc_distinct, lp_mean, lp_min, p_first, ent_first_top.
     Tiers 1 and 2: lp_mean, lp_min, p_first, ent_first_top.
     AUROC is reported as is. Below 0.5 means the feature runs the other way (entropy should).
  2. Does clarifying help? Per tier, on ambiguous pilot questions, the mean accuracy of the
     clarified prompt minus the raw prompt, with a paired bootstrap CI over questions.
  4. Risk-coverage for Tier 0 (results/pilot_risk_coverage.csv). Tier 0 answers are sorted by
     lp_min and separately by sc_agree, and the most confident 5, 10, 20 and 40 percent are kept.
     Each slice's accuracy is reported with a bootstrap CI, next to Tier 1 and Tier 2 overall and
     on the same questions, because a confident Tier 0 slice is also an easier slice.
  5. Is Tier 0 a useful probe for the other tiers (results/tier0_as_probe.csv)? AUROC of Tier 0's
     lp_min, lp_mean, sc_agree and sc_distinct for predicting, on the raw prompt, (a) Tier 1 is
     right, (b) Tier 2 is right and (c) Tier 2 is right while Tier 1 is wrong, the event that
     makes paying for Tier 2 over Tier 1 worth it.
  3. Does cross-tier agreement on the raw prompt flag ambiguity? agree_12 is whether the Tier 1
     and Tier 2 answers match after SQuAD normalization, agree_0_12 whether the Tier 0 greedy
     answer matches either. Disagreement should mean ambiguity, so AUROC below 0.5 is the
     expected direction. Note agree_12 needs both paid tiers, so it is not a free signal.

The pilot is 300 questions (about 130 ambiguous), so CIs are wide. Treat this as a sanity
check, not a result.

  python scripts/bg.py pilot-check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import DATA, ROOT, iter_jsonl
from .grade import TIERS, latest_good_rows, load_graded
from .leakage import assert_no_test, verify_and_test_qids
from .synthetic import is_synthetic

FEATURES = {
    "tier0": ["sc_agree", "sc_distinct", "lp_mean", "lp_min", "p_first", "ent_first_top"],
    "tier1": ["lp_mean", "lp_min", "p_first", "ent_first_top"],
    "tier2": ["lp_mean", "lp_min", "p_first", "ent_first_top"],
}


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC with average ranks for ties. nan if only one class is present."""
    from scipy.stats import rankdata

    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auroc(score, y, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        v = auroc(score[i], y[i])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot // 2:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def boot_mean(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run(data_dir: Path = DATA, out: Path = ROOT / "results", n_boot: int = 2000, metric: str = "lenient"):
    import pandas as pd

    data_dir, out = Path(data_dir), Path(out)
    test_qids = verify_and_test_qids(data_dir / "splits")           # rule 3, first thing
    pilot = {r["qid"]: r for r in iter_jsonl(data_dir / "splits" / "pilot.jsonl")}
    calib_qids = {r["qid"] for r in iter_jsonl(data_dir / "splits" / "calib.jsonl")}
    assert set(pilot) <= calib_qids, "pilot must be drawn from calib only"
    assert_no_test(pilot, test_qids, "pilot")
    pids = {f"{q}::orig" for q in pilot} | {f"{q}::clar" for q, r in pilot.items() if r["is_ambiguous"]}

    rows = []
    for k, tier in enumerate(TIERS):
        cache = data_dir / "cache" / f"{tier}.jsonl"
        if not cache.exists():
            print(f"{tier}: no cache, skipped")
            continue
        feats = {r["pid"]: r for r in latest_good_rows(cache, set(pilot)) if r["pid"] in pids}
        g = load_graded(tier, data_dir)
        g = g[g["pid"].isin(pids)].set_index("pid")
        if metric not in g:
            raise SystemExit(f"{tier} graded file has no {metric} column. Run python scripts/bg.py grade")

        # 1. AUROC on raw prompts of pilot questions
        orig = [f"{q}::orig" for q in pilot if f"{q}::orig" in feats and f"{q}::orig" in g.index]
        y_correct = np.array([bool(g.at[p, metric]) for p in orig])
        y_amb = np.array([pilot[p.split("::")[0]]["is_ambiguous"] for p in orig])
        for j, f in enumerate(FEATURES[tier]):
            x = np.array([np.nan if feats[p].get(f) is None else float(feats[p][f]) for p in orig])
            ok = np.isfinite(x)
            for target, y in (("raw_answer_correct", y_correct), ("is_ambiguous", y_amb)):
                lo, hi = boot_auroc(x[ok], y[ok], n_boot, seed=100 * k + 10 * j + (target == "is_ambiguous"))
                rows.append({"kind": "auroc", "tier": tier, "feature": f, "target": target, "metric": metric,
                             "n": int(ok.sum()), "n_pos": int(y[ok].sum()), "value": auroc(x[ok], y[ok]),
                             "ci_lo": lo, "ci_hi": hi})

        # 2. clarified minus raw accuracy, ambiguous pilot questions, paired over questions
        amb = [q for q, r in pilot.items() if r["is_ambiguous"]
               and f"{q}::orig" in g.index and f"{q}::clar" in g.index]
        if amb:
            d = np.array([float(g.at[f"{q}::clar", metric]) - float(g.at[f"{q}::orig", metric]) for q in amb])
            lo, hi = boot_mean(d, n_boot, seed=900 + k)
            rows.append({"kind": "clarify_gain", "tier": tier, "feature": "clar_minus_raw", "target": "accuracy",
                         "metric": metric, "n": len(amb), "n_pos": int((d > 0).sum()), "value": float(d.mean()),
                         "ci_lo": lo, "ci_hi": hi})

    rows += _agreement_rows(data_dir, pilot, n_boot)
    rc = risk_coverage(data_dir, pilot, n_boot)
    t0p = tier0_as_probe(data_dir, pilot, n_boot)
    df = pd.DataFrame(rows)
    df.insert(0, "data", "SYNTHETIC" if is_synthetic(data_dir) else "REAL")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pilot_signal_check.csv"
    df.to_csv(path, index=False)
    _print(df, len(pilot), path)
    if t0p is not None:
        t0p.insert(0, "data", df["data"].iloc[0])
        t0p_path = out / "tier0_as_probe.csv"
        t0p.to_csv(t0p_path, index=False)
        _print_t0p(t0p, t0p_path)
    if rc is not None:
        rc.insert(0, "data", df["data"].iloc[0])
        rc_path = out / "pilot_risk_coverage.csv"
        rc.to_csv(rc_path, index=False)
        _print_rc(rc, rc_path)
    return df


COVERAGES = (0.05, 0.10, 0.20, 0.40)


def top_share_accuracy(score: np.ndarray, correct: np.ndarray, coverage: float, others=()) -> list[float]:
    """Accuracy of the top `coverage` share by score, tie aware.

    Answers strictly above the cut value are in, and the tied group at the cut is taken in
    fractionally at its own mean accuracy, so sc_agree, with only six distinct values, gives a
    result that does not depend on arbitrary sort order. `others` are more correctness arrays
    scored on exactly the same slice.
    """
    n = len(score)
    k = coverage * n
    order = np.sort(score)[::-1]
    cut = order[min(int(np.ceil(k)) - 1, n - 1)]
    above, tied = score > cut, score == cut
    take = k - above.sum()
    out = []
    for c in (correct, *others):
        c = c.astype(float)
        out.append(float((c[above].sum() + take * c[tied].mean()) / k))
    return out


def risk_coverage(data_dir: Path, pilot: dict, n_boot: int):
    """Tier 0 accuracy on its most confident slices, against Tier 1 and 2 on the same questions."""
    import pandas as pd

    feats0 = {r["pid"]: r for r in latest_good_rows(data_dir / "cache" / "tier0.jsonl", set(pilot))}
    graded = {}
    for tier in TIERS:
        try:
            g = load_graded(tier, data_dir)
        except Exception:
            return None
        graded[tier] = dict(zip(g["pid"], g["lenient"]))
    qs = [q for q in pilot if f"{q}::orig" in feats0 and all(f"{q}::orig" in graded[t] for t in TIERS)]
    if not qs:
        return None
    pid = [f"{q}::orig" for q in qs]
    c0, c1, c2 = (np.array([bool(graded[t][p]) for p in pid]) for t in TIERS)
    rng = np.random.default_rng(31)
    boots = rng.integers(0, len(qs), size=(n_boot, len(qs)))
    rows = []

    def ci(vals):
        return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

    for tier, c in (("tier1", c1), ("tier2", c2)):
        lo, hi = ci(c[boots].mean(axis=1))
        rows.append({"sort_by": "none", "coverage": 1.0, "n_questions": len(qs), "slice": f"{tier} overall",
                     "accuracy": float(c.mean()), "ci_lo": lo, "ci_hi": hi})
    for f in ("lp_min", "sc_agree"):
        x = np.array([np.nan if feats0[p].get(f) is None else float(feats0[p][f]) for p in pid])
        x = np.nan_to_num(x, nan=-np.inf)                          # missing confidence sorts last
        for cov in COVERAGES:
            # share of all questions tied at the cut value. Above the coverage, the slice is one tie group
            cut = np.sort(x)[::-1][min(int(np.ceil(cov * len(x))) - 1, len(x) - 1)]
            tied_share = float((x == cut).mean())
            a0, a1, a2 = top_share_accuracy(x, c0, cov, (c1, c2))
            b = np.array([top_share_accuracy(x[i], c0[i], cov, (c1[i], c2[i])) for i in boots])
            for name, val, col in (("tier0 slice", a0, b[:, 0]), ("tier1 on same questions", a1, b[:, 1]),
                                   ("tier2 on same questions", a2, b[:, 2]),
                                   ("tier0 minus tier1, same questions", a0 - a1, b[:, 0] - b[:, 1])):
                lo, hi = ci(col)
                rows.append({"sort_by": f, "coverage": cov, "n_questions": round(cov * len(qs), 1), "slice": name,
                             "accuracy": val, "ci_lo": lo, "ci_hi": hi, "share_tied_at_cut": tied_share})
    return pd.DataFrame(rows)


def _print_rc(rc, path: Path) -> None:
    fmt = lambda r: f"{r.accuracy:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"  # noqa: E731
    print(f"\nTier 0 risk-coverage on {int(rc[rc['sort_by'] == 'none']['n_questions'].iloc[0])} raw pilot prompts, "
          "lenient accuracy [bootstrap 95 percent CI]")
    for r in rc[rc["sort_by"] == "none"].itertuples():
        print(f"  {r.slice:16s} {fmt(r)}")
    print(f"  {'sorted by':9s} {'cover':>5s} {'n':>5s} {'tier0 slice':>18s} {'tier1 same qs':>18s} "
          f"{'tier2 same qs':>18s} {'tier0 - tier1':>20s}")
    for (f, cov), g in rc[rc["sort_by"] != "none"].groupby(["sort_by", "coverage"], sort=False):
        g = g.set_index("slice")
        cells = [fmt(next(g.loc[[s]].itertuples())) for s in ("tier0 slice", "tier1 on same questions",
                                                              "tier2 on same questions")]
        d = g.loc["tier0 minus tier1, same questions"]
        print(f"  {f:9s} {cov:5.0%} {g['n_questions'].iloc[0]:5.0f} {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s} "
              f"{d['accuracy']:+.2f} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]")
    flat = rc[(rc["sort_by"] != "none") & (rc["share_tied_at_cut"] > rc["coverage"])]
    for f, g in flat.groupby("sort_by"):
        print(f"  note: {g['share_tied_at_cut'].max():.0%} of answers share the top {f} value, so coverages "
              f"{', '.join(f'{c:.0%}' for c in sorted(g['coverage'].unique()))} are all the same tie group")
    print(f"wrote {path}")


PROBE_FEATURES = ("lp_min", "lp_mean", "sc_agree", "sc_distinct")
PROBE_TARGETS = ("tier1_right", "tier2_right", "tier2_right_tier1_wrong")


def tier0_as_probe(data_dir: Path, pilot: dict, n_boot: int):
    """AUROC of Tier 0 features for the other tiers' correctness on the raw prompt."""
    import pandas as pd

    feats0 = {r["pid"]: r for r in latest_good_rows(data_dir / "cache" / "tier0.jsonl", set(pilot))}
    graded = {}
    for tier in ("tier1", "tier2"):
        try:
            g = load_graded(tier, data_dir)
        except Exception:
            return None
        graded[tier] = dict(zip(g["pid"], g["lenient"]))
    pid = [f"{q}::orig" for q in pilot if f"{q}::orig" in feats0 and all(f"{q}::orig" in graded[t] for t in graded)]
    if not pid:
        return None
    c1 = np.array([bool(graded["tier1"][p]) for p in pid])
    c2 = np.array([bool(graded["tier2"][p]) for p in pid])
    targets = {"tier1_right": c1, "tier2_right": c2, "tier2_right_tier1_wrong": c2 & ~c1}
    rows = []
    for j, f in enumerate(PROBE_FEATURES):
        x = np.array([np.nan if feats0[p].get(f) is None else float(feats0[p][f]) for p in pid])
        ok = np.isfinite(x)
        for k, (name, y) in enumerate(targets.items()):
            lo, hi = boot_auroc(x[ok], y[ok], n_boot, seed=500 + 10 * j + k)
            rows.append({"feature": f, "target": name, "n": int(ok.sum()), "n_pos": int(y[ok].sum()),
                         "auroc": auroc(x[ok], y[ok]), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def _print_t0p(t, path: Path) -> None:
    print(f"\nTier 0 features as a probe for the paid tiers, {int(t['n'].max())} raw pilot prompts. "
          "AUROC [bootstrap 95 percent CI]")
    pos = t.groupby("target")["n_pos"].first()
    print(f"  {'feature':12s} " + " ".join(f"{c + f' ({pos[c]} pos)':>30s}" for c in PROBE_TARGETS))
    for f, g in t.groupby("feature", sort=False):
        g = g.set_index("target")
        cells = [f"{g.at[c, 'auroc']:.2f} [{g.at[c, 'ci_lo']:.2f}, {g.at[c, 'ci_hi']:.2f}]" for c in PROBE_TARGETS]
        print(f"  {f:12s} " + " ".join(f"{c:>30s}" for c in cells))
    print(f"wrote {path}")


def _agreement_rows(data_dir: Path, pilot: dict, n_boot: int) -> list[dict]:
    """AUROC of cross-tier answer agreement on the raw prompt for predicting is_ambiguous."""
    norm = {}
    for tier in TIERS:
        try:
            g = load_graded(tier, data_dir)
        except Exception:
            return []
        if g.empty:
            return []
        norm[tier] = dict(zip(g["pid"], g["answer_norm"]))
    qs = [q for q in pilot if all(f"{q}::orig" in norm[t] for t in TIERS)]
    if not qs:
        return []
    a = {t: [norm[t][f"{q}::orig"] for q in qs] for t in TIERS}
    same = lambda x, y: bool(x) and x == y  # noqa: E731, an empty answer agrees with nothing
    feats = {
        "agree_12": np.array([same(x, y) for x, y in zip(a["tier1"], a["tier2"])], float),
        "agree_0_12": np.array([same(z, x) or same(z, y) for z, x, y in zip(a["tier0"], a["tier1"], a["tier2"])], float),
    }
    y = np.array([pilot[q]["is_ambiguous"] for q in qs], bool)
    rows = []
    for j, (name, x) in enumerate(feats.items()):
        lo, hi = boot_auroc(x, y, n_boot, seed=700 + j)
        rows.append({"kind": "auroc", "tier": "cross", "feature": name, "target": "is_ambiguous", "metric": "",
                     "n": len(qs), "n_pos": int(y.sum()), "value": auroc(x, y), "ci_lo": lo, "ci_hi": hi,
                     "agree_rate_ambiguous": float(x[y].mean()), "agree_rate_clear": float(x[~y].mean())})
    return rows


def _print(df, n_pilot: int, path: Path) -> None:
    fmt = lambda r: f"{r.value:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"  # noqa: E731
    a = df[df["kind"] == "auroc"]
    print(f"\nPilot signal check, {df['data'].iloc[0]} data, {n_pilot} calib pilot questions, "
          f"metric {df['metric'].iloc[0]}. AUROC [bootstrap 95 percent CI]")
    print(f"{'tier':6s} {'feature':14s} {'predicts raw answer correct':>28s} {'predicts is_ambiguous':>24s}")
    for (tier, feat), g in a[a["tier"] != "cross"].groupby(["tier", "feature"], sort=False):
        c = g[g["target"] == "raw_answer_correct"].iloc[0]
        m = g[g["target"] == "is_ambiguous"].iloc[0]
        print(f"{tier:6s} {feat:14s} {fmt(c):>28s} {fmt(m):>24s}")
    cross = a[a["tier"] == "cross"]
    if len(cross):
        print("\nCross-tier agreement on the raw prompt, predicting is_ambiguous")
        for r in cross.itertuples():
            print(f"  {r.feature:11s} AUROC {fmt(r)}   agrees on {r.agree_rate_ambiguous:.0%} of ambiguous, "
                  f"{r.agree_rate_clear:.0%} of clear questions")
    gain = df[df["kind"] == "clarify_gain"]
    if len(gain):
        print("\nClarified minus raw accuracy on ambiguous pilot questions [paired bootstrap 95 percent CI]")
        for r in gain.itertuples():
            print(f"  {r.tier}: {r.value:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  over {r.n} questions")
    print("\nAUROC 0.5 is no signal, and below 0.5 means the feature runs the other way. With about 300 "
          "questions the intervals are wide, so read this as a sanity check before the full run, not a result.")
    print(f"wrote {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--metric", choices=["lenient", "em", "lenient_plus"], default="lenient")
    args = ap.parse_args(argv)
    run(args.data_dir, args.out, args.n_boot, args.metric)


if __name__ == "__main__":
    main()
