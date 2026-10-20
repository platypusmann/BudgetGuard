"""Final evaluation: sweep budgets, clarify fees and Tier 0 prices, write CSVs and figures.

Order of operations, which is what keeps test clean:
  0. check the caches are complete (completeness.py) and hash the experiment design, printing it
     in the header. On real data, refuse on any missing pid, unresolved error or wrong prompt
     version, and refuse unless the design is the frozen design in configs/experiment.sha256
     and no --quick overrides are in play
  1. split.verify() and load the test qids
  2. fit the shared signals on calib (asserts no test qid)
  3. per (fee, price) config: fit and tune every policy on calib and tune only (asserts no test qid)
  4. only then run every policy on the evaluation split streams

Outputs in --out. Every kind of run has its own default folder, so runs never mix:
  results/synthetic/  synthetic data
  results/dev_tune/   real data, dry run on tune
  results/final/      real data, the one test evaluation (must not exist yet or be empty)
  summary.csv        one row per config x policy x budget x masked
  paired.csv         per-episode accuracy differences against every reference: threshold_fixed (the
                     pre-registered two-tier reference), threshold_tuned, threshold_three_tier and
                     best_simple_tier1 (the best of always Tier 1, the cascade and the three-tier
                     threshold at each budget, picked on tune, never on the evaluation split)
  best_simple_tier1.csv  which policy was picked at each budget, with every candidate's tune score
  soft_frontier.csv  the fixed-threshold rule swept with no cap, the average-cost frontier
  metric_ranking.csv headline policy ranking under the primary metric and each secondary one,
                     same decisions rescored, with every rank change flagged
  clarify_reference.csv  where each policy lands between blind clarify-all (0) and the ambiguity
                     oracle (1) at each headline budget, with clarify precision and recall
  schedule.csv       the fitted slack schedule, headline config
  decisions_*.parquet  every decision for a few policies at one budget, headline config
  run_info.json      settings, signal AUROCs, timings, synthetic flag
  figures/           made by report.py

  python -m budget_guard.evaluate --data-dir data/synthetic
  python -m budget_guard.evaluate --eval-split tune          # development, never looks at test
  python -m budget_guard.evaluate --quick                     # tiny settings, for tests
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import completeness, design, devdata, provenance
from . import episodes as ep
from .common import DATA, ROOT
from .features import load_embeddings, load_features
from .leakage import assert_no_test, verify_and_test_qids
from .policies import FitContext, build_all, fit_all
from .policies.threshold import ESC_RATES, ThresholdRule
from .signals import SignalModel
from .synthetic import is_synthetic

RESULTS = ROOT / "results"
OUT_SYNTHETIC, OUT_DRY_RUN, OUT_FINAL = RESULTS / "synthetic", RESULTS / "dev_tune", RESULTS / "final"


def default_out(synthetic: bool, eval_split: str) -> Path:
    if synthetic:
        return OUT_SYNTHETIC
    return OUT_DRY_RUN if eval_split == "tune" else OUT_FINAL
DECISION_LOG_POLICIES = ("threshold_fixed", "threshold_tuned", "dp_oracle", "learned_router")
# simple policies that can use Tier 1. The best one per budget is picked on tune. Ties go to the
# earlier, simpler one
TIER1_SIMPLE = ("always_tier1", "cascade_frugal", "threshold_three_tier")
REFERENCES = ("threshold_fixed", "threshold_tuned", "threshold_three_tier", "best_simple_tier1")
HEADLINE_REFS = ("threshold_fixed", "threshold_three_tier", "best_simple_tier1")


def bootstrap_ci(x: np.ndarray, n: int, seed: int = 0) -> tuple[float, float]:
    """Percentile 95 percent CI of the mean, resampling episodes."""
    if len(x) < 2 or n <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def metrics(r: ep.RunResult, env: ep.Env, seeds: np.ndarray, n_boot: int) -> tuple[dict, np.ndarray]:
    acc_e = r.correct.mean(axis=1)
    lo, hi = bootstrap_ci(acc_e, n_boot)
    amb = env.is_amb[r.episodes]
    clar = r.clarified
    tier = np.where(clar, r.final - ep.C0, r.final)
    answered = r.final != ep.SKIP
    total = r.total_spend
    K = r.correct.shape[1]
    seed_means = [acc_e[seeds == s].mean() for s in np.unique(seeds)]
    # the same decisions under every other graded metric, for the ranking check
    alt = {}
    for m in env.correct_by:
        if m == env.metric:
            continue
        acc_m = ep.rescore(r, env, m).mean(axis=1)
        lo_m, hi_m = bootstrap_ci(acc_m, n_boot)
        alt.update({f"accuracy_{m}": float(acc_m.mean()), f"acc_{m}_ci_lo": lo_m, f"acc_{m}_ci_hi": hi_m})
    return {
        "accuracy": float(acc_e.mean()), "acc_ci_lo": lo, "acc_ci_hi": hi,
        "acc_seed_sd": float(np.std(seed_means)) if len(seed_means) > 1 else float("nan"),
        "spend_mean": float(total.mean()), "spend_p95": float(np.quantile(total, 0.95)),
        "spend_per_q": float(total.mean() / K), "spend_frac_of_B": float(total.mean() / r.B) if r.B > 0 else float("nan"),
        "violation_rate": float(r.violation.mean()),
        "skip_rate": float((~answered).mean()), "downgrade_rate": float(r.downgraded.mean()),
        "tier2_rate": float((answered & (tier == 2)).mean()), "tier1_rate": float((answered & (tier == 1)).mean()),
        "probe_rate": float((r.probes > 0).mean()),
        "clarify_rate": float(clar.mean()),
        "clarify_precision": float(amb[clar].mean()) if clar.any() else float("nan"),
        "clarify_recall": float(clar[amb].mean()) if amb.any() else float("nan"),
        **alt,
    }, acc_e


SECONDARY = {"lenient": "lenient_plus", "lenient_plus": "lenient", "em": "lenient"}


def ranking_check(summary, primary: str, other: str):
    """Rank headline policies under both metrics at each budget and flag every change.

    Ranks use accuracy rounded to 4 decimals, ties share the better rank. A swap between two
    policies whose accuracies are within a CI of each other is noise, so the table keeps both
    accuracies next to the ranks for the reader to judge.
    """
    import pandas as pd

    col = f"accuracy_{other}"
    h = summary[summary["headline"] & summary["masked"]]
    if col not in h:
        return None
    rows = []
    for b, g in h.groupby("budget_u2_per_q"):
        g = g.set_index("policy")
        r1 = g["accuracy"].round(4).rank(ascending=False, method="min")
        r2 = g[col].round(4).rank(ascending=False, method="min")
        for pol in g.index:
            rows.append({"budget_u2_per_q": b, "policy": pol, f"accuracy_{primary}": g.at[pol, "accuracy"],
                         f"rank_{primary}": int(r1[pol]), col: g.at[pol, col], f"rank_{other}": int(r2[pol]),
                         "rank_changed": int(r1[pol]) != int(r2[pol])})
    return pd.DataFrame(rows)


def print_headline_comparisons(paired) -> None:
    """DP minus each headline reference at every headline budget, paired over episodes."""
    h = paired[paired["headline"] & (paired["policy"] == "dp_oracle") & paired["reference"].isin(HEADLINE_REFS)]
    if h.empty:
        return
    names = {"threshold_fixed": "fixed two-tier (pre-registered)", "threshold_three_tier": "three-tier threshold",
             "best_simple_tier1": "best simple with Tier 1"}
    print("\nDP minus each reference, headline config, points [paired 95 percent CI]:")
    for b, g in h.groupby("budget_u2_per_q"):
        cells = []
        for ref in HEADLINE_REFS:
            r = g[g["reference"] == ref]
            if len(r):
                r = r.iloc[0]
                pick = f" ({r['reference_policy']})" if ref == "best_simple_tier1" else ""
                cells.append(f"{names[ref]}{pick} {r['diff_mean'] * 100:+.1f} [{r['diff_ci_lo'] * 100:+.1f}, "
                             f"{r['diff_ci_hi'] * 100:+.1f}]")
        print(f"  {b:g} u2: " + " | ".join(cells))


def clarify_positions(summary):
    """Place every policy between the two clarify references at each headline budget.

    position = (acc - acc clarify_all) / (acc ambiguity_oracle - acc clarify_all), so blind
    clarify-all is 0 and clarify-exactly-when-ambiguous is 1. Values outside [0, 1] are possible
    and meaningful: under a tight budget clarifying everything can be worse than never clarifying,
    and a policy that also routes tiers better than the oracle's DP can pass 1. Undefined (nan)
    where the two references are within half a point of each other.
    """
    import pandas as pd

    h = summary[summary["headline"] & summary["masked"]]
    rows = []
    for b, g in h.groupby("budget_u2_per_q"):
        g = g.set_index("policy")
        if not {"clarify_all", "ambiguity_oracle"} <= set(g.index):
            return None
        lo, hi = g.at["clarify_all", "accuracy"], g.at["ambiguity_oracle", "accuracy"]
        span = hi - lo
        for pol, r in g.iterrows():
            rows.append({"budget_u2_per_q": b, "policy": pol, "accuracy": r["accuracy"],
                         "acc_clarify_all": lo, "acc_ambiguity_oracle": hi,
                         "position": (r["accuracy"] - lo) / span if abs(span) >= 0.005 else float("nan"),
                         "clarify_rate": r["clarify_rate"], "clarify_precision": r["clarify_precision"],
                         "clarify_recall": r["clarify_recall"]})
    return pd.DataFrame(rows)


def build_envs(data_dir: Path, eval_split: str, metric: str, embed: str, test_qids):
    feats = load_features(data_dir)
    emb = None
    if embed != "none":
        try:
            emb = load_embeddings(data_dir, embed)
        except FileNotFoundError as e:
            print(f"no embeddings, routellm skipped: {e}")
    prices = ep.load_prices()
    envs = {s: ep.Env(feats, s, prices, metric=metric, emb=emb) for s in {"calib", "tune", eval_split}}
    del feats
    sm = SignalModel().fit(envs["calib"], test_qids)
    for e in envs.values():
        sm.apply(e)
    for s in ("calib", "tune"):
        if envs[s].n == 0:
            raise SystemExit(f"the {s} pool is empty. Is the full cache there, or only the pilot?")
    return envs, sm, emb is not None


def run_config(envs, eval_split, cfg, fee_mult, t0_price, budgets_u2, test_qids, n_boot, with_routellm,
               out_dir, headline, verbose=True):
    prices = ep.load_prices(tuple(t0_price), batch=cfg.get("batch_prices", False))
    calib, tune, ev = (envs["calib"].with_prices(prices), envs["tune"].with_prices(prices),
                       envs[eval_split].with_prices(prices))
    K = int(cfg["K"])
    u2 = calib.unit_cost()
    fee = fee_mult * u2
    budgets = [b * u2 * K for b in budgets_u2]
    tune_eps = ep.sample_episodes(tune.n, K, int(cfg["tune_episodes"]), 1000, "tune")
    fit_eps = ep.sample_episodes(calib.n, K, int(cfg["fit_episodes"]), 2000, "calib")
    ctx = FitContext(calib, tune, fee, u2, K, budgets, tune_eps, fit_eps, cfg, test_qids)
    if verbose:
        print(f"  config fee {fee_mult}x u2, tier0 price {t0_price}: u2 = ${u2:.3e}, fee = ${fee:.3e}")
    policies = fit_all(build_all(include_routellm=with_routellm), ctx, verbose=verbose)
    # the best simple Tier 1 policy at each budget, by accuracy on the tune streams only
    best_simple, best_scores = {}, {}
    for bu, B in zip(budgets_u2, budgets):
        sc = {n: ctx.tune_score(ctx.fitted[n], B)[0] for n in TIER1_SIMPLE if n in ctx.fitted}
        best_simple[bu] = max(sc, key=lambda n: (round(sc[n], 6), -TIER1_SIMPLE.index(n)))
        best_scores[bu] = sc

    # fitting is over. From here on the evaluation split is read, nothing is fitted
    eval_eps = np.concatenate([ep.sample_episodes(ev.n, K, int(cfg["episodes_per_seed"]), s, eval_split)
                               for s in cfg["seeds"]])
    seeds = np.repeat(np.asarray(cfg["seeds"]), int(cfg["episodes_per_seed"]))
    base = {"eval_split": eval_split, "metric": ev.metric, "fee_mult": fee_mult, "fee_usd": fee,
            "tier0_price": f"{t0_price[0]:g}/{t0_price[1]:g}", "u2_usd": u2, "K": K, "n_episodes": len(eval_eps),
            "headline": headline}
    rows, paired_rows, acc_by = [], [], {}
    for bu, B in zip(budgets_u2, budgets):
        for p in policies:
            modes = [True] if p.budget_aware else [True, False]
            for masked in modes:
                if getattr(p, "is_oracle", False):
                    r = p.run(ev, eval_eps, B, fee)
                else:
                    r = ep.run(ev, p, eval_eps, B, fee, masked=masked)
                m, acc_e = metrics(r, ev, seeds, n_boot)
                rows.append({**base, "policy": p.name, "label": p.label, "budget_aware": p.budget_aware,
                             "masked": masked, "budget_u2_per_q": bu, "budget_usd": B, "budget_usd_per_q": B / K, **m})
                if masked:
                    acc_by[(p.name, bu)] = acc_e
                if masked and headline and bu == budgets_u2[len(budgets_u2) // 2] and p.name in DECISION_LOG_POLICIES:
                    r.log_frame(ev).to_parquet(out_dir / f"decisions_{p.name}.parquet", index=False)
        for ref in REFERENCES:
            ref_policy = best_simple[bu] if ref == "best_simple_tier1" else ref
            if (ref_policy, bu) not in acc_by:
                continue
            a_ref = acc_by[(ref_policy, bu)]
            for p in policies:
                if p.name == ref:
                    continue
                d = acc_by[(p.name, bu)] - a_ref
                lo, hi = bootstrap_ci(d, n_boot, seed=1)
                paired_rows.append({**base, "policy": p.name, "reference": ref, "reference_policy": ref_policy,
                                    "budget_u2_per_q": bu,
                                    "budget_usd": B, "diff_mean": float(d.mean()), "diff_ci_lo": lo,
                                    "diff_ci_hi": hi, "share_episodes_better": float((d > 0).mean()),
                                    "share_episodes_worse": float((d < 0).mean())})

    soft_rows, schedule = [], None
    if headline:
        fixed = ctx.fitted["threshold_fixed"]
        u = 1 - calib.signals["p0"][0]
        rates = ESC_RATES
        taus = [np.inf if r <= 0 else -np.inf if r >= 1 else float(np.quantile(u, 1 - r)) for r in rates]
        for rate, tau in zip(rates, taus):
            pol = ThresholdRule(tau, fixed.kappa)
            pol.name = "soft_frontier"
            r = ep.run(ev, pol, eval_eps, 1e9, fee, masked=False)
            m, _ = metrics(r, ev, seeds, n_boot)
            soft_rows.append({**base, "policy": "soft_frontier", "escalate_rate_calib": rate, **m})
        schedule = ctx.fitted["slack_schedule"].table()
    extras = {"dp_text_bins": getattr(ctx.fitted.get("dp_oracle"), "T", None),
              "dp_text_bins_tune_scores": getattr(ctx.fitted.get("dp_oracle"), "text_bins_scores", None),
              "router_model": getattr(ctx.fitted.get("learned_router"), "model_name", None),
              "router_scores": getattr(ctx.fitted.get("learned_router"), "scores", None),
              "schedule_agreement": getattr(ctx.fitted.get("slack_schedule"), "agreement", None)}
    best_rows = [{**base, "budget_u2_per_q": bu, "picked": best_simple[bu],
                  **{f"tune_accuracy_{n}": v for n, v in best_scores[bu].items()}} for bu in budgets_u2]
    return rows, paired_rows, soft_rows, schedule, extras, best_rows


def main(argv=None) -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=None,
                    help="default results/synthetic, results/dev_tune or results/final by kind of run")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--eval-split", choices=["test", "tune"], default="test")
    ap.add_argument("--metric", choices=["lenient", "em", "lenient_plus"], default=None)
    ap.add_argument("--embed", choices=["hash", "minilm", "none"], default="hash")
    ap.add_argument("--no-sweeps", action="store_true", help="headline config only")
    ap.add_argument("--quick", action="store_true", help="tiny settings for tests and smoke runs")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    t_start = time.time()

    synthetic = is_synthetic(args.data_dir)
    if args.out is None:
        args.out = default_out(synthetic, args.eval_split)
    design_sha, frozen = design.check(args.config or design.DESIGN)
    stamp = design_sha + ("+quick" if args.quick else "") + ("" if frozen else "+unfrozen")
    code = provenance.git_state()
    print(provenance.describe(code))
    print(f"experiment design sha256 {design_sha}: "
          + ("frozen design" if frozen and not args.quick else
             "NOT the frozen design" + (" (--quick overrides)" if args.quick else "")))
    if not synthetic and (args.quick or not frozen):
        raise SystemExit("refusing to evaluate real data on anything but the frozen design. "
                         f"configs/experiment.sha256 says {design.locked_hash()}, this run is {stamp}")

    test_qids = verify_and_test_qids(args.data_dir / "splits")    # rule 3, first thing
    # a dry run on tune never needs test: it checks, grades, featurizes and embeds calib and tune
    # only, into data/dev, and no test row is ever decoded (devdata.py)
    dev = args.eval_split == "tune"
    if dev:
        print("DRY RUN on tune, calib and tune only. Debugging numbers, not results.")
    problems, cache_report = completeness.check_cache(args.data_dir, splits=devdata.DEV_SPLITS if dev else None)
    print("cache completeness:\n" + completeness.describe(cache_report))
    if problems and not synthetic:
        raise SystemExit("refusing to evaluate an incomplete real cache:\n  " + "\n  ".join(problems))
    cfg = ep.load_experiment(args.config)
    if args.quick:
        cfg.update(K=20, episodes_per_seed=20, seeds=[0, 1], tune_episodes=20, fit_episodes=10, bootstrap=200,
                   budgets=[0.05, 0.2, 0.75], sweep_budgets=[0.2], clarify_fees=[0.5, 1.0],
                   tier0_prices=[[0.02, 0.08], [0.0, 0.0]])
        cfg["dp"] = {**cfg["dp"], "grid": 200}
    metric = args.metric or cfg.get("metric", "lenient")
    args.out.mkdir(parents=True, exist_ok=True)

    feature_root = args.data_dir
    if dev:
        devdata.build(args.data_dir, args.embed)
        devdata.assert_clean(args.data_dir, test_qids)
        feature_root = devdata.dev_root(args.data_dir)
    envs, sm, with_routellm = build_envs(feature_root, args.eval_split, metric, args.embed, test_qids)
    for s in ("calib", "tune"):
        assert_no_test(envs[s].qids, test_qids, f"{s} pool")
    print(f"evaluate on {args.eval_split} ({envs[args.eval_split].n} questions), metric {metric}, "
          f"{'SYNTHETIC' if synthetic else 'REAL'} data")

    default_fee, default_price = 1.0, [0.02, 0.08]
    configs = [(default_fee, default_price, cfg["budgets"], True)]
    if not args.no_sweeps:
        configs += [(f, default_price, cfg["sweep_budgets"], False) for f in cfg["clarify_fees"] if f != default_fee]
        configs += [(default_fee, p, cfg["sweep_budgets"], False) for p in cfg["tier0_prices"] if p != default_price]

    rows, paired, soft, extras, best = [], [], [], {}, []
    for fee_mult, price, budgets, headline in configs:
        t = time.time()
        r, pr, so, sched, ex, be = run_config(envs, args.eval_split, cfg, fee_mult, price, budgets, test_qids,
                                          int(cfg["bootstrap"]), with_routellm, args.out, headline)
        rows += r
        paired += pr
        best += be
        soft += so
        extras[f"fee {fee_mult} price {price[0]:g}/{price[1]:g}"] = ex
        if sched is not None:
            sched.insert(0, "design_sha256", stamp)
            sched.to_csv(args.out / "schedule.csv", index=False)
        print(f"  config done in {time.time() - t:.0f}s")

    tag = "SYNTHETIC" if synthetic else "REAL"
    for name, data in (("summary", rows), ("paired", paired), ("soft_frontier", soft), ("best_simple_tier1", best)):
        df = pd.DataFrame(data)
        df.insert(0, "data", tag)
        df.insert(1, "design_sha256", stamp)
        df.to_csv(args.out / f"{name}.csv", index=False)
    info = {"data": tag, "code": code, "mode": "dry run on tune, debugging only" if dev else "final", "design_sha256": stamp, "design_frozen": frozen and not args.quick,
            "cache_completeness": cache_report, "data_dir": str(args.data_dir), "eval_split": args.eval_split, "metric": metric,
            "config": cfg, "signal_auc_eval": sm.auc_report(envs[args.eval_split]),
            "pool_sizes": {s: e.n for s, e in envs.items()},
            "eval_ambiguous_rate": float(envs[args.eval_split].is_amb.mean()),
            "dropped": {s: e.n_dropped for s, e in envs.items()}, "routellm": with_routellm,
            "fits": extras, "seconds": round(time.time() - t_start, 1)}
    print_headline_comparisons(pd.DataFrame(paired))
    other = SECONDARY.get(metric)
    summary = pd.read_csv(args.out / "summary.csv")
    rank = ranking_check(summary, metric, other) if other else None
    if rank is None:
        print(f"ranking check skipped: no {other} grades. Rerun grade.py, then features.py")
        info["ranking_check"] = "skipped"
    else:
        rank.insert(0, "design_sha256", stamp)
        rank.to_csv(args.out / "metric_ranking.csv", index=False)
        changed = rank[rank["rank_changed"]]
        print(f"\npolicy ranking, {metric} (primary) against {other} (secondary), same decisions rescored:")
        if changed.empty:
            print("  ranking is identical at every headline budget")
        for b, g in changed.groupby("budget_u2_per_q"):
            moves = ", ".join(f"{r.policy} {getattr(r, 'rank_' + metric)}->{getattr(r, 'rank_' + other)}"
                              for r in g.itertuples())
            print(f"  RANKING CHANGES at budget {b:g} u2: {moves}")
        info["ranking_check"] = {"secondary": other, "budgets_with_changes": sorted(changed["budget_u2_per_q"].unique().tolist()),
                                 "n_rank_changes": int(len(changed))}
    pos = clarify_positions(summary)
    if pos is not None:
        pos.insert(0, "design_sha256", stamp)
        pos.to_csv(args.out / "clarify_reference.csv", index=False)
        mid = sorted(pos["budget_u2_per_q"].unique())[len(pos["budget_u2_per_q"].unique()) // 2]
        m = pos[pos["budget_u2_per_q"] == mid].sort_values("position", ascending=False)
        r0 = m.iloc[0]
        print(f"\nclarify references at {mid:g} u2: clarify-all {r0.acc_clarify_all:.3f}, "
              f"ambiguity oracle {r0.acc_ambiguity_oracle:.3f}. Position 0 is clarify-all, 1 the oracle")
        for r in m.itertuples():
            print(f"  {r.policy:18s} acc {r.accuracy:.3f}  position {r.position:+.2f}  clarify {r.clarify_rate:.0%}  "
                  f"precision {r.clarify_precision:.2f}  recall {r.clarify_recall:.2f}")
        info["clarify_reference"] = {p: round(float(v), 3) for p, v in zip(m["policy"], m["position"])}
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"wrote {args.out} in {time.time() - t_start:.0f}s")

    if not args.no_figures:
        from .report import make_figures

        make_figures(args.out)


if __name__ == "__main__":
    main()
