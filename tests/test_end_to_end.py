"""The whole synthetic pipeline, offline, ending in the headline figure."""
import json

import pandas as pd

from budget_guard import evaluate

POLICIES = {"always_tier0", "always_tier1", "always_tier2", "random_matched", "cascade_frugal", "threshold_fixed",
            "threshold_tuned", "slack_schedule", "dp_oracle", "learned_router", "routellm", "hindsight_oracle",
            "clarify_all", "ambiguity_oracle"}


def test_pipeline_produces_headline_figure(world, tmp_path):
    out = tmp_path / "results"
    evaluate.main(["--data-dir", str(world), "--out", str(out), "--quick"])

    fig = out / "figures" / "headline_cost_vs_accuracy.png"
    assert fig.exists() and fig.stat().st_size > 10_000
    s = pd.read_csv(out / "summary.csv")
    assert set(s["policy"]) == POLICIES
    assert (s["data"] == "SYNTHETIC").all()
    assert (s[s["masked"]]["violation_rate"] == 0).all()
    assert (s["eval_split"] == "test").all()
    # sweeps ran: two clarify fees and two Tier 0 prices in --quick
    assert set(s["fee_mult"]) == {0.5, 1.0}
    assert set(s["tier0_price"]) == {"0.02/0.08", "0/0"}
    # paired differences against both references, with CIs
    p = pd.read_csv(out / "paired.csv")
    assert set(p["reference"]) == {"threshold_fixed", "threshold_tuned"}
    assert (p["diff_ci_lo"] <= p["diff_mean"] + 1e-12).all() and (p["diff_mean"] <= p["diff_ci_hi"] + 1e-12).all()
    # clarify precision and recall are reported for policies that clarify
    dp = s[(s["policy"] == "dp_oracle") & s["headline"]]
    assert dp["clarify_recall"].notna().all()
    assert (out / "soft_frontier.csv").exists() and (out / "results_table.md").exists()
    info = json.loads((out / "run_info.json").read_text())
    assert info["data"] == "SYNTHETIC"
    assert any(out.glob("decisions_*.parquet"))
    pos = pd.read_csv(out / "clarify_reference.csv")
    ref = pos[pos["policy"].isin(["clarify_all", "ambiguity_oracle"])]
    assert set(ref.dropna(subset=["position"]).groupby("policy")["position"].first().round(6)) <= {0.0, 1.0}


def test_hindsight_beats_everyone_and_dp_beats_floor(world, tmp_path):
    out = tmp_path / "r2"
    evaluate.main(["--data-dir", str(world), "--out", str(out), "--quick", "--no-sweeps", "--no-figures",
                   "--eval-split", "tune"])
    s = pd.read_csv(out / "summary.csv")
    s = s[s["masked"]]
    for b, g in s.groupby("budget_u2_per_q"):
        acc = g.set_index("policy")["accuracy"]
        assert acc["hindsight_oracle"] >= acc.drop("hindsight_oracle").max() - 1e-12
        assert acc["dp_oracle"] >= acc["always_tier0"] - 0.05


def test_both_metrics_reported_and_ranking_checked(world, tmp_path):
    out = tmp_path / "r3"
    evaluate.main(["--data-dir", str(world), "--out", str(out), "--quick", "--no-sweeps", "--no-figures"])
    s = pd.read_csv(out / "summary.csv")
    assert {"accuracy_lenient_plus", "acc_lenient_plus_ci_lo", "acc_lenient_plus_ci_hi", "accuracy_em"} <= set(s.columns)
    # lenient_plus keeps every lenient match, so rescoring can only add correct answers
    assert (s["accuracy_lenient_plus"] >= s["accuracy"] - 1e-12).all()
    r = pd.read_csv(out / "metric_ranking.csv")
    assert {"rank_lenient", "rank_lenient_plus", "rank_changed"} <= set(r.columns)
    info = json.loads((out / "run_info.json").read_text())
    assert info["ranking_check"]["secondary"] == "lenient_plus"


def test_ranking_check_flags_a_swap():
    rows = []
    for pol, a1, a2 in [("a", 0.40, 0.40), ("b", 0.35, 0.45), ("c", 0.20, 0.20)]:
        rows.append({"headline": True, "masked": True, "budget_u2_per_q": 0.35, "policy": pol,
                     "accuracy": a1, "accuracy_lenient_plus": a2})
    r = evaluate.ranking_check(pd.DataFrame(rows), "lenient", "lenient_plus").set_index("policy")
    assert r.loc["a", "rank_changed"] and r.loc["b", "rank_changed"] and not r.loc["c", "rank_changed"]
    assert r.loc["b", "rank_lenient"] == 2 and r.loc["b", "rank_lenient_plus"] == 1
