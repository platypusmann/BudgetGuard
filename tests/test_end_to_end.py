"""The whole synthetic pipeline, offline, ending in the headline figure."""
import json

import pandas as pd

from budget_guard import evaluate

POLICIES = {"always_tier0", "always_tier1", "always_tier2", "random_matched", "cascade_frugal", "threshold_fixed",
            "threshold_tuned", "slack_schedule", "dp_oracle", "learned_router", "routellm", "hindsight_oracle"}


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
