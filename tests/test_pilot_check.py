"""The pilot signal check reads calib pilot rows only and writes one tidy CSV."""
import numpy as np

from budget_guard import pilot_check
from budget_guard.common import iter_jsonl


def test_pilot_check_runs_on_pilot_only(world, tmp_path):
    df = pilot_check.run(world, tmp_path, n_boot=200)
    assert (tmp_path / "pilot_signal_check.csv").exists()
    pilot = [r for r in iter_jsonl(world / "splits" / "pilot.jsonl")]
    a = df[df["kind"] == "auroc"]
    assert (a["n"] <= len(pilot)).all()
    assert set(a["target"]) == {"raw_answer_correct", "is_ambiguous"}
    assert len(a[a["tier"] != "cross"]) == 2 * (6 + 4 + 4)
    cross = a[a["tier"] == "cross"].set_index("feature")
    assert set(cross.index) == {"agree_12", "agree_0_12"} and (cross["target"] == "is_ambiguous").all()
    assert cross[["agree_rate_ambiguous", "agree_rate_clear"]].notna().all().all()
    g = df[df["kind"] == "clarify_gain"]
    assert set(g["tier"]) == {"tier0", "tier1", "tier2"}
    assert (g["n"] == sum(r["is_ambiguous"] for r in pilot)).all()
    ok = a.dropna(subset=["ci_lo"])
    assert ((ok["ci_lo"] <= ok["value"] + 1e-9) & (ok["value"] <= ok["ci_hi"] + 1e-9)).all()


def test_auroc_helper():
    y = np.array([0, 0, 1, 1], bool)
    assert pilot_check.auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert pilot_check.auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert pilot_check.auroc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5
    assert np.isnan(pilot_check.auroc(np.array([0.1, 0.2]), np.array([1, 1], bool)))


def test_top_share_accuracy_is_tie_aware():
    score = np.array([3.0, 2.0, 2.0, 2.0, 1.0])
    correct = np.array([1, 1, 0, 0, 1], bool)
    other = np.array([0, 1, 1, 1, 0], bool)
    # top 40 percent is 2 answers: the 3.0 one, then one third of the tie group's worth of 2.0 answers
    a, b = pilot_check.top_share_accuracy(score, correct, 0.4, (other,))
    assert a == (1 + 1 * (1 / 3)) / 2
    assert b == (0 + 1 * 1.0) / 2
    # order inside a tie group must not matter
    perm = np.array([0, 3, 2, 1, 4])
    assert pilot_check.top_share_accuracy(score[perm], correct[perm], 0.4)[0] == a


def test_risk_coverage_written(world, tmp_path):
    import pandas as pd

    pilot_check.run(world, tmp_path, n_boot=100)
    rc = pd.read_csv(tmp_path / "pilot_risk_coverage.csv")
    assert set(rc["sort_by"]) == {"none", "lp_min", "sc_agree"}
    assert set(rc[rc["sort_by"] == "lp_min"]["coverage"]) == set(pilot_check.COVERAGES)
    assert {"tier1 overall", "tier2 overall", "tier0 slice", "tier1 on same questions"} <= set(rc["slice"])


def test_tier0_as_probe_written(world, tmp_path):
    import pandas as pd

    pilot_check.run(world, tmp_path, n_boot=100)
    t = pd.read_csv(tmp_path / "tier0_as_probe.csv")
    assert set(t["feature"]) == set(pilot_check.PROBE_FEATURES)
    assert set(t["target"]) == set(pilot_check.PROBE_TARGETS)
    pos = t.groupby("target")["n_pos"].first()
    assert pos["tier2_right_tier1_wrong"] <= min(pos["tier1_right"], pos["tier2_right"]) or pos["tier1_right"] == 0
