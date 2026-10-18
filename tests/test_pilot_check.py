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
    assert len(a) == 2 * (6 + 4 + 4)
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
