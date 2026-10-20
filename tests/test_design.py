"""The frozen experiment design: hash, lock and the text-bin selection rule."""
import pytest

from budget_guard import design, evaluate


def test_design_matches_lock():
    h, frozen = design.check()
    assert frozen, "configs/experiment.yaml changed after it was frozen"
    assert len(h) == 64


def test_hash_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_bytes(b"K: 100\nseeds: [0]\n")
    b.write_bytes(b"K: 100\r\nseeds: [0]\r\n")
    assert design.design_hash(a) == design.design_hash(b)


@pytest.mark.parametrize("scores, expected", [
    ({1: 0.300, 2: 0.304, 3: 0.304}, 1),     # under 0.5 points, keep the simplest
    ({1: 0.300, 2: 0.305, 3: 0.306}, 2),     # 2 earns its keep, 3 does not beat 2 by 0.5
    ({1: 0.300, 2: 0.303, 3: 0.306}, 3),     # 3 beats the current choice (1) by 0.6
    ({1: 0.310, 2: 0.300, 3: 0.309}, 1),
])
def test_text_bin_rule(scores, expected):
    assert design.choose_text_bins(scores, 0.5) == expected


def test_evaluate_refuses_unfrozen_design_on_real_data(world, tmp_path):
    import shutil

    real = tmp_path / "real"
    shutil.copytree(world, real)
    (real / "SYNTHETIC").unlink()                       # pretend it is a real cache
    with pytest.raises(SystemExit, match="frozen design"):
        evaluate.main(["--data-dir", str(real), "--out", str(tmp_path / "o"), "--quick"])
    other = tmp_path / "other.yaml"
    other.write_text(design.DESIGN.read_text(encoding="utf-8").replace("K: 100", "K: 50"), encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen design"):
        evaluate.main(["--data-dir", str(real), "--out", str(tmp_path / "o"), "--config", str(other)])


def test_git_state_reads_the_repo():
    from budget_guard import provenance

    st = provenance.git_state()
    assert len(st["commit"]) == 40 and st["dirty_code"] in (True, False)
    assert "code commit" in provenance.describe(st)


def test_every_output_carries_the_hash(world, tmp_path):
    import json

    import pandas as pd

    out = tmp_path / "r"
    evaluate.main(["--data-dir", str(world), "--out", str(out), "--quick", "--no-sweeps", "--no-figures"])
    h = design.design_hash()
    for name in ("summary", "paired", "soft_frontier", "metric_ranking", "clarify_reference", "schedule"):
        col = pd.read_csv(out / f"{name}.csv")["design_sha256"]
        assert (col == h + "+quick").all(), name
    info = json.loads((out / "run_info.json").read_text())
    assert info["design_sha256"] == h + "+quick"
    assert len(info["code"]["commit"]) == 40
    from budget_guard.report import write_table

    assert info["code"]["commit"] in write_table(out).read_text(encoding="utf-8")
