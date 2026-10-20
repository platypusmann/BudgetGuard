"""The run guard: one evaluation at a time, and a real one only into a fresh folder of its own."""
import json
import os
import shutil

import pytest

from budget_guard import evaluate, runguard


@pytest.fixture()
def lock(tmp_path, monkeypatch):
    path = tmp_path / ".evaluate.lock"
    monkeypatch.setattr(runguard, "LOCK", path)
    monkeypatch.delenv(runguard.TOKEN_ENV, raising=False)
    return path


def test_refuses_output_shared_with_another_run(tmp_path, lock):
    other = tmp_path / "results" / "synthetic"
    other.mkdir(parents=True)
    for out in (other, other / "inner", tmp_path / "results"):          # same, inside, containing
        with pytest.raises(SystemExit, match="shared with"):
            with runguard.guard(out, True, other_dirs=[other]):
                pass
    assert not lock.exists()


def test_refuses_non_empty_output(tmp_path, lock):
    out = tmp_path / "final"
    out.mkdir()
    (out / "left_over.csv").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        with runguard.guard(out, True):
            pass
    assert not lock.exists()


def test_refuses_while_another_run_holds_the_lock(tmp_path, lock):
    lock.write_text(json.dumps({"token": "someone-else", "pid": 4242, "what": "evaluate synthetic", "out": "x"}))
    for real_final in (True, False):                                   # every run respects the lock
        with pytest.raises(SystemExit, match="holds"):
            with runguard.guard(tmp_path / "final", real_final):
                pass
    assert json.loads(lock.read_text())["token"] == "someone-else"     # not ours, left alone


def test_lock_released_on_exit_error_and_ctrl_c(tmp_path, lock):
    with runguard.guard(tmp_path / "a", True):
        assert lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid()
    assert not lock.exists() and runguard.TOKEN_ENV not in os.environ
    for exc in (RuntimeError("boom"), KeyboardInterrupt()):
        with pytest.raises(type(exc)):
            with runguard.guard(tmp_path / "b", True):
                raise exc
        assert not lock.exists()


def test_steps_of_the_same_run_share_its_lock(tmp_path, lock):
    with runguard.guard(tmp_path / "final", True):
        # what scripts/bg.py real does: its evaluate step sees its own token and goes ahead
        with runguard.guard(tmp_path / "final", True):
            assert lock.exists()
        assert lock.exists()                                           # the inner step does not release it
    assert not lock.exists()


def _real_copy(world, tmp_path):
    d = tmp_path / "real"
    shutil.copytree(world, d, ignore=shutil.ignore_patterns("dev"))
    (d / "SYNTHETIC").unlink()
    return d


def test_evaluate_refuses_each_case_before_doing_anything(world, tmp_path, lock, capsys):
    real = _real_copy(world, tmp_path)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "summary.csv").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        evaluate.main(["--data-dir", str(real), "--out", str(busy)])
    with pytest.raises(SystemExit, match="shared with"):
        evaluate.main(["--data-dir", str(real), "--out", str(evaluate.OUT_SYNTHETIC)])
    lock.write_text(json.dumps({"token": "other", "pid": 1, "what": "evaluate synthetic", "out": "y"}))
    with pytest.raises(SystemExit, match="holds"):
        evaluate.main(["--data-dir", str(real), "--out", str(tmp_path / "fresh")])
    # each refusal came before the header, so nothing was read or written
    assert "experiment design" not in capsys.readouterr().out
    assert not (tmp_path / "fresh").exists()


def test_synthetic_runs_take_and_release_the_lock(world, tmp_path, lock):
    evaluate.main(["--data-dir", str(world), "--out", str(tmp_path / "s"), "--quick", "--no-sweeps", "--no-figures"])
    assert not lock.exists()
