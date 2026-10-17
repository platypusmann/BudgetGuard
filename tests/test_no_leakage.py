"""Rule 3: fit on calib, tune on tune, test only in the final evaluate step."""
import dataclasses
import shutil

import numpy as np
import pytest

from budget_guard import episodes as ep
from budget_guard.leakage import assert_no_test, verify_and_test_qids
from budget_guard.policies import FitContext
from budget_guard.signals import SignalModel


def test_split_pools_are_disjoint(envs):
    calib, tune, test = (set(envs[s].qids) for s in ("calib", "tune", "test"))
    assert not (calib & tune) and not (calib & test) and not (tune & test)
    assert set(envs["test"].qids) <= envs["test_qids"]


def test_verify_catches_edited_split(world, tmp_path):
    d = tmp_path / "splits"
    shutil.copytree(world / "splits", d)
    verify_and_test_qids(d)
    with (d / "test.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write('{"qid": "sneaky"}\n')
    with pytest.raises(RuntimeError):
        verify_and_test_qids(d)


def test_signal_model_refuses_other_splits(envs):
    with pytest.raises(AssertionError):
        SignalModel().fit(envs["tune"], envs["test_qids"])
    with pytest.raises(AssertionError):
        SignalModel().fit(envs["test"], envs["test_qids"])


def _env_with_qids(env, qids):
    new = object.__new__(ep.Env)
    new.__dict__ = dict(env.__dict__)
    new.qids = np.array(qids, dtype=object)
    return new


def test_fit_context_refuses_test_qids(envs):
    calib, tune = envs["calib"], envs["tune"]
    leaked_tune = _env_with_qids(tune, list(tune.qids[:-1]) + [envs["test"].qids[0]])
    eps = ep.sample_episodes(tune.n, 10, 2, 0, "t")
    with pytest.raises(AssertionError, match="test qids"):
        FitContext(calib, leaked_tune, 1.0, 1.0, 10, [1.0], eps, eps, {}, envs["test_qids"])
    leaked_calib = _env_with_qids(calib, list(calib.qids[:-1]) + [envs["test"].qids[0]])
    with pytest.raises(AssertionError, match="test qids"):
        FitContext(leaked_calib, tune, 1.0, 1.0, 10, [1.0], eps, eps, {}, envs["test_qids"])


def test_fit_context_refuses_test_env_as_tune(envs):
    eps = ep.sample_episodes(envs["test"].n, 10, 2, 0, "t")
    with pytest.raises(AssertionError):
        FitContext(envs["calib"], envs["test"], 1.0, 1.0, 10, [1.0], eps, eps, {}, envs["test_qids"])


def test_fitted_policies_never_saw_test(fitted, envs):
    ctx = fitted["ctx"]
    assert ctx.calib.split == "calib" and ctx.tune.split == "tune"
    for f in dataclasses.fields(ctx):
        v = getattr(ctx, f.name)
        if isinstance(v, ep.Env):
            assert_no_test(v.qids, envs["test_qids"], f.name)


def test_assert_no_test_helper():
    assert_no_test(["a", "b"], frozenset({"c"}), "x")
    with pytest.raises(AssertionError):
        assert_no_test(["a", "c"], frozenset({"c"}), "x")


def test_signals_hidden_until_paid(envs):
    env = envs["test"]
    idx = np.arange(4)
    step = ep.Step(env, 0, 10, 1.0, 0.0, idx, np.ones(4), np.zeros((4, 3), bool), 0, np.ones(4, bool))
    assert np.isnan(step.sig("p0")).all()
    step.revealed[:2, 0] = True
    assert np.isfinite(step.sig("p0")[:2]).all() and np.isnan(step.sig("p0")[2:]).all()
    assert np.isnan(step.sig("p1")[:2]).all()            # p1 needs Tier 1 too


def test_policy_view_of_clarify_cost_does_not_leak_ambiguity(envs):
    env = envs["test"]
    idx = np.arange(env.n)
    step = ep.Step(env, 0, 10, 1.0, 0.5, idx, np.ones(env.n), np.zeros((env.n, 3), bool), 0, np.ones(env.n, bool))
    for k, a in enumerate((ep.C0, ep.C1, ep.C2)):
        assert np.allclose(step.wc(a), 0.5 + env.wc[k, 0, idx])  # raw prompt length only
