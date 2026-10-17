"""Shared fixtures. Everything is offline: a small synthetic world built in a temp dir."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def world(tmp_path_factory) -> Path:
    """600 train and 200 validation synthetic questions, graded, with features and hash embeddings."""
    from budget_guard import features, grade, synthetic

    d = tmp_path_factory.mktemp("world")
    synthetic.build_world(d, n_train=600, n_val=200, seed=3, pilot_n=60)
    grade.main(["--data-dir", str(d)])
    features.main(["--data-dir", str(d), "--embed", "hash"])
    return d


@pytest.fixture(scope="session")
def envs(world):
    """calib, tune and test envs with signals, at default prices."""
    from budget_guard import episodes as ep
    from budget_guard.features import load_embeddings, load_features
    from budget_guard.leakage import verify_and_test_qids
    from budget_guard.signals import SignalModel

    test_qids = verify_and_test_qids(world / "splits")
    f = load_features(world)
    emb = load_embeddings(world, "hash")
    pr = ep.load_prices()
    out = {s: ep.Env(f, s, pr, emb=emb, verbose=False) for s in ("calib", "tune", "test")}
    sm = SignalModel().fit(out["calib"], test_qids)
    for e in out.values():
        sm.apply(e)
    out["test_qids"] = test_qids
    return out


def small_cfg():
    from budget_guard import episodes as ep

    cfg = ep.load_experiment()
    cfg.update(K=20, tune_episodes=15, fit_episodes=8)
    cfg["dp"] = {**cfg["dp"], "grid": 200}
    return cfg


@pytest.fixture(scope="session")
def fitted(envs):
    """Every policy fitted at three budgets, default fee."""
    from budget_guard import episodes as ep
    from budget_guard.policies import FitContext, build_all, fit_all

    cfg = small_cfg()
    K = cfg["K"]
    calib, tune = envs["calib"], envs["tune"]
    u2 = calib.unit_cost()
    budgets = [b * u2 * K for b in (0.02, 0.2, 1.0)]
    ctx = FitContext(calib, tune, u2, u2, K, budgets, ep.sample_episodes(tune.n, K, 15, 1000, "tune"),
                     ep.sample_episodes(calib.n, K, 8, 2000, "calib"), cfg, envs["test_qids"])
    pols = fit_all(build_all(), ctx, verbose=False)
    return {"ctx": ctx, "policies": pols, "budgets": budgets, "u2": u2, "K": K}


os.environ.setdefault("PYTHONHASHSEED", "0")
