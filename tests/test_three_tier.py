"""The three-tier ReDAct-style threshold: hard cap, threshold ordering, tune-only fitting."""
import numpy as np
import pytest

from budget_guard import episodes as ep
from budget_guard.policies import FitContext, ThreeTierThreshold
from budget_guard.policies.threshold import CLAR_RATES, ESC_RATES, ThreeTierRule, three_tier_grid

from conftest import small_cfg


def _get(fitted):
    return next(p for p in fitted["policies"] if p.name == "threshold_three_tier")


def test_masked_never_breaks_the_cap_and_unmasked_would(fitted, envs):
    pol = _get(fitted)
    K, u2 = fitted["K"], fitted["u2"]
    for split in ("tune", "test"):
        env = envs[split]
        eps = ep.sample_episodes(env.n, K, 30, 31, split)
        for B in fitted["budgets"] + [0.0, 0.005 * u2 * K]:
            r = ep.run(env, pol, eps, B, u2)
            assert not r.violation.any() and (r.total_spend <= B * (1 + 1e-9) + 1e-15).all()
    # the guard is what holds it: always escalating to Tier 2 without the mask overspends a tight budget
    eager = ThreeTierRule(-np.inf, -np.inf, np.inf)
    eager.name = "eager"
    eps = ep.sample_episodes(envs["tune"].n, K, 30, 32, "tune")
    assert ep.run(envs["tune"], eager, eps, 0.05 * u2 * K, u2, masked=False).violation.mean() > 0.5


def test_thresholds_are_ordered(fitted):
    pol = _get(fitted)
    assert pol.params and all(t1 <= t2 for t1, t2, _ in pol.params.values())
    with pytest.raises(AssertionError):
        ThreeTierRule(0.6, 0.4, np.inf)


def test_tier_rises_with_uncertainty(envs):
    env = envs["tune"]
    u = 1 - env.signals["p0"][0]
    t1, t2 = float(np.quantile(u, 0.4)), float(np.quantile(u, 0.8))
    pol = ThreeTierRule(t1, t2, np.inf)
    idx = np.arange(env.n)
    revealed = np.zeros((env.n, 3), bool)
    revealed[:, 0] = True
    step = ep.Step(env, 0, 10, 1.0, 0.0, idx, np.ones(env.n), revealed, 1, np.ones(env.n, bool))
    act = pol.decide(step)
    assert set(np.unique(act)) <= {ep.T0, ep.T1, ep.T2}           # no clarify with kappa at infinity
    order = np.argsort(u)
    assert (np.diff(act[order]) >= 0).all()                      # never a lower tier for a higher u
    assert (act[u > t2] == ep.T2).all() and (act[u <= t1] == ep.T0).all()


def test_grid_reuses_the_existing_rate_grids(envs):
    env = envs["calib"]
    grid = three_tier_grid(1 - env.signals["p0"][0], env.signals["p_amb"][0])
    n_pairs = sum(1 for r1 in ESC_RATES for r2 in ESC_RATES if r2 <= r1)
    assert len(grid) == n_pairs * len(CLAR_RATES)
    assert all(t1 <= t2 for t1, t2, _ in grid)
    assert any(t1 == t2 for t1, t2, _ in grid)                    # the two-tier rule is in the grid


def test_fits_on_tune_streams_only(envs, monkeypatch):
    from budget_guard.policies import base

    seen = []
    real = base.run

    def spy(env, *a, **k):
        seen.append(env.split)
        return real(env, *a, **k)
    monkeypatch.setattr(base, "run", spy)
    cfg = small_cfg()
    K = cfg["K"]
    calib, tune = envs["calib"], envs["tune"]
    u2 = calib.unit_cost()
    eps = ep.sample_episodes(tune.n, K, 5, 0, "tune")
    ctx = FitContext(calib, tune, u2, u2, K, [0.2 * u2 * K], eps, eps, cfg, envs["test_qids"])
    ThreeTierThreshold().fit(ctx)
    assert seen and set(seen) == {"tune"}
    with pytest.raises(AssertionError):
        FitContext(calib, envs["test"], u2, u2, K, [1.0], eps, eps, cfg, envs["test_qids"])
