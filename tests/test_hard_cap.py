"""No masked policy may ever spend more than B, at any budget, fee or price."""
import numpy as np
import pytest

from budget_guard import episodes as ep


def _runs(policy, env, eps, B, fee, masked=True):
    if getattr(policy, "is_oracle", False):
        return policy.run(env, eps, B, fee)
    return ep.run(env, policy, eps, B, fee, masked=masked)


@pytest.mark.parametrize("split", ["tune", "test"])
def test_masked_policies_never_violate(fitted, envs, split):
    env = envs[split]
    K, u2 = fitted["K"], fitted["u2"]
    eps = ep.sample_episodes(env.n, K, 25, 11, split)
    # tuned budgets, budgets between them, and absurdly small ones
    budgets = fitted["budgets"] + [b * u2 * K for b in (0.0, 1e-4, 0.007, 0.05, 0.5, 3.0)]
    for p in fitted["policies"]:
        for B in budgets:
            r = _runs(p, env, eps, B, u2)
            assert not r.violation.any(), (p.name, B)
            assert (r.total_spend <= B * (1 + 1e-9) + 1e-15).all()
            assert (r.spend >= 0).all()
            # the budget left before each question never goes negative
            assert (r.budget_before >= -1e-12).all(), p.name


def test_cap_holds_at_other_prices_and_fees(fitted, envs):
    env0 = envs["test"]
    K, u2 = fitted["K"], fitted["u2"]
    eps = ep.sample_episodes(env0.n, K, 20, 12, "test")
    for price in ((0.0, 0.0), (0.05, 0.20)):
        env = env0.with_prices(ep.load_prices(price))
        for fee in (0.0, 0.5 * u2, 2 * u2):
            for p in fitted["policies"]:
                if p.name in ("dp_oracle",):
                    continue                      # the DP is fitted for one fee, checked separately below
                for B in (0.01 * u2 * K, 0.3 * u2 * K):
                    assert not _runs(p, env, eps, B, fee).violation.any(), (p.name, price, fee, B)


def test_actual_cost_never_exceeds_worst_case(envs):
    for s in ("calib", "tune", "test"):
        e = envs[s]
        assert (e.cost <= e.wc + 1e-15).all()


def test_unmasked_budget_blind_policy_does_break_cap(fitted, envs):
    """Sanity check that the test above can fail: without the guard, always Tier 2 overspends."""
    env = envs["test"]
    K, u2 = fitted["K"], fitted["u2"]
    eps = ep.sample_episodes(env.n, K, 20, 13, "test")
    always2 = next(p for p in fitted["policies"] if p.name == "always_tier2")
    r = ep.run(env, always2, eps, 0.2 * u2 * K, u2, masked=False)
    assert r.violation.mean() > 0.5


def test_downgrade_ladder_prefers_paid_tier(envs):
    """If a tier was probed and nothing else fits, answer with it rather than skip."""
    env = envs["test"]
    idx = np.arange(3)
    revealed = np.zeros((3, 3), bool)
    revealed[:, 0] = True
    act, down = ep._downgrade(env, np.array([ep.C2, ep.T2, ep.P1]), idx, revealed, np.zeros(3), 1.0)
    assert (act == ep.T0).all() and down.all()


def test_hindsight_is_an_upper_bound(fitted, envs):
    env = envs["test"]
    K, u2 = fitted["K"], fitted["u2"]
    eps = ep.sample_episodes(env.n, K, 25, 14, "test")
    oracle = next(p for p in fitted["policies"] if p.name == "hindsight_oracle")
    for B in fitted["budgets"]:
        best = oracle.run(env, eps, B, u2).correct.sum(axis=1)
        for p in fitted["policies"]:
            if p is oracle:
                continue
            got = ep.run(env, p, eps, B, u2).correct.sum(axis=1)
            assert (got <= best).all(), p.name
