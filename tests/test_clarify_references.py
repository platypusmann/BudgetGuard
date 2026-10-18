"""The two clarify reference policies and the free text signal in the DP."""
import numpy as np

from budget_guard import episodes as ep


def _get(fitted, name):
    return next(p for p in fitted["policies"] if p.name == name)


def test_ambiguity_oracle_clarifies_only_ambiguous(fitted, envs):
    env = envs["test"]
    K, u2 = fitted["K"], fitted["u2"]
    eps = ep.sample_episodes(env.n, K, 30, 21, "test")
    oracle = _get(fitted, "ambiguity_oracle")
    for B in fitted["budgets"]:
        r = ep.run(env, oracle, eps, B, u2)
        amb = env.is_amb[r.episodes]
        assert not r.clarified[~amb].any()                 # never clarifies a clear question
    # with a loose budget it clarifies almost every ambiguous question
    r = ep.run(env, oracle, eps, 3.0 * u2 * K, u2)
    assert r.clarified[env.is_amb[r.episodes]].mean() > 0.9


def test_clarify_all_never_answers_without_clarifying_when_it_can(fitted, envs):
    env = envs["test"]
    K, u2 = fitted["K"], fitted["u2"]
    eps = ep.sample_episodes(env.n, K, 30, 22, "test")
    blind = _get(fitted, "clarify_all")
    r = ep.run(env, blind, eps, 3.0 * u2 * K, u2)
    answered = r.final != ep.SKIP
    # every plain answer must come from the runner downgrading an unaffordable clarify
    assert (r.clarified | ~answered | r.downgraded).all()
    assert r.clarified.mean() > 0.9


def test_dp_uses_free_text_bins(fitted, envs):
    dp = _get(fitted, "dp_oracle")
    assert "p_text" in envs["calib"].signals and envs["calib"].signals["p_text"][1] == ()
    assert set(dp.text_bins_scores) == {1, 2, 3}             # picked on tune
    from budget_guard.design import choose_text_bins

    assert dp.T == choose_text_bins(dp.text_bins_scores, 0.5)
    assert dp.pi.shape == (dp.T,) and np.isclose(dp.pi.sum(), 1)
    assert dp.p.shape[0] == dp.T and np.allclose(dp.pi_s.sum(axis=1), 1)


def test_learned_router_sees_p_text(fitted):
    router = _get(fitted, "learned_router")
    n_in = router.model[0].n_features_in_
    x0_dim = fitted["ctx"].calib.signals["x0"][0].shape[1]
    assert n_in == x0_dim + 6                                # p0, p_amb, p_text and three budget inputs
