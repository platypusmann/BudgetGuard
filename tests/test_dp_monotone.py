"""The lemma from proposal section 3.4.3, checked on the real value table: V(n, b) is non-decreasing in b."""
import numpy as np
import pytest

from budget_guard import episodes as ep
from budget_guard.policies import DPOracle, FitContext

from conftest import small_cfg


@pytest.fixture(scope="module")
def ctx_for(envs):
    def make(fee_mult, price=None):
        calib, tune = envs["calib"], envs["tune"]
        if price is not None:
            p = ep.load_prices(price)
            calib, tune = calib.with_prices(p), tune.with_prices(p)
        cfg = small_cfg()
        K = cfg["K"]
        u2 = calib.unit_cost()
        eps = ep.sample_episodes(tune.n, K, 5, 0, "t")
        return FitContext(calib, tune, fee_mult * u2, u2, K, [], eps, eps, cfg, envs["test_qids"])
    return make


@pytest.mark.parametrize("fee_mult, price", [(1.0, None), (0.5, None), (2.0, None), (1.0, (0.0, 0.0)),
                                             (1.0, (0.05, 0.20))])
def test_value_non_decreasing_in_budget(ctx_for, fee_mult, price):
    ctx = ctx_for(fee_mult, price)
    dp = DPOracle()
    dp.fit(ctx)
    for b_mult in (0.01, 0.05, 0.3, 1.5):
        V = dp.value_table(b_mult * ctx.u2 * ctx.K)
        assert np.isfinite(V).all()
        assert (np.diff(V, axis=1) >= -1e-10).all(), f"lemma broken at {b_mult}"
        # sanity: more questions left can never lower the value, and V(0, b) = 0
        assert (V[0] == 0).all()
        assert (np.diff(V, axis=0) >= -1e-10).all()
        # value is bounded by the number of questions
        assert (V[-1] <= ctx.K + 1e-9).all()


def test_value_grows_with_total_budget(ctx_for):
    """Across tables, a bigger B gives at least the value at the same absolute budget."""
    ctx = ctx_for(1.0)
    dp = DPOracle()
    dp.fit(ctx)
    lo, hi = 0.1 * ctx.u2 * ctx.K, 0.4 * ctx.u2 * ctx.K
    v_lo = dp.value_table(lo)[-1, -1]
    v_hi = dp.value_table(hi)[-1, -1]
    assert v_hi >= v_lo - 1e-9


def test_dp_plan_matches_its_own_value(ctx_for, envs):
    """Rolling the DP out on calib should land near what its own table promised."""
    ctx = ctx_for(1.0)
    dp = DPOracle()
    dp.fit(ctx)
    B = 0.3 * ctx.u2 * ctx.K
    promised = dp.value_table(B)[ctx.K, -1] / ctx.K
    eps = ep.sample_episodes(ctx.calib.n, ctx.K, 200, 5, "calib")
    got = ep.run(ctx.calib, dp, eps, B, ctx.fee).correct.mean()
    assert abs(got - promised) < 0.06
