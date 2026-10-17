"""Policy interface and the context every policy is fitted in.

A policy is fitted once per (clarify fee, price sheet) configuration with a FitContext, which only
holds calib and tune environments. Test never reaches a fit method. Budget-specific tuning is done
inside fit for every budget in ctx.budgets, and start() picks the parameters for the budget it is
run at.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..episodes import P0, SKIP, T0, Env, RunResult, Step, run
from ..leakage import assert_no_test


@dataclass
class FitContext:
    calib: Env                  # signals out of fold
    tune: Env
    fee: float                  # clarify fee, USD
    u2: float                   # mean Tier 2 raw-prompt cost on calib, the budget unit
    K: int
    budgets: list[float]        # total episode budgets B, USD, to tune for
    tune_eps: np.ndarray        # streams over the tune pool
    fit_eps: np.ndarray         # streams over the calib pool
    cfg: dict
    test_qids: frozenset
    fitted: dict = field(default_factory=dict)  # name -> already fitted policy, for dependencies

    def __post_init__(self):
        assert self.calib.split == "calib" and self.tune.split == "tune"
        assert_no_test(self.calib.qids, self.test_qids, "calib pool")
        assert_no_test(self.tune.qids, self.test_qids, "tune pool")

    def tune_score(self, policy, B: float) -> tuple[float, float]:
        """Accuracy on the tune streams at budget B (masked), and mean spend as a tie breaker."""
        r = run(self.tune, policy, self.tune_eps, B, self.fee)
        return float(r.correct.mean()), float(r.total_spend.mean())


class Policy:
    name = "base"
    label = "base"
    budget_aware = False       # reads remaining budget when it decides
    uses_clarify = True

    def fit(self, ctx: FitContext) -> None:
        pass

    def start(self, env: Env, episodes: np.ndarray, B: float, fee: float) -> None:
        self.B, self.fee = B, fee

    def decide(self, step: Step) -> np.ndarray:
        raise NotImplementedError


def nearest(params: dict, B: float):
    """Parameters tuned for the budget closest to B (relative distance)."""
    if not params:
        raise RuntimeError("policy was not fitted")
    key = min(params, key=lambda b: abs(np.log(b / B)) if b > 0 and B > 0 else abs(b - B))
    return params[key]


def grid_search(ctx: FitContext, make, grid: list, B: float):
    """Best parameters on the tune streams. Ties go to the cheaper setting."""
    best, best_key = None, None
    for params in grid:
        acc, spend = ctx.tune_score(make(params), B)
        key = (round(acc, 6), -spend)
        if best_key is None or key > best_key:
            best, best_key = params, key
    return best, best_key[0]


def quantile_thresholds(values: np.ndarray, rates) -> list[float]:
    """Thresholds so that a share `rate` of values lies strictly above. 0 means never, 1 always."""
    out = []
    for r in rates:
        if r <= 0:
            out.append(np.inf)
        elif r >= 1:
            out.append(-np.inf)
        else:
            out.append(float(np.quantile(values, 1 - r)))
    return out


def probe_then(step: Step, final_fn) -> np.ndarray:
    """Common shape: probe Tier 0 at stage 0, then a final decision from Tier 0 signals."""
    if step.stage == 0:
        return np.full(len(step.idx), P0)
    return final_fn(step)


def threshold_action(esc: np.ndarray, clar: np.ndarray) -> np.ndarray:
    """Two-model rule shared by the threshold family: escalate means Tier 2, else Tier 0."""
    from ..episodes import C0, C2, T2

    return np.where(clar, np.where(esc, C2, C0), np.where(esc, T2, T0))


__all__ = ["FitContext", "Policy", "nearest", "grid_search", "quantile_thresholds", "probe_then",
           "threshold_action", "RunResult", "SKIP"]
