"""Hindsight oracle routing, the upper bound.

Knows, for every question in the episode, which actions answer it correctly and what each costs,
and pays no probe. Reward is one per correct answer, so the best plan is: for each question find
the cheapest correct action (Tk on the raw prompt or Ck), then take questions in order of that
cost until the budget is used up. That greedy is exactly optimal because every item is worth the
same. Spend never exceeds B by construction.
"""
from __future__ import annotations

import numpy as np

from ..episodes import C0, SKIP, T0, Env, RunResult
from .base import Policy


class HindsightOracle(Policy):
    name, label = "hindsight_oracle", "Hindsight oracle (upper bound)"
    budget_aware = True
    is_oracle = True

    def run(self, env: Env, episodes: np.ndarray, B: float, fee: float) -> RunResult:
        E, K = episodes.shape
        idx = episodes
        acts = [T0 + k for k in range(3)] + [C0 + k for k in range(3)]
        cost = np.stack([env.cost[k, 0][idx] for k in range(3)] + [fee + env.cost[k, 1][idx] for k in range(3)], -1)
        ok = np.stack([env.correct[k, 0][idx] for k in range(3)] + [env.correct[k, 1][idx] for k in range(3)], -1)
        c = np.where(ok, cost, np.inf)
        best = c.argmin(axis=-1)
        cheapest = c.min(axis=-1)                                  # (E, K), inf if nothing is correct
        order = np.argsort(cheapest, axis=1, kind="stable")
        sorted_c = np.take_along_axis(cheapest, order, axis=1)
        take_sorted = np.cumsum(np.where(np.isfinite(sorted_c), sorted_c, np.inf), axis=1) <= B
        take = np.zeros((E, K), bool)
        np.put_along_axis(take, order, take_sorted, axis=1)
        final = np.where(take, np.array(acts)[best], SKIP).astype(np.int8)
        spend = np.where(take, cheapest, 0.0)
        before = B - np.concatenate([np.zeros((E, 1)), np.cumsum(spend, axis=1)[:, :-1]], axis=1)
        return RunResult(self.name, float(B), float(fee), True, episodes, correct=take, spend=spend, final=final,
                         probes=np.zeros((E, K), np.int8), downgraded=np.zeros((E, K), bool), budget_before=before)
