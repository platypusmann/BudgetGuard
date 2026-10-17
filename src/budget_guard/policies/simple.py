"""Floors, ceilings and the signal-blind control."""
from __future__ import annotations

import numpy as np

from ..episodes import T0, Env, Step, run
from .base import FitContext, Policy, nearest, probe_then, threshold_action


class AlwaysTier(Policy):
    """Answer every question with one tier. No clarify, these are single-action reference points."""
    uses_clarify = False

    def __init__(self, k: int):
        self.k = k
        self.name = f"always_tier{k}"
        self.label = f"Always Tier {k}"

    def decide(self, step: Step) -> np.ndarray:
        return np.full(len(step.idx), T0 + self.k)


class RandomMatched(Policy):
    """Probe Tier 0, then escalate and clarify at random, ignoring every signal.

    Clarify rate is copied from the tuned threshold at the same budget, and the escalation rate is
    set by bisection so mean spend on tune matches the tuned threshold's. So any gap to the tuned
    threshold is what the signal is worth at equal money.
    """
    name, label = "random_matched", "Random, matched spend"

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.params: dict[float, tuple[float, float]] = {}
        self.q = (0.0, 0.0)

    def fit(self, ctx: FitContext) -> None:
        ref = ctx.fitted["threshold_tuned"]
        for B in ctx.budgets:
            r = run(ctx.tune, ref, ctx.tune_eps, B, ctx.fee)
            target = r.total_spend.mean()
            q_clar = float(r.clarified.mean())
            lo, hi = 0.0, 1.0
            for _ in range(12):
                mid = (lo + hi) / 2
                probe = RandomMatched(self.seed)
                probe.params = {B: (mid, q_clar)}
                spend = run(ctx.tune, probe, ctx.tune_eps, B, ctx.fee).total_spend.mean()
                lo, hi = (mid, hi) if spend < target else (lo, mid)
            self.params[B] = ((lo + hi) / 2, q_clar)

    def start(self, env: Env, episodes: np.ndarray, B: float, fee: float) -> None:
        super().start(env, episodes, B, fee)
        self.q = nearest(self.params, B)
        rng = np.random.default_rng([self.seed, int(episodes[:, 0].sum()) % (2 ** 31), episodes.shape[0]])
        self.draws = rng.random((episodes.shape[0], episodes.shape[1], 2))

    def decide(self, step: Step) -> np.ndarray:
        def final(st):
            d = self.draws[:, st.t]
            return threshold_action(d[:, 0] < self.q[0], d[:, 1] < self.q[1])
        return probe_then(step, final)
