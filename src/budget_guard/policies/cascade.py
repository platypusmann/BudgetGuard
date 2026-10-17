"""FrugalGPT-style confidence cascade with a clarify gate.

Probe Tier 0. If the ambiguity belief is high, clarify and let Tier 2 answer. Otherwise accept
Tier 0 when its calibrated confidence p0 clears tau0, else pay for Tier 1, accept it when p1 clears
tau1, else go to Tier 2. Thresholds are tuned on tune for each budget, which is how FrugalGPT
fits its cascade to a budget, but the rule never reads the remaining budget during an episode.
"""
from __future__ import annotations

import numpy as np

from ..episodes import C2, P1, T0, T1, T2, Step
from .base import FitContext, Policy, grid_search, nearest, probe_then, quantile_thresholds

ACCEPT0 = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
ACCEPT1 = [1.0, 0.7, 0.4, 0.0]
CLAR = [0.0, 0.1, 0.25]


class CascadeRule(Policy):
    def __init__(self, tau0=-np.inf, tau1=-np.inf, kappa=np.inf):
        self.tau0, self.tau1, self.kappa = tau0, tau1, kappa

    def decide(self, step: Step) -> np.ndarray:
        if step.stage <= 1:
            def final(st):
                p0 = st.sig("p0")
                return np.where(st.sig("p_amb") > self.kappa, C2, np.where(p0 >= self.tau0, T0, P1))
            return probe_then(step, final)
        # stage 2: Tier 1 was paid for
        return np.where(step.sig("p1") >= self.tau1, T1, T2)


class FrugalCascade(CascadeRule):
    name, label = "cascade_frugal", "FrugalGPT-style cascade"

    def fit(self, ctx: FitContext) -> None:
        sig = ctx.calib.signals
        p0, p1, pa = sig["p0"][0], sig["p1"][0], sig["p_amb"][0]
        # accept the top `rate` share by confidence, so the threshold is the (1 - rate) quantile
        t0 = [-np.inf if r >= 1 else np.inf if r <= 0 else float(np.quantile(p0, 1 - r)) for r in ACCEPT0]
        t1 = [-np.inf if r >= 1 else np.inf if r <= 0 else float(np.quantile(p1, 1 - r)) for r in ACCEPT1]
        ks = quantile_thresholds(pa, CLAR)
        grid = [(a, b, c) for a in t0 for b in t1 for c in ks]
        self.params = {}
        for B in ctx.budgets:
            self.params[B], _ = grid_search(ctx, lambda p: CascadeRule(*p), grid, B)

    def start(self, env, episodes, B, fee) -> None:
        super().start(env, episodes, B, fee)
        self.tau0, self.tau1, self.kappa = nearest(self.params, B)
