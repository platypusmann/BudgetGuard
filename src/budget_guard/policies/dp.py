"""Exact DP over (questions left, budget left, signal bin), additive reward.

The SIT320 toy's oracle, moved from a product reward over the steps of one task to a sum of
correct answers over a stream of K questions.

Model, all estimated on calib with out-of-fold signals:
  signal bin s    quantile bin of p0 (conf_bins) crossed with quantile bin of p_amb (amb_bins)
  pi[s]           share of questions in bin s
  p[s, a]         P(correct | s, a) for a in T0 T1 T2 C0 C1 C2, shrunk toward the action's overall
                  rate with prior_strength pseudo counts so thin bins stay sane
  c[s, a]         mean extra cost of a after the Tier 0 probe, fee included
  w[a]            mean worst-case extra cost of a, used for feasibility while planning
  c_probe, w_probe  mean actual and worst-case cost of the Tier 0 probe
  pd[a], cd[a], wd[a]  success rate, mean cost and mean worst case of answering directly with no
                  probe, for when the probe itself is not worth its price

Recurrence, with b continuous and V linearly interpolated on a grid of `grid` levels over [0, B]:
  V(0, b) = 0
  Vp(n, b) = sum_s pi[s] max over a with w[a] <= b - c_probe of p[s, a] + V(n - 1, b - c_probe - c[s, a])
  Vd(n, b) = max over a with wd[a] <= b of pd[a] + V(n - 1, b - cd[a])
  V(n, b) = max(V(n - 1, b), Vd(n, b), Vp(n, b) if w_probe <= b)
The first option is skipping the question. Every option that is feasible at b stays feasible at a
larger b, which is what the monotonicity lemma needs: V(n, b) is non-decreasing in b for every n.
tests/test_dp_monotone.py checks this on the real table.

At run time stage 0 compares skip, probe and every direct answer with the same formulas, using
each question's exact worst case (as policies see it) for feasibility. After a probe it picks the
action with the best p[s, a] + V(n - 1, left - c[s, a]) among feasible actions. The runner's mask
still applies on top.
"""
from __future__ import annotations

import numpy as np

from ..episodes import C0, C1, C2, P0, SKIP, T0, T1, T2, Env, Step
from .base import FitContext, Policy

ACTIONS = np.array([T0, T1, T2, C0, C1, C2, SKIP])


def _interp(V: np.ndarray, x: np.ndarray, delta: float) -> np.ndarray:
    """V on the grid, evaluated at budgets x by linear interpolation. Assumes x >= 0."""
    G = len(V) - 1
    pos = np.clip(x / delta, 0, G) if delta > 0 else np.zeros_like(x)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, G)
    w = pos - lo
    return V[lo] * (1 - w) + V[hi] * w


class DPOracle(Policy):
    name, label = "dp_oracle", "DP oracle (budget-aware)"
    budget_aware = True

    def fit(self, ctx: FitContext) -> None:
        cfg = ctx.cfg["dp"]
        env, fee = ctx.calib, ctx.fee
        self.K, self.G, self.fee = ctx.K, int(cfg["grid"]), fee
        p0, pa = env.signals["p0"][0], env.signals["p_amb"][0]
        nc, na = int(cfg["conf_bins"]), int(cfg["amb_bins"])
        self.conf_edges = np.quantile(p0, np.linspace(0, 1, nc + 1)[1:-1])
        self.amb_edges = np.quantile(pa, np.linspace(0, 1, na + 1)[1:-1])
        self.na = na
        s = self.bins(p0, pa)
        S = nc * na
        amb = env.is_amb
        # outcome and cost of every final action for every calib question, after the probe
        correct = np.stack([env.correct[0, 0], env.correct[1, 0], env.correct[2, 0],
                            env.correct[0, 1], env.correct[1, 1], env.correct[2, 1],
                            np.zeros(env.n, bool)], axis=1).astype(float)
        cost = np.stack([np.zeros(env.n), env.cost[1, 0], env.cost[2, 0],
                         fee + np.where(amb, env.cost[0, 1], 0.0),
                         fee + np.where(amb, env.cost[1, 1], env.cost[1, 0]),
                         fee + np.where(amb, env.cost[2, 1], env.cost[2, 0]),
                         np.zeros(env.n)], axis=1)
        wc = np.stack([np.zeros(env.n), env.wc[1, 0], env.wc[2, 0], fee + env.wc[0, 0],
                       fee + env.wc[1, 0], fee + env.wc[2, 0], np.zeros(env.n)], axis=1)
        m = float(cfg["prior_strength"])
        overall_p, overall_c = correct.mean(axis=0), cost.mean(axis=0)
        self.pi = np.zeros(S)
        self.p = np.zeros((S, len(ACTIONS)))
        self.c = np.zeros((S, len(ACTIONS)))
        for b in range(S):
            sel = s == b
            n = sel.sum()
            self.pi[b] = n / env.n
            self.p[b] = (correct[sel].sum(axis=0) + m * overall_p) / (n + m)
            self.c[b] = (cost[sel].sum(axis=0) + m * overall_c) / (n + m)
        self.w = wc.mean(axis=0)
        self.c_probe, self.w_probe = float(env.cost[0, 0].mean()), float(env.wc[0, 0].mean())
        # answering with no probe: nothing is revealed, so a clarify always pays for its call
        direct_cost = np.stack([env.cost[0, 0], env.cost[1, 0], env.cost[2, 0], fee + env.cost[0, 1],
                                fee + env.cost[1, 1], fee + env.cost[2, 1]], axis=1)
        direct_wc = np.stack([env.wc[0, 0], env.wc[1, 0], env.wc[2, 0], fee + env.wc[0, 0],
                              fee + env.wc[1, 0], fee + env.wc[2, 0]], axis=1)
        self.pd, self.cd, self.wd = correct[:, :6].mean(axis=0), direct_cost.mean(axis=0), direct_wc.mean(axis=0)
        self.tables: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for B in ctx.budgets:
            self.solve(B)

    def bins(self, p0: np.ndarray, pa: np.ndarray) -> np.ndarray:
        cb = np.searchsorted(self.conf_edges, np.nan_to_num(p0, nan=0.0), side="right")
        ab = np.searchsorted(self.amb_edges, np.nan_to_num(pa, nan=0.0), side="right")
        return cb * self.na + ab

    def solve(self, B: float) -> tuple[np.ndarray, np.ndarray]:
        """Fill V (K+1, G+1) and Vp (K+1, G+1) for total budget B. Cached per B."""
        key = round(float(B), 15)
        if key in self.tables:
            return self.tables[key]
        G, K = self.G, self.K
        delta = B / G
        b = np.arange(G + 1) * delta
        V = np.zeros((K + 1, G + 1))
        Vp = np.full((K + 1, G + 1), -1e9)
        b1 = b - self.c_probe
        probe_ok = self.w_probe <= b + 1e-15
        rem = b1[:, None, None] - self.c[None, :, :]                       # (G+1, S, A)
        feas = (self.w[None, None, :] <= b1[:, None, None] + 1e-15) & (rem >= -1e-15)
        rem = np.maximum(rem, 0.0)
        rem_d = b[:, None] - self.cd[None, :]                              # (G+1, 6)
        feas_d = (self.wd[None, :] <= b[:, None] + 1e-15) & (rem_d >= -1e-15)
        rem_d = np.maximum(rem_d, 0.0)
        for n in range(1, K + 1):
            prev = V[n - 1]
            Q = self.p[None] + _interp(prev, rem, delta)
            Q = np.where(feas, Q, -np.inf)
            best = Q.max(axis=2)                                           # (G+1, S)
            # -1e9 rather than -inf marks "cannot probe", so interpolation never makes nan
            vp = np.where(probe_ok, (np.where(np.isfinite(best), best, 0.0) * self.pi[None]).sum(axis=1), -1e9)
            vd = np.where(feas_d, self.pd[None] + _interp(prev, rem_d, delta), -np.inf).max(axis=1)
            Vp[n] = vp
            V[n] = np.maximum(np.maximum(prev, vd), vp)
        self.tables[key] = (V, Vp)
        return V, Vp

    def value_table(self, B: float) -> np.ndarray:
        return self.solve(B)[0]

    def start(self, env: Env, episodes: np.ndarray, B: float, fee: float) -> None:
        super().start(env, episodes, B, fee)
        assert abs(fee - self.fee) < 1e-15, "DP was fitted for another clarify fee"
        self.V, self.Vp = self.solve(B)
        self.delta = B / self.G

    def decide(self, step: Step) -> np.ndarray:
        E = len(step.idx)
        n = step.n_left
        if step.stage == 0:
            prev = self.V[n - 1]
            # columns: probe, skip, then direct T0 T1 T2 C0 C1 C2. argmax takes the first of
            # equal values, so ties go to probing, then skipping, then cheaper direct answers
            codes = np.array([P0, SKIP, T0, T1, T2, C0, C1, C2])
            opts = np.full((E, len(codes)), -np.inf)
            opts[:, 0] = np.where(step.feasible(P0), _interp(self.Vp[n], step.left, self.delta), -np.inf)
            opts[:, 1] = _interp(prev, step.left, self.delta)
            for j in range(6):
                rem = step.left - self.cd[j]
                v = self.pd[j] + _interp(prev, np.maximum(rem, 0.0), self.delta)
                opts[:, 2 + j] = np.where(step.feasible(int(ACTIONS[j])) & (rem >= -1e-15), v, -np.inf)
            return codes[np.argmax(opts, axis=1)]
        s = self.bins(step.sig("p0"), step.sig("p_amb"))
        prev = self.V[n - 1]
        Q = np.empty((E, len(ACTIONS)))
        for j, a in enumerate(ACTIONS):
            rem = step.left - self.c[s, j]
            Q[:, j] = self.p[s, j] + _interp(prev, np.maximum(rem, 0.0), self.delta)
            Q[:, j] = np.where(step.feasible(int(a)) & (rem >= -1e-15), Q[:, j], -np.inf)
        return ACTIONS[np.argmax(Q, axis=1)]


__all__ = ["DPOracle", "ACTIONS", "C1", "C2", "T1", "T2"]
