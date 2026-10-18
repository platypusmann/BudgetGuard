"""Exact DP over (questions left, budget left, signal bin), additive reward.

The SIT320 toy's oracle, moved from a product reward over the steps of one task to a sum of
correct answers over a stream of K questions.

The signal comes in two parts:
  t   a free bin known before paying anything: quantile bin of p_text, the question-text
      ambiguity detector. The number of bins follows the frozen rule in configs/experiment.yaml
      (design.choose_text_bins): fewest bins unless more bins win by at least min_gain_points
      on tune streams. 1 bin means the text signal is ignored. Test is never used.
      One bin when that signal is not available.
  s   a bin revealed by the Tier 0 probe: quantile bin of p0 (conf_bins) crossed with
      quantile bin of p_amb (amb_bins)

Model, all estimated on calib with out-of-fold signals:
  pi[t], pi_s[t, s]   share of questions in t, and in s given t
  p[t, s, a]          P(correct | t, s, a) for a in T0 T1 T2 C0 C1 C2. Shrunk with prior_strength
                      pseudo counts toward the pooled p_s[s, a], which is shrunk toward the action's
                      overall rate, so thin cells stay sane
  c[t, s, a]          mean extra cost of a after the probe, fee included, shrunk the same way
  w[a]                mean worst-case extra cost of a, used for feasibility while planning
  c_probe, w_probe    mean actual and worst-case cost of the Tier 0 probe
  pd, cd [t, a], wd[a]  answering directly with no probe, for when the probe is not worth it
  allowed[t, a]       which final actions a subclass may take in text bin t. All, for the DP itself

Recurrence, with b continuous and V linearly interpolated on a grid of `grid` levels over [0, B]:
  V(0, b) = 0
  Vp(n, b, t) = sum_s pi_s[t, s] max over allowed a with w[a] <= b - c_probe of
                p[t, s, a] + V(n - 1, b - c_probe - c[t, s, a])
  Vd(n, b, t) = max over allowed a with wd[a] <= b of pd[t, a] + V(n - 1, b - cd[t, a])
  V(n, b) = sum_t pi[t] max(V(n - 1, b), Vd(n, b, t), Vp(n, b, t) if w_probe <= b)
The first option is skipping the question. Every option that is feasible at b stays feasible at a
larger b, which is what the monotonicity lemma needs: V(n, b) is non-decreasing in b for every n.
tests/test_dp_monotone.py checks this on the real table.

At run time stage 0 compares skip, probe and every allowed direct answer for the question's text
bin, using its exact worst case (as policies see it) for feasibility. After a probe it picks the
allowed action with the best p[t, s, a] + V(n - 1, left - c[t, s, a]). The runner's mask applies
on top.

Two reference policies reuse this machinery and change only the clarify rule:
  AmbiguityOracleDP  text axis is the true is_ambiguous label, clarify actions only on ambiguous
                     questions and plain answers only on clear ones
  ClarifyAllDP       no text axis, clarify actions only, on every question
"""
from __future__ import annotations

import numpy as np

from ..episodes import C0, C1, C2, P0, SKIP, T0, T1, T2, Env, Step
from ..design import choose_text_bins
from .base import FitContext, Policy

ACTIONS = np.array([T0, T1, T2, C0, C1, C2, SKIP])
DIRECT = ACTIONS[:6]
IS_CLARIFY = np.array([False, False, False, True, True, True, False])


def _interp(V: np.ndarray, x: np.ndarray, delta: float) -> np.ndarray:
    """V on the grid, evaluated at budgets x by linear interpolation. Assumes x >= 0."""
    G = len(V) - 1
    pos = np.clip(x / delta, 0, G) if delta > 0 else np.zeros_like(x)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, G)
    w = pos - lo
    return V[lo] * (1 - w) + V[hi] * w


def _interp_col(M: np.ndarray, x: np.ndarray, t: np.ndarray, delta: float) -> np.ndarray:
    """Column t[i] of M (G+1, T) at budget x[i], for each episode i."""
    G = M.shape[0] - 1
    pos = np.clip(x / delta, 0, G) if delta > 0 else np.zeros_like(x)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, G)
    w = pos - lo
    return M[lo, t] * (1 - w) + M[hi, t] * w


def _shrink(total: np.ndarray, n: np.ndarray, prior: np.ndarray, m: float) -> np.ndarray:
    return (total + m * prior) / (n + m)


class DPOracle(Policy):
    name, label = "dp_oracle", "DP oracle (budget-aware)"
    budget_aware = True

    # ---- hooks the reference policies override

    def _text_calib(self, env: Env, T: int) -> tuple[np.ndarray, int]:
        """Text bin of every calib question, and the number of bins."""
        if T <= 1 or "p_text" not in env.signals:
            self.text_edges = np.array([])
            return np.zeros(env.n, int), 1
        pt = env.signals["p_text"][0]
        self.text_edges = np.quantile(pt, np.linspace(0, 1, T + 1)[1:-1])
        return np.searchsorted(self.text_edges, pt, side="right"), T

    def _text_step(self, step: Step) -> np.ndarray:
        if not len(self.text_edges):
            return np.zeros(len(step.idx), int)
        return np.searchsorted(self.text_edges, np.nan_to_num(step.sig("p_text"), nan=0.0), side="right")

    def _allowed(self, T: int) -> np.ndarray:
        """allowed[t, j] over ACTIONS. SKIP must stay allowed."""
        return np.ones((T, len(ACTIONS)), bool)

    # ---- fitting

    def fit(self, ctx: FitContext) -> None:
        cfg = ctx.cfg["dp"]
        rule = cfg.get("text_bins", {"grid": [1], "min_gain_points": 0.5})
        grid = sorted(int(t) for t in rule["grid"])
        if type(self) is not DPOracle or len(grid) == 1 or "p_text" not in ctx.calib.signals:
            self._fit_tables(ctx, grid[0] if type(self) is DPOracle else 1)
            return
        assert rule.get("select_on", "tune") == "tune" and ctx.tune.split == "tune"
        # mean accuracy over every tuning budget on the tune streams, then the frozen rule
        self.text_bins_scores = {}
        for T in grid:
            self._fit_tables(ctx, T)
            self.text_bins_scores[T] = round(float(np.mean([ctx.tune_score(self, B)[0] for B in ctx.budgets])), 5)
        best = choose_text_bins(self.text_bins_scores, float(rule["min_gain_points"]))
        self._fit_tables(ctx, best)

    def _fit_tables(self, ctx: FitContext, text_bins: int) -> None:
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
        t, T = self._text_calib(env, text_bins)
        self.T = T
        self.allowed = self._allowed(T)
        assert self.allowed[:, -1].all(), "SKIP must stay allowed"
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
        A = len(ACTIONS)
        overall_p, overall_c = correct.mean(axis=0), cost.mean(axis=0)
        # pooled over the text axis first, then each text bin shrinks toward the pooled table
        p_s, c_s = np.zeros((S, A)), np.zeros((S, A))
        for b in range(S):
            sel = s == b
            p_s[b] = _shrink(correct[sel].sum(axis=0), sel.sum(), overall_p, m)
            c_s[b] = _shrink(cost[sel].sum(axis=0), sel.sum(), overall_c, m)
        self.pi = np.array([(t == i).mean() for i in range(T)])
        self.pi_s = np.zeros((T, S))
        self.p, self.c = np.zeros((T, S, A)), np.zeros((T, S, A))
        for i in range(T):
            in_t = t == i
            for b in range(S):
                sel = in_t & (s == b)
                self.pi_s[i, b] = sel.sum() / max(in_t.sum(), 1)
                self.p[i, b] = _shrink(correct[sel].sum(axis=0), sel.sum(), p_s[b], m)
                self.c[i, b] = _shrink(cost[sel].sum(axis=0), sel.sum(), c_s[b], m)
        self.w = wc.mean(axis=0)
        self.c_probe, self.w_probe = float(env.cost[0, 0].mean()), float(env.wc[0, 0].mean())
        # answering with no probe: nothing is revealed, so a clarify always pays for its call
        direct_cost = np.stack([env.cost[0, 0], env.cost[1, 0], env.cost[2, 0], fee + env.cost[0, 1],
                                fee + env.cost[1, 1], fee + env.cost[2, 1]], axis=1)
        direct_wc = np.stack([env.wc[0, 0], env.wc[1, 0], env.wc[2, 0], fee + env.wc[0, 0],
                              fee + env.wc[1, 0], fee + env.wc[2, 0]], axis=1)
        self.pd, self.cd = np.zeros((T, 6)), np.zeros((T, 6))
        for i in range(T):
            sel = t == i
            self.pd[i] = _shrink(correct[sel, :6].sum(axis=0), sel.sum(), overall_p[:6], m)
            self.cd[i] = _shrink(direct_cost[sel].sum(axis=0), sel.sum(), direct_cost.mean(axis=0), m)
        self.wd = direct_wc.mean(axis=0)
        self.tables: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for B in ctx.budgets:
            self.solve(B)

    def bins(self, p0: np.ndarray, pa: np.ndarray) -> np.ndarray:
        cb = np.searchsorted(self.conf_edges, np.nan_to_num(p0, nan=0.0), side="right")
        ab = np.searchsorted(self.amb_edges, np.nan_to_num(pa, nan=0.0), side="right")
        return cb * self.na + ab

    def solve(self, B: float) -> tuple[np.ndarray, np.ndarray]:
        """Fill V (K+1, G+1) and Vp (K+1, G+1, T) for total budget B. Cached per B."""
        key = round(float(B), 15)
        if key in self.tables:
            return self.tables[key]
        G, K, T = self.G, self.K, self.T
        delta = B / G
        b = np.arange(G + 1) * delta
        V = np.zeros((K + 1, G + 1))
        Vp = np.full((K + 1, G + 1, T), -1e9)
        b1 = b - self.c_probe
        probe_ok = self.w_probe <= b + 1e-15
        rem = b1[:, None, None, None] - self.c[None]                                 # (G+1, T, S, A)
        feas = ((self.w[None, None, None, :] <= b1[:, None, None, None] + 1e-15) & (rem >= -1e-15)
                & self.allowed[None, :, None, :])
        rem = np.maximum(rem, 0.0)
        rem_d = b[:, None, None] - self.cd[None]                                     # (G+1, T, 6)
        feas_d = ((self.wd[None, None, :] <= b[:, None, None] + 1e-15) & (rem_d >= -1e-15)
                  & self.allowed[None, :, :6])
        rem_d = np.maximum(rem_d, 0.0)
        for n in range(1, K + 1):
            prev = V[n - 1]
            Q = np.where(feas, self.p[None] + _interp(prev, rem, delta), -np.inf)
            best = Q.max(axis=3)                                                    # (G+1, T, S)
            vp = (np.where(np.isfinite(best), best, 0.0) * self.pi_s[None]).sum(axis=2)  # (G+1, T)
            # -1e9 rather than -inf marks "cannot probe", so interpolation never makes nan
            vp = np.where(probe_ok[:, None], vp, -1e9)
            vd = np.where(feas_d, self.pd[None] + _interp(prev, rem_d, delta), -np.inf).max(axis=2)
            Vp[n] = vp
            per_t = np.maximum(np.maximum(prev[:, None], vd), vp)                   # (G+1, T)
            V[n] = (per_t * self.pi[None]).sum(axis=1)
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
        t = self._text_step(step)
        prev = self.V[n - 1]
        if step.stage == 0:
            # columns: probe, skip, then direct T0 T1 T2 C0 C1 C2. argmax takes the first of
            # equal values, so ties go to probing, then skipping, then cheaper direct answers
            codes = np.array([P0, SKIP, *DIRECT])
            opts = np.full((E, len(codes)), -np.inf)
            opts[:, 0] = np.where(step.feasible(P0), _interp_col(self.Vp[n], step.left, t, self.delta), -np.inf)
            opts[:, 1] = _interp(prev, step.left, self.delta)
            for j in range(6):
                rem = step.left - self.cd[t, j]
                v = self.pd[t, j] + _interp(prev, np.maximum(rem, 0.0), self.delta)
                ok = step.feasible(int(DIRECT[j])) & (rem >= -1e-15) & self.allowed[t, j]
                opts[:, 2 + j] = np.where(ok, v, -np.inf)
            return codes[np.argmax(opts, axis=1)]
        s = self.bins(step.sig("p0"), step.sig("p_amb"))
        Q = np.empty((E, len(ACTIONS)))
        for j, a in enumerate(ACTIONS):
            rem = step.left - self.c[t, s, j]
            q = self.p[t, s, j] + _interp(prev, np.maximum(rem, 0.0), self.delta)
            Q[:, j] = np.where(step.feasible(int(a)) & (rem >= -1e-15) & self.allowed[t, j], q, -np.inf)
        return ACTIONS[np.argmax(Q, axis=1)]


class AmbiguityOracleDP(DPOracle):
    """Reference: clarifies exactly when the question is ambiguous, using the true label.

    Same DP, with the text axis replaced by is_ambiguous and the actions restricted: clarify
    actions only on ambiguous questions, plain answers only on clear ones. Tier choice and budget
    pacing stay optimal for the DP's model. Under a tight budget the runner can still downgrade an
    unaffordable clarify, so "exactly" holds whenever a clarify fits.
    """
    name, label = "ambiguity_oracle", "Ambiguity oracle (clarify iff ambiguous)"

    def _text_calib(self, env, T):
        self.text_edges = np.array([0.5])
        return env.is_amb.astype(int), 2

    def _text_step(self, step):
        return step.env.is_amb[step.idx].astype(int)      # the oracle reads the label on purpose

    def _allowed(self, T):
        allowed = np.ones((2, len(ACTIONS)), bool)
        allowed[0, IS_CLARIFY] = False                      # clear: never clarify
        allowed[1, :6] = IS_CLARIFY[:6]                     # ambiguous: always clarify
        return allowed


class ClarifyAllDP(DPOracle):
    """Reference: clarifies every question, blind to ambiguity. Same DP, clarify actions only."""
    name, label = "clarify_all", "Clarify every question (blind)"

    def _text_calib(self, env, T):
        self.text_edges = np.array([])
        return np.zeros(env.n, int), 1

    def _allowed(self, T):
        allowed = np.ones((1, len(ACTIONS)), bool)
        allowed[0, :6] = IS_CLARIFY[:6]
        return allowed


__all__ = ["DPOracle", "AmbiguityOracleDP", "ClarifyAllDP", "ACTIONS", "C1", "C2", "T1", "T2"]
