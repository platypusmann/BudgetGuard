"""The threshold family, from the SIT320 notebook, on calibrated Tier 0 signals.

All three probe Tier 0, then escalate to Tier 2 when uncertainty u = 1 - p0 is above a threshold
and clarify when the ambiguity belief p_amb is above another. Clarify plus escalate is C2,
clarify alone is C0.

  threshold_fixed   ReDAct reference. Thresholds calibrated once on calib to fixed rates and
                    reused at every budget, fee and price. Never reads the budget.
  threshold_tuned   same rule, thresholds re-tuned on tune for each budget. Knows the budget at
                    tuning time but not during the episode. The pre-registered two-tier reference.
  threshold_three_tier  ReDAct-style with all three tiers: Tier 1 when u is above tau1, Tier 2 when
                    u is above tau2, with tau1 <= tau2, same signals and clarify rule as above.
                    Both thresholds and the clarify threshold tuned on tune per budget, from the
                    same rate grids. Added because on real data Tier 1 is nearly as good as Tier 2
                    at a twentieth of the price, which a two-tier rule can't use.
  slack_schedule    the proposal's schedule. One small table of cuts per slack bucket, where slack
                    is remaining budget per remaining question in u2, fitted once across all
                    budgets to reproduce the DP oracle's decisions on tune streams. Each cut learns
                    its direction, because the strong tier's gain need not grow with Tier 0
                    uncertainty, and the tier rule is fitted separately after a clarify.
"""
from __future__ import annotations

import numpy as np

from ..episodes import C0, C1, C2, P0, T0, T1, T2, Step, run
from .base import FitContext, Policy, grid_search, nearest, probe_then, quantile_thresholds, threshold_action

ESC_RATES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
CLAR_RATES = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]


class ThresholdRule(Policy):
    def __init__(self, tau: float = np.inf, kappa: float = np.inf):
        self.tau, self.kappa = tau, kappa

    def decide(self, step: Step) -> np.ndarray:
        def final(st):
            u = 1 - st.sig("p0")
            return threshold_action(u > self.tau, st.sig("p_amb") > self.kappa)
        return probe_then(step, final)


class FixedThreshold(ThresholdRule):
    name, label = "threshold_fixed", "Fixed threshold (ReDAct-style)"

    def fit(self, ctx: FitContext) -> None:
        c = ctx.cfg["fixed_threshold"]
        u = 1 - ctx.calib.signals["p0"][0]
        pa = ctx.calib.signals["p_amb"][0]
        self.tau = quantile_thresholds(u, [c["escalate_rate"]])[0]
        self.kappa = quantile_thresholds(pa, [c["clarify_rate"]])[0]


class TunedThreshold(ThresholdRule):
    name, label = "threshold_tuned", "Threshold tuned per budget"

    def fit(self, ctx: FitContext) -> None:
        u = 1 - ctx.calib.signals["p0"][0]
        pa = ctx.calib.signals["p_amb"][0]
        grid = [(t, k) for t in quantile_thresholds(u, ESC_RATES) for k in quantile_thresholds(pa, CLAR_RATES)]
        self.params = {}
        for B in ctx.budgets:
            self.params[B], _ = grid_search(ctx, lambda p: ThresholdRule(*p), grid, B)

    def start(self, env, episodes, B, fee) -> None:
        super().start(env, episodes, B, fee)
        self.tau, self.kappa = nearest(self.params, B)


class ThreeTierRule(Policy):
    """Probe Tier 0, then Tier 0, 1 or 2 by uncertainty u = 1 - p0, clarifying when p_amb > kappa."""

    def __init__(self, tau1: float = np.inf, tau2: float = np.inf, kappa: float = np.inf):
        assert tau1 <= tau2, "the Tier 2 threshold can't sit below the Tier 1 threshold"
        self.tau1, self.tau2, self.kappa = tau1, tau2, kappa

    def decide(self, step: Step) -> np.ndarray:
        def final(st):
            u = 1 - st.sig("p0")
            tier = np.where(u > self.tau2, 2, np.where(u > self.tau1, 1, 0))
            return np.where(st.sig("p_amb") > self.kappa, C0 + tier, T0 + tier)
        return probe_then(step, final)


def three_tier_grid(u: np.ndarray, pa: np.ndarray) -> list[tuple[float, float, float]]:
    """Every (tau1, tau2, kappa) from the existing rate grids with tau1 <= tau2.

    r1 is the share of calib questions sent past Tier 0 and r2 <= r1 the share sent on to Tier 2,
    so tau2 = quantile(1 - r2) >= tau1 = quantile(1 - r1). r1 == r2 is the two-tier rule.
    """
    taus = dict(zip(ESC_RATES, quantile_thresholds(u, ESC_RATES)))
    kappas = quantile_thresholds(pa, CLAR_RATES)
    return [(taus[r1], taus[r2], k) for r1 in ESC_RATES for r2 in ESC_RATES if r2 <= r1 for k in kappas]


class ThreeTierThreshold(ThreeTierRule):
    name, label = "threshold_three_tier", "Three-tier threshold tuned per budget"

    def fit(self, ctx: FitContext) -> None:
        u = 1 - ctx.calib.signals["p0"][0]
        pa = ctx.calib.signals["p_amb"][0]
        grid = three_tier_grid(u, pa)
        self.params = {}
        for B in ctx.budgets:
            self.params[B], _ = grid_search(ctx, lambda p: ThreeTierRule(*p), grid, B)

    def start(self, env, episodes, B, fee) -> None:
        super().start(env, episodes, B, fee)
        self.tau1, self.tau2, self.kappa = nearest(self.params, B)


def _p_text(step: Step) -> np.ndarray:
    """The free text ambiguity score, or 0.5 everywhere when the env has none."""
    return step.sig("p_text") if "p_text" in step.env.signals else np.full(len(step.idx), 0.5)


class Recorder:
    """Wraps a policy and keeps what it saw and chose at its final decision."""

    def __init__(self, policy, u2: float, keep_x0: bool = False):
        self.policy, self.u2, self.keep_x0 = policy, u2, keep_x0
        self.name = policy.name
        self.rows: list[dict] = []
        self.rows0: list[dict] = []

    def start(self, *a):
        self.policy.start(*a)

    def decide(self, step: Step):
        act = self.policy.decide(step)
        if step.stage == 0:
            m = step.active
            self.rows0.append({"slack": step.left[m] / (step.n_left * self.u2), "left_frac": step.left[m] / max(step.B, 1e-300),
                               "n_left_frac": np.full(m.sum(), step.n_left / step.K), "p_text": _p_text(step)[m],
                               "action": act[m]})
        if step.stage >= 1:
            m = step.active
            rec = {"slack": step.left[m] / (step.n_left * self.u2), "u": 1 - step.sig("p0")[m],
                   "p_amb": step.sig("p_amb")[m], "left_frac": step.left[m] / max(step.B, 1e-300),
                   "n_left_frac": np.full(m.sum(), step.n_left / step.K), "p_text": _p_text(step)[m],
                   "action": act[m]}
            if self.keep_x0:
                rec["x0"] = step.sig("x0")[m]
            self.rows.append(rec)
        return act

    def table(self) -> dict:
        """Decisions made after the Tier 0 probe."""
        return {k: np.concatenate([r[k] for r in self.rows]) for k in self.rows[0]}

    def table0(self) -> dict:
        """First decision for every question: probe, skip or answer directly."""
        return {k: np.concatenate([r[k] for r in self.rows0]) for k in self.rows0[0]}


def _fit_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Cut (t, sign) minimizing disagreement between the rule and y. The rule fires when x > t
    for sign +1 and when x < t for sign -1, so the direction is learned, not assumed."""
    if len(x) == 0:
        return np.inf, 1
    cands = np.concatenate([[-np.inf, np.inf], np.quantile(x, np.linspace(0.02, 0.98, 49))])
    best, best_err = (np.inf, 1), np.inf
    for sign in (1, -1):
        for t in cands:
            err = np.mean(_fire(x, t, sign) != y)
            if err < best_err - 1e-12:
                best, best_err = (float(t), sign), err
    return best


def _fire(x, t, sign):
    return np.where(sign > 0, x > t, x < t)


def _describe(name, t, sign):
    if not np.isfinite(t):
        return "always" if (t < 0) == (sign > 0) else "never"
    return f"{name} {'>' if sign > 0 else '<'} {t:.3f}"


class SlackSchedule(Policy):
    name, label = "slack_schedule", "Slack threshold schedule"
    budget_aware = True

    def fit(self, ctx: FitContext) -> None:
        dp = ctx.fitted["dp_oracle"]
        self.u2 = ctx.u2
        self.edges = np.asarray(ctx.cfg["slack_edges"], float)
        recs, recs0 = [], []
        for B in ctx.budgets:
            rec = Recorder(dp, ctx.u2)
            run(ctx.tune, rec, ctx.tune_eps, B, ctx.fee)
            recs.append(rec.table())
            recs0.append(rec.table0())
        d = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
        d0 = {k: np.concatenate([r[k] for r in recs0]) for k in recs0[0]}
        # stage 0 per bucket: the DP's most common first move (probe, or answer directly)
        b0 = self._bucket(d0["slack"])
        self.first = np.full(len(self.edges), P0)
        for b in range(len(self.edges)):
            if (b0 == b).sum():
                vals, counts = np.unique(d0["action"][b0 == b], return_counts=True)
                self.first[b] = vals[np.argmax(counts)]
        bucket = self._bucket(d["slack"])
        a = d["action"]
        clar = (a >= C0) & (a <= C2)
        tier = np.where(clar, a - C0, a - T0)
        n_b = len(self.edges)
        never = (np.inf, 1)
        # cuts[b] = {"clar": cut on p_amb, (c, 2): cut on u for Tier 2, (c, 1): cut on u for Tier 1}
        # where c is 1 on the clarify branch and 0 otherwise
        self.cuts = [None] * n_b
        self.support = np.zeros(n_b, int)
        for b in range(n_b):
            m = (bucket == b) & (tier <= 2)
            self.support[b] = m.sum()
            if m.sum() < 20:
                continue
            cut = {"clar": _fit_threshold(d["p_amb"][m], clar[m])}
            for c in (0, 1):
                mc = m & (clar == bool(c))
                if mc.sum() < 10:
                    cut[(c, 2)], cut[(c, 1)] = never, never
                    continue
                cut[(c, 2)] = _fit_threshold(d["u"][mc], tier[mc] == 2)
                m1 = mc & (tier < 2)
                cut[(c, 1)] = _fit_threshold(d["u"][m1], tier[m1] == 1) if m1.sum() else never
            self.cuts[b] = cut
        # thin buckets borrow from the nearest bucket with data, or never escalate if there is none
        good = [b for b in range(n_b) if self.cuts[b] is not None]
        for b in range(n_b):
            if self.cuts[b] is None:
                j = min(good, key=lambda g: abs(g - b)) if good else None
                self.cuts[b] = self.cuts[j] if j is not None else {"clar": never, (0, 2): never, (0, 1): never,
                                                                   (1, 2): never, (1, 1): never}
        self.agreement = float(np.mean(self._rule(bucket, d["u"], d["p_amb"]) == a))

    def _bucket(self, slack):
        return np.clip(np.searchsorted(self.edges, slack, side="right") - 1, 0, len(self.edges) - 1)

    def _rule(self, bucket, u, pa):
        out = np.full(len(u), T0)
        for b in np.unique(bucket):
            m = bucket == b
            cut = self.cuts[b]
            clar = _fire(pa[m], *cut["clar"])
            c = clar.astype(int)
            t2 = np.where(c == 1, _fire(u[m], *cut[(1, 2)]), _fire(u[m], *cut[(0, 2)]))
            t1 = np.where(c == 1, _fire(u[m], *cut[(1, 1)]), _fire(u[m], *cut[(0, 1)]))
            tier = np.where(t2, 2, np.where(t1, 1, 0))
            out[m] = np.where(clar, C0 + tier, T0 + tier)
        return out

    def table(self):
        import pandas as pd

        from ..episodes import NAMES

        rows = []
        for b, edge in enumerate(self.edges):
            cut = self.cuts[b]
            rows.append({"slack_from_u2": edge, "first_move": NAMES[self.first[b]],
                         "clarify_when": _describe("p_amb", *cut["clar"]),
                         "tier2_when": _describe("u", *cut[(0, 2)]), "tier1_when": _describe("u", *cut[(0, 1)]),
                         "after_clarify_tier2_when": _describe("u", *cut[(1, 2)]),
                         "after_clarify_tier1_when": _describe("u", *cut[(1, 1)]),
                         "dp_decisions": int(self.support[b])})
        return pd.DataFrame(rows)

    def decide(self, step: Step) -> np.ndarray:
        bucket = self._bucket(step.left / (step.n_left * self.u2))
        if step.stage == 0:
            return self.first[bucket]
        return self._rule(bucket, 1 - step.sig("p0"), step.sig("p_amb"))


__all__ = ["FixedThreshold", "TunedThreshold", "ThreeTierThreshold", "ThreeTierRule", "SlackSchedule", "ThresholdRule", "Recorder", "T1", "T2", "C1"]
