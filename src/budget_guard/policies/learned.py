"""Budget-conditioned learned router and the RouteLLM-style baseline.

learned_router follows the proposal's section 3.2. A small model takes the Tier 0 features, the
calibrated p0 and p_amb, log remaining budget per remaining question (in u2), remaining budget as a
share of B and remaining questions as a share of K, and outputs a distribution over the final
actions. It is trained by behaviour cloning on the DP oracle's decisions on calib streams at every
budget in the grid, so one model covers all budgets. The model class (multinomial logistic or a
one hidden layer MLP) is picked on tune. The PPO-Lagrangian fine-tune from the proposal is out of
scope here (see DECISIONS.md), and masking keeps the cap hard regardless.

routellm is a question-only router in the spirit of RouteLLM. Logistic regression on the question
embedding predicts whether Tier 2 beats Tier 0 on the raw prompt and whether the question is
ambiguous. It never probes, so it pays nothing for Tier 0 signals. Thresholds are tuned on tune
per budget.
"""
from __future__ import annotations

import numpy as np

from ..episodes import C0, C1, C2, P0, SKIP, T0, T1, T2, Env, Step, run
from .base import FitContext, Policy, grid_search, nearest, quantile_thresholds, threshold_action
from .threshold import Recorder

FINAL_ACTIONS = np.array([T0, T1, T2, C0, C1, C2, SKIP])


def router_features(x0: np.ndarray, p0, pa, slack, left_frac, n_left_frac) -> np.ndarray:
    return np.column_stack([x0, p0, pa, np.log(np.maximum(slack, 0) + 1e-3), left_frac, n_left_frac])


class LearnedRouter(Policy):
    name, label = "learned_router", "Learned router (BC from DP)"
    budget_aware = True

    def _candidates(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
            "mlp32": make_pipeline(StandardScaler(), MLPClassifier((32,), alpha=1e-3, max_iter=300,
                                                                   early_stopping=True, random_state=0)),
        }

    def fit(self, ctx: FitContext) -> None:
        dp = ctx.fitted["dp_oracle"]
        self.u2 = ctx.u2
        tabs, tabs0 = [], []
        for B in ctx.budgets:
            rec = Recorder(dp, ctx.u2, keep_x0=True)
            run(ctx.calib, rec, ctx.fit_eps, B, ctx.fee)
            tabs.append(rec.table())
            tabs0.append(rec.table0())
        d = {k: np.concatenate([t[k] for t in tabs]) for k in tabs[0]}
        d0 = {k: np.concatenate([t[k] for t in tabs0]) for k in tabs0[0]}
        self._fit_first_move(d0)
        X = router_features(d["x0"], 1 - d["u"], d["p_amb"], d["slack"], d["left_frac"], d["n_left_frac"])
        y = d["action"]
        self.n_train = len(y)
        mids = sorted(ctx.budgets)[1::max(1, len(ctx.budgets) // 3)][:3] or list(ctx.budgets)
        self.scores = {}
        best, best_score = None, -1.0
        for name, model in self._candidates().items():
            model.fit(X, y)
            self.model, self.classes = model, model.classes_
            score = float(np.mean([ctx.tune_score(self, B)[0] for B in mids]))
            self.scores[name] = round(score, 4)
            if score > best_score:
                best, best_score = (name, model), score
        self.model_name, self.model = best
        self.classes = self.model.classes_

    def _fit_first_move(self, d0: dict) -> None:
        """Stage 0 head: probe, skip or answer directly, from budget features only."""
        from sklearn.linear_model import LogisticRegression

        y0 = d0["action"]
        self.first_const = int(y0[0]) if len(np.unique(y0)) == 1 else None
        if self.first_const is None:
            X0 = np.column_stack([np.log(np.maximum(d0["slack"], 0) + 1e-3), d0["left_frac"], d0["n_left_frac"]])
            self.first_model = LogisticRegression(C=1.0, max_iter=3000).fit(X0, y0)

    def _first_move(self, step: Step) -> np.ndarray:
        E = len(step.idx)
        if self.first_const is not None:
            return np.full(E, self.first_const)
        X0 = np.column_stack([np.log(np.maximum(step.left / (step.n_left * self.u2), 0) + 1e-3),
                              step.left / max(step.B, 1e-300), np.full(E, step.n_left / step.K)])
        proba = self.first_model.predict_proba(X0)
        cls = self.first_model.classes_
        score = np.where(np.column_stack([step.feasible(int(a)) for a in cls]), proba, -np.inf)
        return cls[np.argmax(score, axis=1)]

    def decide(self, step: Step) -> np.ndarray:
        if step.stage == 0:
            return self._first_move(step)
        p0, pa = step.sig("p0"), step.sig("p_amb")
        X = router_features(np.nan_to_num(step.sig("x0")), np.nan_to_num(p0), np.nan_to_num(pa),
                            step.left / (step.n_left * self.u2), step.left / max(step.B, 1e-300),
                            np.full(len(step.idx), step.n_left / step.K))
        proba = self.model.predict_proba(X)
        score = np.full((len(step.idx), len(FINAL_ACTIONS)), -np.inf)
        for j, a in enumerate(FINAL_ACTIONS):
            hit = np.where(self.classes == a)[0]
            p = proba[:, hit[0]] if len(hit) else np.zeros(len(step.idx))
            score[:, j] = np.where(step.feasible(int(a)), p, -np.inf)
        return FINAL_ACTIONS[np.argmax(score, axis=1)]


ROUTE_RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
ROUTE_CLAR = [0.0, 0.05, 0.15, 0.3]


class RouteLLMRule(Policy):
    name, label = "routellm", "RouteLLM-style (embedding + logistic)"

    def __init__(self, parent=None, alpha=np.inf, beta=np.inf):
        self.parent, self.alpha, self.beta = parent, alpha, beta

    def start(self, env: Env, episodes, B, fee) -> None:
        super().start(env, episodes, B, fee)
        self.s_win, self.s_amb = self.parent.scores_for(env)

    def decide(self, step: Step) -> np.ndarray:
        i = step.idx
        return threshold_action(self.s_win[i] > self.alpha, self.s_amb[i] > self.beta)


class RouteLLM(RouteLLMRule):
    def __init__(self):
        super().__init__(self)
        self._cache: dict[int, tuple] = {}

    def scores_for(self, env: Env):
        key = id(env)
        if key not in self._cache:
            if env.emb is None:
                raise RuntimeError("routellm needs question embeddings, run features with --embed")
            self._cache[key] = (self.win.predict_proba(env.emb)[:, 1], self.amb.predict_proba(env.emb)[:, 1])
        return self._cache[key]

    def fit(self, ctx: FitContext) -> None:
        from sklearn.linear_model import LogisticRegression

        env = ctx.calib
        self.win = LogisticRegression(C=1.0, max_iter=3000).fit(env.emb, env.correct[2, 0] & ~env.correct[0, 0])
        self.amb = LogisticRegression(C=1.0, max_iter=3000).fit(env.emb, env.is_amb)
        s_win, s_amb = self.scores_for(env)
        grid = [(a, b) for a in quantile_thresholds(s_win, ROUTE_RATES) for b in quantile_thresholds(s_amb, ROUTE_CLAR)]
        self.params = {}
        for B in ctx.budgets:
            self.params[B], _ = grid_search(ctx, lambda p: RouteLLMRule(self, *p), grid, B)

    def start(self, env, episodes, B, fee) -> None:
        super().start(env, episodes, B, fee)
        self.alpha, self.beta = nearest(self.params, B)


__all__ = ["LearnedRouter", "RouteLLM", "C1", "T1", "T2", "C0", "C2"]
