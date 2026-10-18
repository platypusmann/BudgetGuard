"""Calibrated signals every policy shares, fitted on calib only.

  p0     P(Tier 0 raw answer is correct | Tier 0 features), needs Tier 0 paid for
  p_amb  P(question is ambiguous | Tier 0 features, question features), needs Tier 0
  p1     P(Tier 1 raw answer is correct | Tier 1 features, agreement with Tier 0), needs Tiers 0 and 1
  x0     standardized Tier 0 feature matrix, for the learned router, needs Tier 0
  p_text P(question is ambiguous | question text), the free detector from ambiguity_text.py,
         needs nothing, so a policy can use it before paying for any call. Hand features plus the
         env's question embedding when there is one, hand features alone otherwise

On calib itself the values are out of fold (5 folds), so tables estimated on calib, like the DP's
success rates per bin, are not optimistic. Other splits get the model fitted on all of calib.
Policies get the same information, so comparisons are between decision rules, as in the toy.
"""
from __future__ import annotations

import numpy as np

from .episodes import Env
from .leakage import assert_no_test

T0_FEATS = ["t0_lp_mean", "t0_lp_min", "t0_p_first", "t0_ent", "t0_sc_agree", "t0_sc_distinct",
            "t0_n_out", "t0_hit_max"]
Q_FEATS = ["time_sensitive", "q_words"]
T1_FEATS = ["t1_lp_mean", "t1_lp_min", "t1_p_first", "t1_ent", "t1_n_out", "agree_01"]


def _matrix(env: Env, cols: list[str], fill: dict) -> np.ndarray:
    df = env.orig[cols].astype(float)
    return df.fillna(fill).to_numpy()


class SignalModel:
    def __init__(self, folds: int = 5, C: float = 1.0):
        self.folds, self.C = folds, C

    def _clf(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))

    def fit(self, calib: Env, test_qids: frozenset[str]) -> "SignalModel":
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.preprocessing import StandardScaler

        assert calib.split == "calib", "signals are fitted on calib only"
        assert_no_test(calib.qids, test_qids, "signal fitting data")
        c0, c1 = T0_FEATS + Q_FEATS, T1_FEATS
        self.fill0 = calib.orig[c0].astype(float).median().to_dict()
        self.fill1 = calib.orig[c1].astype(float).median().to_dict()
        X0, X1 = _matrix(calib, c0, self.fill0), _matrix(calib, c1, self.fill1)
        targets = {"p0": (X0, calib.correct[0, 0]), "p_amb": (X0, calib.is_amb),
                   "p1": (X1, calib.correct[1, 0])}
        cv = StratifiedKFold(self.folds, shuffle=True, random_state=0)
        self.models, self.oof = {}, {}
        for name, (X, y) in targets.items():
            self.oof[name] = cross_val_predict(self._clf(), X, y, cv=cv, method="predict_proba")[:, 1]
            self.models[name] = self._clf().fit(X, y)
        self.scaler0 = StandardScaler().fit(X0)
        self.calib_qids = calib.qids
        self.text = None
        if calib.questions is not None:
            from .ambiguity_text import TextAmbiguityDetector

            self.text = TextAmbiguityDetector("combined" if calib.emb is not None else "hand")
            self.oof["p_text"] = self.text.out_of_fold(calib.questions, calib.emb, calib.is_amb)
            self.text.fit(calib.questions, calib.emb, calib.is_amb)
        return self

    def apply(self, env: Env) -> Env:
        X0 = _matrix(env, T0_FEATS + Q_FEATS, self.fill0)
        X1 = _matrix(env, T1_FEATS, self.fill1)
        in_sample = env.split == "calib" and len(env.qids) == len(self.calib_qids) and (env.qids == self.calib_qids).all()
        for name, X, needs in (("p0", X0, (0,)), ("p_amb", X0, (0,)), ("p1", X1, (0, 1))):
            vals = self.oof[name] if in_sample else self.models[name].predict_proba(X)[:, 1]
            env.add_signal(name, vals, needs)
        env.add_signal("x0", self.scaler0.transform(X0).astype("float32"), (0,))
        if self.text is not None and env.questions is not None:
            vals = self.oof["p_text"] if in_sample else self.text.predict(env.questions, env.emb)
            env.add_signal("p_text", vals, ())
        return env

    def auc_report(self, env: Env) -> dict:
        from sklearn.metrics import roc_auc_score

        out = {}
        pairs = [("p0", env.correct[0, 0]), ("p_amb", env.is_amb), ("p1", env.correct[1, 0])]
        if "p_text" in env.signals:
            pairs.append(("p_text", env.is_amb))
        for name, y in pairs:
            if len(set(y)) == 2:
                out[name] = round(float(roc_auc_score(y, env.signals[name][0])), 3)
        return out
