"""Episodes: a stream of K questions under one hard dollar cap B.

Per question the runner works in stages. At each stage the policy returns one action code:
  P0 P1 P2   probe: call tier k on the raw prompt, pay for it, reveal its signals, then decide again
  T0 T1 T2   answer with tier k on the raw prompt (free if tier k was already probed)
  C0 C1 C2   clarify, pay the clarify fee, then tier k answers the clarified prompt
  SKIP       give no answer, costs nothing, scores 0

Hard cap. Before any action is taken the runner computes its worst-case extra cost
(input tokens times input price plus max output tokens times output price, plus the fee for a
clarify) and masks it if that exceeds the remaining budget. A masked action is downgraded along a
fixed ladder (Ck to Tk, Tk to Tk-1, ..., to SKIP, or to the best tier already paid for). The
runner then charges the actual cost recomputed from the cached token counts. Actual never exceeds
worst case, so a masked policy can never break the cap. Pass masked=False to see what a
budget-blind policy would spend without the guard.

Information rules. A policy only sees a tier's signals after paying for that tier, enforced by
Step.sig, which blanks unrevealed rows. Worst-case costs shown to policies for clarify actions use
the raw prompt's length, because the clarified prompt's length would leak whether the question is
ambiguous. The runner masks with the true clarified length.

Everything is vectorized across episodes: the loop is over the K positions, never over episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .common import CONFIGS, load_yaml, stable_hash

T0, T1, T2, C0, C1, C2, SKIP, P0, P1, P2 = range(10)
FINALS = (T0, T1, T2, C0, C1, C2, SKIP)
NAMES = ["T0", "T1", "T2", "C0", "C1", "C2", "SKIP", "P0", "P1", "P2"]
N_TIERS = 3
MAX_STAGES = 4
METRICS = ("em", "lenient", "lenient_plus")
# where a masked action falls to. P_k becomes T_k, which then keeps falling if needed
LADDER = np.array([SKIP, T0, T1, T0, T1, T2, SKIP, T0, T1, T2])


# ---------------------------------------------------------------- prices

@dataclass(frozen=True)
class Prices:
    pin: tuple[float, float, float]     # USD per input token, per tier
    pout: tuple[float, float, float]    # USD per output token, per tier
    max_out: tuple[int, int, int]       # max completion tokens, per tier
    label: str = ""


def load_prices(tier0: tuple[float, float] | None = None, batch: bool = False) -> Prices:
    """Price sheet for evaluation. Standard (non batch) API prices by default, because a live
    router cannot wait a day for a batch. Tier 0 uses its nominal price unless overridden."""
    cfg = load_yaml(CONFIGS / "tiers.yaml")
    t0 = cfg["tier0"]
    p0 = tier0 if tier0 is not None else (t0["nominal_price_in"], t0["nominal_price_out"])
    pin, pout, mx = [p0[0] / 1e6], [p0[1] / 1e6], [t0["max_new_tokens"]]
    for t in ("tier1", "tier2"):
        d = cfg[t].get("batch_discount", 1.0) if batch else 1.0
        pin.append(cfg[t]["price_in"] * d / 1e6)
        pout.append(cfg[t]["price_out"] * d / 1e6)
        mx.append(cfg[t]["max_completion_tokens"])
    label = f"t0 {p0[0]:g}/{p0[1]:g}" + (" batch" if batch else "")
    return Prices(tuple(pin), tuple(pout), tuple(mx), label)


# ---------------------------------------------------------------- environment

class Env:
    """One split at one price sheet, as flat arrays indexed by pool position.

    correct, cost and wc have shape (3 tiers, 2 variants, n). Variant 0 is the raw prompt and
    variant 1 the resolved prompt, which is the clarified prompt for ambiguous questions and the
    raw prompt again otherwise.
    """

    def __init__(self, feats, split: str, prices: Prices, metric: str = "lenient", emb: dict | None = None,
                 verbose: bool = True):
        self.split, self.prices, self.metric = split, prices, metric
        f = feats[feats["split"] == split]
        orig = f[f["variant"] == "orig"].set_index("qid")
        clar = f[f["variant"] == "clar"].set_index("qid")
        ok = orig[[f"t{k}_present" for k in range(N_TIERS)]].all(axis=1)
        amb = orig["is_ambiguous"].astype(bool)
        clar_ok = clar[[f"t{k}_present" for k in range(N_TIERS)]].all(axis=1).reindex(orig.index, fill_value=False)
        graded = orig[[f"t{k}_{metric}" for k in range(N_TIERS)]].notna().all(axis=1)
        clar_graded = clar[[f"t{k}_{metric}" for k in range(N_TIERS)]].notna().all(axis=1).reindex(
            orig.index, fill_value=False)
        keep = ok & graded & (~amb | (clar_ok & clar_graded))
        self.n_dropped = int((~keep).sum())
        if verbose and self.n_dropped:
            print(f"env {split}: dropped {self.n_dropped} questions with missing cache rows or grades")
        orig = orig[keep]
        self.qids = orig.index.to_numpy()
        self.n = len(self.qids)
        self.is_amb = orig["is_ambiguous"].to_numpy(bool)
        # raw question text, for the free text ambiguity detector. Older feature tables lack it
        self.questions = orig["question"].tolist() if "question" in orig else None
        self.time_sensitive = orig["time_sensitive"].to_numpy(bool)
        res = orig.copy()
        clar = clar.reindex(self.qids)
        for c in [c for c in orig.columns if c[:1] == "t" and c[1:2].isdigit()]:
            res.loc[self.is_amb, c] = clar.loc[self.is_amb, c].to_numpy()
        self.orig, self.res = orig.reset_index(), res.reset_index()

        # correctness under every metric that is fully graded for this pool. self.correct is the
        # primary one, the one policies are fitted and tuned on. The others are only for rescoring
        self.correct_by: dict[str, np.ndarray] = {}
        for m in METRICS:
            cols = [f"t{k}_{m}" for k in range(N_TIERS)]
            if all(c in self.orig for c in cols) and all(self.orig[c].notna().all() and self.res[c].notna().all()
                                                         for c in cols):
                arr = np.zeros((N_TIERS, 2, self.n), bool)
                for k in range(N_TIERS):
                    for v, df in enumerate((self.orig, self.res)):
                        arr[k, v] = df[f"t{k}_{m}"].astype(bool).to_numpy()
                self.correct_by[m] = arr
        self.correct = np.zeros((N_TIERS, 2, self.n), bool)
        self.cost = np.zeros((N_TIERS, 2, self.n))
        self.wc = np.zeros((N_TIERS, 2, self.n))
        for k in range(N_TIERS):
            for v, df in enumerate((self.orig, self.res)):
                n_in = df[f"t{k}_n_in"].to_numpy(float)
                n_out = df[f"t{k}_n_out"].to_numpy(float)
                self.correct[k, v] = df[f"t{k}_{metric}"].fillna(False).to_numpy(bool)
                self.cost[k, v] = n_in * prices.pin[k] + n_out * prices.pout[k]
                self.wc[k, v] = n_in * prices.pin[k] + prices.max_out[k] * prices.pout[k]
        assert (self.cost <= self.wc + 1e-15).all(), "an actual cost exceeds its worst case"
        self.emb = None
        if emb is not None:
            self.emb = np.vstack([emb[q] for q in self.qids]).astype("float32")
        # name -> (array of shape (n,) or (n, d), tiers that must be revealed to read it)
        self.signals: dict[str, tuple[np.ndarray, tuple[int, ...]]] = {}

    def add_signal(self, name: str, values: np.ndarray, needs: tuple[int, ...]) -> None:
        assert len(values) == self.n
        self.signals[name] = (np.asarray(values), tuple(needs))

    def with_prices(self, prices: Prices) -> "Env":
        """Same questions and signals, costs recomputed for another price sheet."""
        new = object.__new__(Env)
        new.__dict__ = dict(self.__dict__)
        new.prices = prices
        new.cost = np.zeros_like(self.cost)
        new.wc = np.zeros_like(self.wc)
        for k in range(N_TIERS):
            for v, df in enumerate((self.orig, self.res)):
                n_in = df[f"t{k}_n_in"].to_numpy(float)
                n_out = df[f"t{k}_n_out"].to_numpy(float)
                new.cost[k, v] = n_in * prices.pin[k] + n_out * prices.pout[k]
                new.wc[k, v] = n_in * prices.pin[k] + prices.max_out[k] * prices.pout[k]
        return new

    def unit_cost(self) -> float:
        """Mean Tier 2 raw-prompt cost. Budgets and clarify fees are quoted in this unit."""
        return float(self.cost[2, 0].mean())


# ---------------------------------------------------------------- sampler

def sample_episodes(n_pool: int, K: int, n_episodes: int, seed: int, tag: str = "") -> np.ndarray:
    """Seeded streams of K distinct pool positions. Same seed and tag, same streams, on any machine."""
    if K > n_pool:
        raise ValueError(f"K={K} is larger than the pool of {n_pool} questions")
    rng = np.random.default_rng([seed, stable_hash(tag, "episodes") % (2 ** 32)])
    return np.stack([rng.permutation(n_pool)[:K] for _ in range(n_episodes)]).astype(np.int32)


# ---------------------------------------------------------------- runner

@dataclass
class Step:
    """What a policy may look at when it decides. Arrays are over episodes."""
    env: Env
    t: int
    K: int
    B: float
    fee: float
    idx: np.ndarray
    left: np.ndarray
    revealed: np.ndarray
    stage: int
    active: np.ndarray

    @property
    def n_left(self) -> int:
        """Questions left including this one."""
        return self.K - self.t

    def sig(self, name: str) -> np.ndarray:
        values, needs = self.env.signals[name]
        out = values[self.idx].astype(float, copy=True)
        hidden = ~self.revealed[:, list(needs)].all(axis=1) if needs else np.zeros(len(self.idx), bool)
        out[hidden] = np.nan
        return out

    def emb(self) -> np.ndarray:
        return self.env.emb[self.idx]

    def wc(self, action: int) -> np.ndarray:
        """Worst-case extra cost as the policy sees it. Clarify uses the raw prompt length."""
        e = self.env
        if action == SKIP:
            return np.zeros(len(self.idx))
        if action in (P0, P1, P2):
            k = action - P0
            return np.where(self.revealed[:, k], np.inf, e.wc[k, 0, self.idx])
        if action in (T0, T1, T2):
            k = action - T0
            return np.where(self.revealed[:, k], 0.0, e.wc[k, 0, self.idx])
        k = action - C0
        return self.fee + e.wc[k, 0, self.idx]

    def feasible(self, action: int) -> np.ndarray:
        return self.wc(action) <= self.left + 1e-15


@dataclass
class RunResult:
    policy: str
    B: float
    fee: float
    masked: bool
    episodes: np.ndarray
    correct: np.ndarray        # (E, K) bool
    spend: np.ndarray          # (E, K) charged for each question
    final: np.ndarray          # (E, K) final action code
    probes: np.ndarray         # (E, K) bitmask of tiers probed
    downgraded: np.ndarray     # (E, K) a requested action was masked
    budget_before: np.ndarray  # (E, K)
    extra: dict = field(default_factory=dict)

    @property
    def total_spend(self) -> np.ndarray:
        return self.spend.sum(axis=1)

    @property
    def violation(self) -> np.ndarray:
        return self.total_spend > self.B * (1 + 1e-9) + 1e-15

    @property
    def clarified(self) -> np.ndarray:
        return (self.final >= C0) & (self.final <= C2)

    def log_frame(self, env: Env):
        """Every decision as one row. Big, so evaluate only writes it for chosen runs."""
        import pandas as pd

        E, K = self.correct.shape
        return pd.DataFrame({
            "policy": self.policy, "B": self.B, "fee": self.fee, "masked": self.masked,
            "episode": np.repeat(np.arange(E), K), "t": np.tile(np.arange(K), E),
            "qid": env.qids[self.episodes].ravel(), "is_ambiguous": env.is_amb[self.episodes].ravel(),
            "budget_before": self.budget_before.ravel(), "probes": self.probes.ravel(),
            "final": np.array(NAMES)[self.final.ravel()], "downgraded": self.downgraded.ravel(),
            "cost": self.spend.ravel(), "correct": self.correct.ravel(),
        })


def rescore(r: RunResult, env: Env, metric: str) -> np.ndarray:
    """Correctness of the same final decisions under another metric. Nothing is rerun or refitted."""
    a = r.final.astype(int)
    clar = (a >= C0) & (a <= C2)
    tier = np.clip(np.where(clar, a - C0, a), 0, N_TIERS - 1)
    return env.correct_by[metric][tier, clar.astype(int), r.episodes] & (a != SKIP)


def true_wc(env: Env, action: np.ndarray, idx: np.ndarray, revealed: np.ndarray, fee: float) -> np.ndarray:
    """Worst-case extra cost the runner masks with, per episode, for an array of actions."""
    out = np.zeros(len(idx))
    amb = env.is_amb[idx]
    for a in range(10):
        m = action == a
        if not m.any() or a == SKIP:
            continue
        if a >= P0:
            k = a - P0
            out[m] = np.where(revealed[m, k], np.inf, env.wc[k, 0, idx[m]])
        elif a <= T2:
            k = a
            out[m] = np.where(revealed[m, k], 0.0, env.wc[k, 0, idx[m]])
        else:
            k = a - C0
            reuse = ~amb[m] & revealed[m, k]
            out[m] = fee + np.where(reuse, 0.0, env.wc[k, 1, idx[m]])
    return out


def actual_cost(env: Env, action: np.ndarray, idx: np.ndarray, revealed: np.ndarray, fee: float) -> np.ndarray:
    out = np.zeros(len(idx))
    amb = env.is_amb[idx]
    for a in range(10):
        m = action == a
        if not m.any() or a == SKIP:
            continue
        if a >= P0:
            out[m] = env.cost[a - P0, 0, idx[m]]
        elif a <= T2:
            out[m] = np.where(revealed[m, a], 0.0, env.cost[a, 0, idx[m]])
        else:
            k = a - C0
            reuse = ~amb[m] & revealed[m, k]
            out[m] = fee + np.where(reuse, 0.0, env.cost[k, 1, idx[m]])
    return out


def _downgrade(env, action, idx, revealed, left, fee):
    """Replace masked actions along the ladder until every action is affordable."""
    down = np.zeros(len(idx), bool)
    for _ in range(12):
        bad = true_wc(env, action, idx, revealed, fee) > left + 1e-15
        if not bad.any():
            break
        down |= bad
        action = np.where(bad, LADDER[action], action)
    # falling all the way to SKIP is silly if a tier was already paid for
    best = np.where(revealed[:, 2], T2, np.where(revealed[:, 1], T1, np.where(revealed[:, 0], T0, SKIP)))
    return np.where((action == SKIP) & down, best, action), down


def run(env: Env, policy, episodes: np.ndarray, B: float, fee: float, masked: bool = True,
        on_step: Callable | None = None) -> RunResult:
    E, K = episodes.shape
    left = np.full(E, float(B))
    out = RunResult(policy.name, float(B), float(fee), masked, episodes,
                    correct=np.zeros((E, K), bool), spend=np.zeros((E, K)), final=np.full((E, K), SKIP, np.int8),
                    probes=np.zeros((E, K), np.int8), downgraded=np.zeros((E, K), bool),
                    budget_before=np.zeros((E, K)))
    policy.start(env, episodes, B, fee)
    rows = np.arange(E)
    for t in range(K):
        idx = episodes[:, t]
        out.budget_before[:, t] = left
        revealed = np.zeros((E, N_TIERS), bool)
        active = np.ones(E, bool)
        spent_q = np.zeros(E)
        for stage in range(MAX_STAGES):
            step = Step(env, t, K, B, fee, idx, left.copy(), revealed.copy(), stage, active.copy())
            act = np.asarray(policy.decide(step), dtype=np.int64)
            if on_step is not None:
                on_step(step, act)
            if stage == MAX_STAGES - 1:
                act = np.where(act >= P0, LADDER[act], act)  # no more probing, must answer
            # probing a tier already paid for means answer with it
            reprobe = (act >= P0) & revealed[rows, np.clip(act - P0, 0, 2)]
            act = np.where(reprobe, act - P0, act)
            if masked:
                act, down = _downgrade(env, act, idx, revealed, left, fee)
                out.downgraded[:, t] |= down & active
            act = np.where(active, act, SKIP)
            cost = np.where(active, actual_cost(env, act, idx, revealed, fee), 0.0)
            left -= cost
            spent_q += cost
            is_probe = active & (act >= P0)
            if is_probe.any():
                k = act[is_probe] - P0
                revealed[rows[is_probe], k] = True
                out.probes[rows[is_probe], t] |= (1 << k).astype(np.int8)
            fin = active & (act < P0)
            if fin.any():
                a = act[fin]
                variant = (a >= C0) & (a <= C2)
                tier = np.where(variant, a - C0, a)
                ok = env.correct[np.minimum(tier, 2), variant.astype(int), idx[fin]] & (a != SKIP)
                out.correct[fin, t] = ok
                out.final[fin, t] = a
                active &= ~fin
            if not active.any():
                break
        out.spend[:, t] = spent_q
    return out


# ---------------------------------------------------------------- config

def load_experiment(path=None) -> dict:
    return load_yaml(path or CONFIGS / "experiment.yaml")
