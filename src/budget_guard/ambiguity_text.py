"""Free ambiguity detector from the question text alone. No model call, so no cost and no probe.

Why it exists: the pilot showed Tier 0 self consistency does not predict ambiguity on real data
(sc_agree AUROC 0.52), so the router had almost no way to know when to clarify before paying.
The question text is free and AmbigNQ labels every question, so a detector can be trained on all
of calib, not just the 300 pilot rows.

Three versions, all logistic regression with the regularization picked by 5-fold CV on calib:
  hand      question length, first wh-word, "how many", time_sensitive, capitalized tokens,
            year, ordinal, digits, quotes, commas
  minilm    all-MiniLM-L6-v2 sentence embedding (or whatever embedding the caller passes)
  combined  both

The CLI trains on calib and evaluates on tune. It never reads test questions.

  python scripts/bg.py ambiguity-text
"""
from __future__ import annotations

import argparse
import re
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .common import DATA, ROOT, iter_jsonl
from .leakage import assert_no_test, verify_and_test_qids
from .split import TIME_WORDS

WH = ["who", "what", "when", "where", "which", "how", "why"]
HAND_NAMES = (["n_chars", "n_words"] + [f"wh_{w}" for w in WH] + ["wh_other", "how_many", "time_sensitive",
              "n_caps", "has_year", "has_ordinal", "has_digit", "has_quote", "n_commas"])
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_ORD = re.compile(r"\b(\d+(st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", re.I)


CS = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.1, 1.0, 10.0]


@contextmanager
def _quiet():
    """sklearn 1.8 FutureWarnings about LogisticRegressionCV attributes, not about our use of it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        yield


def hand_features(questions: list[str]) -> np.ndarray:
    rows = []
    for q in questions:
        toks = q.split()
        low = q.lower()
        first = re.sub(r"[^a-z]", "", toks[0].lower()) if toks else ""
        wh = [float(first == w) for w in WH]
        rows.append([len(q), len(toks), *wh, float(first not in WH), float(low.startswith("how many")),
                     float(bool(TIME_WORDS.search(low))),
                     sum(t[:1].isupper() for t in toks[1:]),
                     float(bool(_YEAR.search(q))), float(bool(_ORD.search(q))), float(any(c.isdigit() for c in q)),
                     float('"' in q or "'" in q), q.count(",")])
    return np.asarray(rows, dtype=float)


class TextAmbiguityDetector:
    """P(ambiguous | question text). version is hand, minilm or combined."""

    def __init__(self, version: str = "combined"):
        assert version in ("hand", "minilm", "combined")
        self.version = version

    def _X(self, questions: list[str], emb: np.ndarray | None) -> np.ndarray:
        parts = []
        if self.version in ("hand", "combined"):
            parts.append(hand_features(questions))
        if self.version in ("minilm", "combined"):
            if emb is None:
                raise ValueError(f"the {self.version} detector needs an embedding")
            parts.append(np.asarray(emb, dtype=float))
        return np.hstack(parts)

    def _model(self):
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # wide grid: on 384 embedding dims the best C sits low, and the edge of the grid must not bind
        return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=5, scoring="roc_auc", max_iter=3000))

    def fit(self, questions: list[str], emb: np.ndarray | None, y: np.ndarray) -> "TextAmbiguityDetector":
        with _quiet():
            self.model = self._model().fit(self._X(questions, emb), np.asarray(y, bool))
        return self

    def predict(self, questions: list[str], emb: np.ndarray | None = None) -> np.ndarray:
        return self.model.predict_proba(self._X(questions, emb))[:, 1]

    def out_of_fold(self, questions: list[str], emb: np.ndarray | None, y: np.ndarray, folds: int = 5) -> np.ndarray:
        """Calib scores from models that never saw the row, so tables built on calib stay honest."""
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        cv = StratifiedKFold(folds, shuffle=True, random_state=0)
        with _quiet():
            return cross_val_predict(self._model(), self._X(questions, emb), np.asarray(y, bool), cv=cv,
                                     method="predict_proba")[:, 1]


def _rows(data_dir: Path, split: str) -> list[dict]:
    return list(iter_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl"))


def run(data_dir: Path = DATA, out: Path = ROOT / "results", backend: str = "minilm", n_boot: int = 2000):
    import pandas as pd

    from .features import build_embeddings, load_embeddings
    from .pilot_check import auroc, boot_auroc
    from .synthetic import is_synthetic

    data_dir, out = Path(data_dir), Path(out)
    test_qids = verify_and_test_qids(data_dir / "splits")           # rule 3, first thing
    calib, tune = _rows(data_dir, "calib"), _rows(data_dir, "tune")
    assert_no_test([r["qid"] for r in calib], test_qids, "detector training data")
    assert_no_test([r["qid"] for r in tune], test_qids, "detector evaluation data")
    build_embeddings(data_dir, backend, splits=("calib", "tune"))   # test is not embedded here
    emb = load_embeddings(data_dir, backend)

    def parts(rows):
        return ([r["question"] for r in rows], np.vstack([emb[r["qid"]] for r in rows]),
                np.array([r["is_ambiguous"] for r in rows], bool))

    qc, ec, yc = parts(calib)
    qt, et, yt = parts(tune)
    results = []
    for version in ("hand", "minilm", "combined"):
        det = TextAmbiguityDetector(version).fit(qc, ec, yc)
        p = det.predict(qt, et)
        lo, hi = boot_auroc(p, yt, n_boot, seed={"hand": 1, "minilm": 2, "combined": 3}[version])
        c = det.model[-1].C_[0]
        results.append({"version": version, "embedding": backend if version != "hand" else "none",
                        "train": "calib", "eval": "tune", "n_train": len(yc), "n_eval": len(yt),
                        "eval_ambiguous_rate": float(yt.mean()), "auroc": auroc(p, yt), "ci_lo": lo, "ci_hi": hi,
                        "chosen_C": float(c)})
    df = pd.DataFrame(results)
    df.insert(0, "data", "SYNTHETIC" if is_synthetic(data_dir) else "REAL")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ambiguity_text_detector.csv"
    df.to_csv(path, index=False)
    print(f"\nText ambiguity detector, {df['data'].iloc[0]} data, trained on calib ({len(yc)}), "
          f"evaluated on tune ({len(yt)}, {yt.mean():.1%} ambiguous). AUROC [bootstrap 95 percent CI]")
    for r in df.itertuples():
        print(f"  {r.version:9s} {r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]   C = {r.chosen_C:g}")
    print(f"wrote {path}")
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--embed", choices=["minilm", "hash"], default="minilm")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    run(args.data_dir, args.out, args.embed, args.n_boot)


if __name__ == "__main__":
    main()
