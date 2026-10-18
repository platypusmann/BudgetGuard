"""Join the tier caches on pid into one compact feature table.

One row per pid in prompts.jsonl (raw and clarified prompts). Columns:
  ids          pid, qid, variant, split, pilot
  question     question (the raw text, for the free text ambiguity detector), is_ambiguous,
               n_interp, time_sensitive, q_chars, q_words
  per tier k   t{k}_present, t{k}_n_in, t{k}_n_out, t{k}_usd (as charged), t{k}_hit_max,
               t{k}_lp_mean, t{k}_lp_min, t{k}_lp_sum, t{k}_p_first, t{k}_ent,
               t{k}_em, t{k}_lenient, t{k}_lenient_plus, t{k}_f1
  tier 0 only  t0_sc_agree, t0_sc_distinct
  agreement    agree_01, agree_02, agree_12, normalized answers equal across tiers.
               Only observable after both tiers were called, so policies may only read
               agree_0k once tier k has been paid for.

Answers and samples are not kept, only what policies and grading need. Optional question
embeddings are cached to disk next to the table, one vector per qid, computed in batches.

  python -m budget_guard.features --data-dir data/synthetic --embed hash
  python -m budget_guard.features --embed minilm          # needs sentence-transformers and the model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import DATA, iter_jsonl
from .grade import TIERS, latest_good_rows, load_graded

LP_KEYS = {"lp_mean": "lp_mean", "lp_min": "lp_min", "lp_sum": "lp_sum", "p_first": "p_first",
           "ent_first_top": "ent"}
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def tier_columns(k: int) -> list[str]:
    base = ["present", "n_in", "n_out", "usd", "hit_max", *LP_KEYS.values(), "em", "lenient", "lenient_plus", "f1"]
    return [f"t{k}_{c}" for c in base]


def _cache_records(cache_file: Path, k: int) -> dict[str, dict]:
    out = {}
    for r in latest_good_rows(cache_file):
        rec = {f"t{k}_present": True, f"t{k}_n_in": r.get("n_in"), f"t{k}_n_out": r.get("n_out"),
               f"t{k}_usd": r.get("usd"), f"t{k}_hit_max": r.get("finish_reason") == "length"}
        for src, dst in LP_KEYS.items():
            rec[f"t{k}_{dst}"] = r.get(src)
        if k == 0:
            rec["t0_sc_agree"] = r.get("sc_agree")
            rec["t0_sc_distinct"] = r.get("sc_distinct")
        out[r["pid"]] = rec
    return out


def build_features(data_dir: Path = DATA):
    import pandas as pd

    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"
    meta = {}
    for name in ("calib", "tune", "test"):
        for r in iter_jsonl(splits_dir / f"{name}.jsonl"):
            meta[r["qid"]] = {"question": r["question"], "is_ambiguous": r["is_ambiguous"], "n_interp": r["n_interpretations"],
                              "time_sensitive": r["time_sensitive"], "q_chars": len(r["question"]),
                              "q_words": len(r["question"].split())}

    rows = {p["pid"]: {"pid": p["pid"], "qid": p["qid"], "variant": p["variant"], "split": p["split"],
                       "pilot": p["pilot"], **meta[p["qid"]]}
            for p in iter_jsonl(splits_dir / "prompts.jsonl")}

    answers = {}
    for k, tier in enumerate(TIERS):
        cache_file = data_dir / "cache" / f"{tier}.jsonl"
        if not cache_file.exists():
            print(f"{tier}: no cache, columns left empty")
            continue
        recs = _cache_records(cache_file, k)
        g = load_graded(tier, data_dir)
        if len(g):
            g = g.set_index("pid")
            # graded files from before lenient_plus existed simply leave that column empty
            lplus = g["lenient_plus"] if "lenient_plus" in g else [None] * len(g)
            for pid, em, len_, lp, f1 in zip(g.index, g["em"], g["lenient"], lplus, g["f1"]):
                if pid in recs:
                    recs[pid].update({f"t{k}_em": bool(em), f"t{k}_lenient": bool(len_), f"t{k}_f1": float(f1)})
                    if lp is not None:
                        recs[pid][f"t{k}_lenient_plus"] = bool(lp)
            answers[k] = dict(zip(g.index, g["answer_norm"]))
        unknown = 0
        for pid, rec in recs.items():
            if pid in rows:
                rows[pid].update(rec)
            else:
                unknown += 1
        if unknown:
            print(f"{tier}: {unknown} cached pids are not in prompts.jsonl, ignored")

    for a, b in ((0, 1), (0, 2), (1, 2)):
        if a in answers and b in answers:
            A, B = answers[a], answers[b]
            for pid, row in rows.items():
                if pid in A and pid in B:
                    row[f"agree_{a}{b}"] = bool(A[pid]) and A[pid] == B[pid]

    df = pd.DataFrame(list(rows.values()))
    for k in range(len(TIERS)):
        for c in tier_columns(k):
            if c not in df:
                df[c] = np.nan
        df[f"t{k}_present"] = df[f"t{k}_present"].fillna(False).astype(bool)
    for c in ("t0_sc_agree", "t0_sc_distinct", "agree_01", "agree_02", "agree_12"):
        if c not in df:
            df[c] = np.nan
    # compact dtypes, the table stays a few MB even on the real data
    for c in df.columns:
        if df[c].dtype == "float64" and not c.endswith("_usd"):
            df[c] = df[c].astype("float32")
    out = data_dir / "features" / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    present = {t: int(df[f"t{k}_present"].sum()) for k, t in enumerate(TIERS)}
    print(f"features: {len(df)} pids -> {out}  present per tier {present}")
    return df


def load_features(data_dir: Path = DATA):
    import pandas as pd

    return pd.read_parquet(Path(data_dir) / "features" / "features.parquet")


# ---------------------------------------------------------------- embeddings

ALL_SPLITS = ("calib", "tune", "test")


def _questions(data_dir: Path, splits: tuple[str, ...] = ALL_SPLITS) -> tuple[list[str], list[str]]:
    qids, texts = [], []
    for name in splits:
        for r in iter_jsonl(Path(data_dir) / "splits" / f"{name}.jsonl"):
            qids.append(r["qid"])
            texts.append(r["question"])
    return qids, texts


def _embed_hash(texts: list[str], dim: int = 512, batch: int = 1024) -> np.ndarray:
    """Deterministic word and bigram hashing. No fitting, no download, so no leakage and no network."""
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(n_features=dim, ngram_range=(1, 2), alternate_sign=True, norm="l2")
    return np.vstack([hv.transform(texts[i:i + batch]).toarray().astype("float32")
                      for i in range(0, len(texts), batch)])


def _embed_minilm(texts: list[str], batch: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MINILM, device="cpu")
    chunks = []
    for i in range(0, len(texts), 1024):  # bounded memory on 8 GB
        chunks.append(model.encode(texts[i:i + 1024], batch_size=batch, normalize_embeddings=True,
                                   show_progress_bar=False).astype("float32"))
    return np.vstack(chunks)


def _text_key(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _read_index(idx: Path) -> tuple[list[str], list[str | None]]:
    """qids and text fingerprints of the cached vectors. Old caches stored the qid list only."""
    raw = json.loads(idx.read_text())
    if isinstance(raw, list):
        return raw, [None] * len(raw)
    return raw["qids"], raw["text_sha1"]


def build_embeddings(data_dir: Path = DATA, backend: str = "hash", splits: tuple[str, ...] = ALL_SPLITS) -> Path:
    """Embed the questions of `splits` and add them to the on-disk cache.

    Incremental: only questions not already cached are embedded, so a calib and tune run now and
    a test run at final evaluation share one cache, and test is never embedded before it is needed.
    Each vector is keyed by qid and a fingerprint of the question text, so a question whose text
    changed (a rebuilt synthetic world reuses qids) is embedded again, never served stale.
    """
    data_dir = Path(data_dir)
    qids, texts = _questions(data_dir, splits)
    out = data_dir / "features" / f"emb_{backend}.npy"
    idx = data_dir / "features" / f"emb_{backend}_qids.json"
    have_q, have_k = _read_index(idx) if out.exists() and idx.exists() else ([], [])
    keys = [_text_key(t) for t in texts]
    have = {q: (k, i) for i, (q, k) in enumerate(zip(have_q, have_k))}
    todo = [(q, t, k) for q, t, k in zip(qids, texts, keys) if q not in have or have[q][0] != k]
    if not todo:
        print(f"embeddings: {out} already has all {len(qids)} questions of {', '.join(splits)}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    stale = {q for q, _, _ in todo if q in have}
    keep = [i for i, q in enumerate(have_q) if q not in stale and have_k[i] is not None]
    old = np.load(out)[keep] if keep else np.zeros((0, 0), "float32")
    new = _embed_minilm([t for _, t, _ in todo]) if backend == "minilm" else _embed_hash([t for _, t, _ in todo])
    emb = np.vstack([old, new]) if len(keep) else new
    q_all = [have_q[i] for i in keep] + [q for q, _, _ in todo]
    k_all = [have_k[i] for i in keep] + [k for _, _, k in todo]
    np.save(out, emb)
    idx.write_text(json.dumps({"qids": q_all, "text_sha1": k_all}))
    print(f"embeddings: embedded {len(todo)} {backend} questions ({len(stale)} with changed text), "
          f"{emb.shape[0]} cached -> {out}")
    return out


def load_embeddings(data_dir: Path = DATA, backend: str = "hash") -> dict[str, np.ndarray]:
    """qid to vector. Falls back to building the hash embedding if nothing is cached."""
    data_dir = Path(data_dir)
    out = data_dir / "features" / f"emb_{backend}.npy"
    if not out.exists():
        if backend != "hash":
            raise FileNotFoundError(f"{out} missing. Run python -m budget_guard.features --embed {backend}")
        build_embeddings(data_dir, "hash")
    qids, _ = _read_index(data_dir / "features" / f"emb_{backend}_qids.json")
    emb = np.load(out)
    return {q: emb[i] for i, q in enumerate(qids)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--embed", choices=["none", "hash", "minilm"], default="hash")
    ap.add_argument("--include-test", action="store_true",
                    help="required on real data: the table and embeddings cover the test split, "
                         "so this belongs to the final evaluation step only")
    args = ap.parse_args(argv)
    from .synthetic import is_synthetic

    if not is_synthetic(args.data_dir) and not args.include_test:
        raise SystemExit("features on real data reads the test split, which only the final evaluation may do. "
                         "Use python scripts/bg.py real, or pass --include-test at that step.")
    build_features(args.data_dir)
    if args.embed != "none":
        build_embeddings(args.data_dir, args.embed)


if __name__ == "__main__":
    main()
