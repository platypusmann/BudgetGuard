"""Join the tier caches on pid into one compact feature table.

One row per pid in prompts.jsonl (raw and clarified prompts). Columns:
  ids          pid, qid, variant, split, pilot
  question     is_ambiguous, n_interp, time_sensitive, q_chars, q_words
  per tier k   t{k}_present, t{k}_n_in, t{k}_n_out, t{k}_usd (as charged), t{k}_hit_max,
               t{k}_lp_mean, t{k}_lp_min, t{k}_lp_sum, t{k}_p_first, t{k}_ent,
               t{k}_em, t{k}_lenient, t{k}_f1
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
    base = ["present", "n_in", "n_out", "usd", "hit_max", *LP_KEYS.values(), "em", "lenient", "f1"]
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
            meta[r["qid"]] = {"is_ambiguous": r["is_ambiguous"], "n_interp": r["n_interpretations"],
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
            for pid, em, len_, f1, ans in zip(g.index, g["em"], g["lenient"], g["f1"], g["answer_norm"]):
                if pid in recs:
                    recs[pid].update({f"t{k}_em": bool(em), f"t{k}_lenient": bool(len_), f"t{k}_f1": float(f1)})
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

def _questions(data_dir: Path) -> tuple[list[str], list[str]]:
    qids, texts = [], []
    for name in ("calib", "tune", "test"):
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


def build_embeddings(data_dir: Path = DATA, backend: str = "hash") -> Path:
    data_dir = Path(data_dir)
    qids, texts = _questions(data_dir)
    out = data_dir / "features" / f"emb_{backend}.npy"
    idx = data_dir / "features" / f"emb_{backend}_qids.json"
    if out.exists() and idx.exists() and json.loads(idx.read_text()) == qids:
        print(f"embeddings: {out} already up to date")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    emb = _embed_minilm(texts) if backend == "minilm" else _embed_hash(texts)
    np.save(out, emb)
    idx.write_text(json.dumps(qids))
    print(f"embeddings: {emb.shape} {backend} -> {out}")
    return out


def load_embeddings(data_dir: Path = DATA, backend: str = "hash") -> dict[str, np.ndarray]:
    """qid to vector. Falls back to building the hash embedding if nothing is cached."""
    data_dir = Path(data_dir)
    out = data_dir / "features" / f"emb_{backend}.npy"
    if not out.exists():
        if backend != "hash":
            raise FileNotFoundError(f"{out} missing. Run python -m budget_guard.features --embed {backend}")
        build_embeddings(data_dir, "hash")
    qids = json.loads((data_dir / "features" / f"emb_{backend}_qids.json").read_text())
    emb = np.load(out)
    return {q: emb[i] for i, q in enumerate(qids)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--embed", choices=["none", "hash", "minilm"], default="hash")
    args = ap.parse_args(argv)
    build_features(args.data_dir)
    if args.embed != "none":
        build_embeddings(args.data_dir, args.embed)


if __name__ == "__main__":
    main()
