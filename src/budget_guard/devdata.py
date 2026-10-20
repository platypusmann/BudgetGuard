"""Calib-and-tune-only data for dry runs, kept apart from the final path.

`evaluate --eval-split tune` builds and uses this instead of data/graded and data/features:
  data/dev/graded/<tier>.parquet     grades for calib and tune questions only
  data/dev/features/features.parquet the feature table for calib and tune only
  data/dev/features/emb_<backend>*   question embeddings for calib and tune only

Nothing here opens test.jsonl, and every cache and prompts row from another split is skipped on
its raw prefix before any JSON is decoded (common.iter_jsonl_where). The only test-split access in
a dry run is split.verify hashing the file's bytes and the leak guard collecting test qids from
each line's prefix, both without decoding a single test row.
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import DATA, read_qids
from .grade import TIERS, grade_tier

DEV_SPLITS = ("calib", "tune")


def dev_root(data_dir: Path = DATA) -> Path:
    return Path(data_dir) / "dev"


def build(data_dir: Path = DATA, embed: str = "hash") -> Path:
    """Grade, build the feature table and embed, calib and tune only, into data_dir/dev."""
    from .features import build_embeddings, build_features

    data_dir = Path(data_dir)
    root = dev_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    print(f"dev data (calib and tune only) -> {root}")
    for tier in TIERS:
        grade_tier(tier, data_dir, "parquet", splits=DEV_SPLITS, out_root=root)
    build_features(data_dir, splits=DEV_SPLITS, out_root=root)
    if embed != "none":
        build_embeddings(data_dir, embed, splits=DEV_SPLITS, out_root=root)
    (root / "DEV").write_text(json.dumps({"splits": list(DEV_SPLITS), "embed": embed}), encoding="utf-8")
    return root


def assert_clean(data_dir: Path, test_qids: frozenset[str]) -> None:
    """Fail loudly if any test qid reached the dev table, grades or embedding index."""
    import pandas as pd

    from .grade import load_graded

    root = dev_root(data_dir)
    allowed = {q for s in DEV_SPLITS for q in read_qids(Path(data_dir) / "splits" / f"{s}.jsonl")}
    f = pd.read_parquet(root / "features" / "features.parquet", columns=["qid", "split"])
    assert set(f["split"]) <= set(DEV_SPLITS), f"dev table has splits {set(f['split'])}"
    assert not (set(f["qid"]) & test_qids) and set(f["qid"]) <= allowed, "a test qid reached the dev table"
    for tier in TIERS:
        g = load_graded(tier, root)
        if len(g):
            assert not ({p.split("::")[0] for p in g["pid"]} & test_qids), f"a test qid reached dev {tier} grades"
    for idx in (root / "features").glob("emb_*_qids.json"):
        raw = json.loads(idx.read_text())
        qids = raw["qids"] if isinstance(raw, dict) else raw
        assert not (set(qids) & test_qids), f"a test qid reached {idx.name}"
