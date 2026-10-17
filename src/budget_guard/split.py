"""Build frozen calib, tune, test and pilot splits from AmbigNQ (light config).

test   = official AmbigNQ validation split, 2,002 questions, multiple annotators. Never tuned on.
calib  = about 70 percent of official train, hash assigned. Fits p(correct | features, tier).
tune   = the other 30 percent of train. Thresholds, schedules and hyperparameters only.
pilot  = exactly PILOT_N questions drawn from calib only, so the pilot never touches tune or test.

Also writes prompts.jsonl, one row per thing to send to a model:
  pid = <qid>::orig   the question as the user typed it
  pid = <qid>::clar   the intended interpretation, used only after a clarify action

Run once:   python -m budget_guard.split
Refuses to overwrite existing splits unless --force, so the test set stays frozen.
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .common import RAW, SPLITS, JsonlAppender, file_sha256, iter_jsonl, stable_hash

HF_REPO = "sewon/ambig_qa"
HF_FILES = {
    "train": "light/train-00000-of-00001.parquet",
    "validation": "light/validation-00000-of-00001.parquet",
}
SPLIT_SALT = "split-v1"
INTENT_SALT = "intent-v1"
PILOT_SALT = "pilot-v1"
CALIB_PCT = 70
PILOT_N = 300

TIME_WORDS = re.compile(
    r"\b(last|latest|current|currently|now|recent|recently|newest|next|"
    r"so far|to date|this year|this season|still|anymore|as of)\b"
)


# ---------------------------------------------------------------- reading

def download_parquet(hf_split: str) -> Path:
    from huggingface_hub import hf_hub_download

    RAW.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(HF_REPO, HF_FILES[hf_split], repo_type="dataset", local_dir=RAW)
    )


def iter_parquet(path: Path, batch_size: int = 512) -> Iterator[dict]:
    """Stream rows in small record batches instead of loading the table."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["id", "question", "annotations"]):
        yield from batch.to_pylist()


def iter_ambignq(hf_split: str) -> Iterator[dict]:
    yield from iter_parquet(download_parquet(hf_split))


# ---------------------------------------------------------------- parsing

def _as_list(x) -> list:
    return [] if x is None else list(x)


def _pairs(obj) -> list[tuple[str, list[str]]]:
    """qaPairs arrives either as {question: [...], answer: [[...]]} or as a list of dicts."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(zip(_as_list(obj.get("question")), _as_list(obj.get("answer"))))
    return [(p.get("question"), _as_list(p.get("answer"))) for p in obj]


def _clean(answers) -> list[str]:
    seen, out = set(), []
    for a in _as_list(answers):
        a = (a or "").strip()
        if a and a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return out


def _annotations(ann) -> list[dict]:
    """Normalise to a list of {type, answer, qaPairs} dicts, one per annotator."""
    if isinstance(ann, dict):
        types = _as_list(ann.get("type"))
        answers = _as_list(ann.get("answer"))
        pairs = _as_list(ann.get("qaPairs"))
        return [
            {
                "type": t,
                "answer": answers[i] if i < len(answers) else [],
                "qaPairs": pairs[i] if i < len(pairs) else [],
            }
            for i, t in enumerate(types)
        ]
    return _as_list(ann)


def parse_row(row: dict) -> dict | None:
    """Turn one AmbigNQ row into our schema. Returns None when there is no usable gold."""
    qid = str(row["id"])
    question = (row.get("question") or "").strip()
    single_gold: list[str] = []
    interpretations: list[dict] = []

    for a in _annotations(row.get("annotations")):
        if a.get("type") == "singleAnswer":
            single_gold.extend(_clean(a.get("answer")))
        elif a.get("type") == "multipleQAs" and not interpretations:
            # first annotator who found multiple readings defines the interpretation set
            for q, ans in _pairs(a.get("qaPairs")):
                ans = _clean(ans)
                if q and ans:
                    interpretations.append({"question": q.strip(), "answers": ans})

    if len(interpretations) == 1:
        # an annotator flagged ambiguity but only one reading survived, treat as single answer
        single_gold.extend(interpretations[0]["answers"])
        interpretations = []

    is_ambiguous = len(interpretations) >= 2
    if is_ambiguous:
        k = stable_hash(qid, INTENT_SALT) % len(interpretations)
        intended = interpretations[k]
        gold = intended["answers"]
        intended_q = intended["question"]
    else:
        gold = _clean(single_gold)
        intended_q, k = None, None

    if not question or not gold:
        return None

    return {
        "qid": qid,
        "question": question,
        "is_ambiguous": is_ambiguous,
        "n_interpretations": len(interpretations),
        "intended_index": k,
        "intended_question": intended_q,
        "gold": gold,
        "interpretations": interpretations,
        "time_sensitive": bool(TIME_WORDS.search(question.lower())),
    }


# ---------------------------------------------------------------- building

def assign(qid: str, source: str) -> str:
    if source == "validation":
        return "test"
    return "calib" if stable_hash(qid, SPLIT_SALT) % 100 < CALIB_PCT else "tune"


def build(
    out_dir: Path = SPLITS,
    source: Callable[[str], Iterable[dict]] = iter_ambignq,
    force: bool = False,
    pilot_n: int = PILOT_N,
) -> dict:
    out_dir = Path(out_dir)
    manifest_path = out_dir / "MANIFEST.json"
    if manifest_path.exists() and not force:
        sys.exit(f"{manifest_path} exists. Splits are frozen. Pass --force only if you mean it.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("calib", "tune", "test", "pilot", "prompts"):
        (out_dir / f"{name}.jsonl").unlink(missing_ok=True)

    stats = Counter()
    seen: set[str] = set()
    pilot_heap: list[tuple[int, str]] = []  # bottom k by hash, streaming, memory is O(pilot_n)

    writers = {s: JsonlAppender(out_dir / f"{s}.jsonl", sync_every=500) for s in ("calib", "tune", "test")}
    try:
        for hf_split in ("train", "validation"):
            for raw in source(hf_split):
                stats[f"raw_{hf_split}"] += 1
                row = parse_row(raw)
                if row is None:
                    stats["skipped_no_gold"] += 1
                    continue
                if row["qid"] in seen:
                    stats["skipped_duplicate"] += 1
                    continue
                seen.add(row["qid"])
                split = assign(row["qid"], hf_split)
                row["split"] = split
                writers[split].write(row)
                stats[f"n_{split}"] += 1
                stats[f"amb_{split}"] += row["is_ambiguous"]
                if split == "calib":
                    item = (-stable_hash(row["qid"], PILOT_SALT), row["qid"])
                    if len(pilot_heap) < pilot_n:
                        heapq.heappush(pilot_heap, item)
                    elif item > pilot_heap[0]:
                        heapq.heapreplace(pilot_heap, item)
    finally:
        for w in writers.values():
            w.close()

    pilot_ids = {qid for _, qid in pilot_heap}
    with JsonlAppender(out_dir / "pilot.jsonl", sync_every=500) as w:
        for row in iter_jsonl(out_dir / "calib.jsonl"):
            if row["qid"] in pilot_ids:
                w.write(row)
                stats["n_pilot"] += 1
                stats["amb_pilot"] += row["is_ambiguous"]

    with JsonlAppender(out_dir / "prompts.jsonl", sync_every=1000) as w:
        for split in ("calib", "tune", "test"):
            for row in iter_jsonl(out_dir / f"{split}.jsonl"):
                base = {"qid": row["qid"], "split": split, "pilot": row["qid"] in pilot_ids}
                w.write({**base, "pid": f"{row['qid']}::orig", "variant": "orig", "text": row["question"]})
                stats["n_prompts"] += 1
                if row["is_ambiguous"]:
                    w.write({**base, "pid": f"{row['qid']}::clar", "variant": "clar",
                             "text": row["intended_question"]})
                    stats["n_prompts"] += 1

    leak = {r["qid"] for r in iter_jsonl(out_dir / "test.jsonl")} & (
        {r["qid"] for r in iter_jsonl(out_dir / "calib.jsonl")}
        | {r["qid"] for r in iter_jsonl(out_dir / "tune.jsonl")}
    )
    assert not leak, f"test overlaps calib or tune: {sorted(leak)[:5]}"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": f"{HF_REPO} (light)",
        "salts": {"split": SPLIT_SALT, "intent": INTENT_SALT, "pilot": PILOT_SALT},
        "calib_pct_of_train": CALIB_PCT,
        "stats": dict(sorted(stats.items())),
        "sha256": {p.name: file_sha256(p) for p in sorted(out_dir.glob("*.jsonl"))},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def verify(out_dir: Path = SPLITS) -> None:
    """Call at the top of every tuning and eval script. Fails loudly if a split file changed."""
    manifest = json.loads((Path(out_dir) / "MANIFEST.json").read_text())
    for name, digest in manifest["sha256"].items():
        if file_sha256(Path(out_dir) / name) != digest:
            raise RuntimeError(f"{name} changed since the splits were frozen")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pilot-n", type=int, default=PILOT_N)
    args = ap.parse_args()
    m = build(force=args.force, pilot_n=args.pilot_n)
    print(json.dumps(m["stats"], indent=2))


if __name__ == "__main__":
    main()
