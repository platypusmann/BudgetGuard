"""Shared helpers. Everything here streams from disk and never holds a whole file in RAM."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Iterable, Iterator

ROOT = Path(os.environ.get("BG_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
SPLITS = DATA / "splits"
CACHE = DATA / "cache"
CONFIGS = ROOT / "configs"


def stable_hash(text: str, salt: str) -> int:
    """Deterministic 64 bit hash, identical on every machine and Python version."""
    return int(hashlib.sha256(f"{salt}|{text}".encode("utf-8")).hexdigest()[:16], 16)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream rows. A crash mid write can leave one truncated line, which is skipped."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# Every cache row starts with its pid, every split and prompts row with its qid (and prompts rows
# with the split right after). These readers decide on that raw prefix, so a line that is filtered
# out is never JSON-decoded at all. That is how calib-and-tune-only code avoids reading test rows.
_LEAD_ID = re.compile(r'^\{"(?:pid|qid)": "([^"]+)"')
_LEAD_SPLIT = re.compile(r'^\{"qid": "[^"]+", "split": "([a-z]+)"')


def lead_qid(line: str) -> str | None:
    """qid of a cache, split or prompts line from its raw prefix, without decoding it."""
    m = _LEAD_ID.match(line)
    return m.group(1).split("::")[0] if m else None


def iter_jsonl_where(path: Path, keep: Callable[[str], bool]) -> Iterator[dict]:
    """Like iter_jsonl, but only lines whose raw text passes keep(line) are decoded."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not keep(line):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def only_qids(qids: Iterable[str]) -> Callable[[str], bool]:
    qids = set(qids)
    return lambda line: lead_qid(line) in qids


def only_splits(splits: Iterable[str]) -> Callable[[str], bool]:
    """For prompts.jsonl lines, which carry the split right after the qid."""
    splits = set(splits)

    def keep(line: str) -> bool:
        m = _LEAD_SPLIT.match(line)
        return bool(m) and m.group(1) in splits
    return keep


def read_qids(path: Path) -> list[str]:
    """qids of a split file from each line's raw prefix. No row is decoded."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            q = lead_qid(line.strip())
            if q is not None:
                out.append(q)
    return out


class JsonlAppender:
    """Append only writer. Flushes and fsyncs every sync_every rows so a kill loses almost nothing."""

    def __init__(self, path: Path, sync_every: int = 1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sync_every = max(1, sync_every)
        self._pending = 0
        needs_newline = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) != b"\n"
        # newline pinned so file bytes, and so MANIFEST hashes, match on Windows, Linux and Kaggle
        self._f = self.path.open("a", encoding="utf-8", newline="\n")
        if needs_newline:
            # previous run died mid line, close that broken line so the next row stays valid
            self._f.write("\n")

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._pending += 1
        if self._pending >= self.sync_every:
            self.sync()

    def sync(self) -> None:
        self._f.flush()
        os.fsync(self._f.fileno())
        self._pending = 0

    def close(self) -> None:
        self.sync()
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def done_ids(path: Path, key: str = "pid", ok_only: bool = True) -> set[str]:
    """Ids already in a cache file. A set of ~20k short strings is a few MB."""
    out: set[str] = set()
    for row in iter_jsonl(path):
        if ok_only and row.get("error"):
            continue
        if key in row:
            out.add(row[key])
    return out


def load_yaml(path: Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
