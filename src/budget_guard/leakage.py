"""Guards for rule 3: fit on calib, tune on tune, touch test only in the final evaluate step."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from . import split
from .common import iter_jsonl


def verify_and_test_qids(splits_dir: Path) -> frozenset[str]:
    """Call first in every tuning and eval entry point. Fails if any split file changed."""
    split.verify(splits_dir)
    return frozenset(r["qid"] for r in iter_jsonl(Path(splits_dir) / "test.jsonl"))


def assert_no_test(qids: Iterable[str], test_qids: frozenset[str], what: str) -> None:
    leaked = set(qids) & test_qids
    if leaked:
        raise AssertionError(f"{what} contains {len(leaked)} test qids, for example {sorted(leaked)[:3]}")
