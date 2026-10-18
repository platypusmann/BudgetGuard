"""The frozen experiment design: its hash, the lock file, and the rules it fixes.

configs/experiment.yaml was frozen on 2026-10-19, before any real cache was evaluated. The hash is
taken over LF-normalized bytes, because git on Windows rewrites line endings on checkout and the
same design must hash the same everywhere. configs/experiment.sha256 records the frozen hash.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .common import CONFIGS

DESIGN = CONFIGS / "experiment.yaml"
LOCK = CONFIGS / "experiment.sha256"


def design_hash(path: Path = DESIGN) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def locked_hash(lock: Path = LOCK) -> str | None:
    return Path(lock).read_text(encoding="utf-8").split()[0] if Path(lock).exists() else None


def check(path: Path = DESIGN, lock: Path = LOCK) -> tuple[str, bool]:
    """Hash of the design in use, and whether it matches the frozen one."""
    h = design_hash(path)
    return h, h == locked_hash(lock)


def choose_text_bins(scores: dict[int, float], min_gain_points: float) -> int:
    """The frozen rule for the DP's number of text bins.

    Start at the fewest bins. Walk up the grid and move to a larger setting only when its tune
    score beats the current choice by at least min_gain_points accuracy points. Scores come
    from tune streams only.
    """
    grid = sorted(scores)
    best = grid[0]
    for T in grid[1:]:
        if scores[T] - scores[best] >= min_gain_points / 100 - 1e-12:
            best = T
    return best
