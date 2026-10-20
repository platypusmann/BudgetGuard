"""One evaluation at a time, and a real one only into a fresh folder of its own.

Why: on 2026-10-20 an accidental real run and a synthetic run wrote into results/ at the same
time, and test-derived files ended up committed (DECISIONS.md 101 to 108). So:

  every run   takes a lock file (.evaluate.lock at the repo root) and refuses to start while
              another run holds it. The lock is released on any exit, Ctrl+C and errors included.
  real final  (real data, test split) also refuses when its output folder is shared with another
              kind of run (the same folder, inside it, or containing it) or is not empty.

`scripts/bg.py real` takes the lock before its grade and features steps, which write test-derived
files, and passes a token to the steps it starts, so the evaluate inside it is recognized as part of
the same run, not as a second one. A lock left by a run that was killed hard has to be deleted by
hand. The refusal message says where it is and who held it.
"""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .common import ROOT

LOCK = ROOT / ".evaluate.lock"
TOKEN_ENV = "BG_EVAL_LOCK_TOKEN"


def _related(a: Path, b: Path) -> bool:
    """True when a and b are the same folder or one contains the other."""
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or a in b.parents or b in a.parents


def _read(lock: Path) -> dict:
    try:
        return json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def check_output(out: Path, other_dirs: Iterable[Path]) -> None:
    """Refusals for a real final evaluation's output folder."""
    out = Path(out)
    for d in other_dirs:
        if _related(out, d):
            raise SystemExit(f"refusing to start a real evaluation: output folder {out} is shared with {d}, "
                             "which belongs to another kind of run. Use results/final or another fresh folder.")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to start a real evaluation: output folder {out} is not empty. "
                         "A real run writes only into a folder of its own.")


@contextmanager
def guard(out: Path, real_final: bool, other_dirs: Iterable[Path] = (), lock: Path | None = None,
          what: str = "evaluate") -> Iterator[None]:
    lock = Path(lock or LOCK)
    mine = os.environ.get(TOKEN_ENV)
    held = _read(lock) if lock.exists() else None
    nested = held is not None and mine is not None and held.get("token") == mine
    if held is not None and not nested:
        raise SystemExit(f"refusing to start: another evaluate run holds {lock} "
                         f"(pid {held.get('pid')}, {held.get('what')}, writing to {held.get('out')}, "
                         f"since {held.get('started')}). If no run is going, it was killed hard: "
                         "delete that file and try again.")
    if real_final:
        check_output(out, other_dirs)
    if nested:                 # part of a run that already holds the lock, which releases it
        yield
        return
    token = uuid.uuid4().hex
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"refusing to start: another evaluate run took {lock} a moment ago") from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"token": token, "pid": os.getpid(), "what": what, "out": str(Path(out).resolve()),
                   "real_final": real_final, "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}, f)
    os.environ[TOKEN_ENV] = token
    try:
        yield
    finally:
        os.environ.pop(TOKEN_ENV, None)
        if _read(lock).get("token") == token:
            lock.unlink(missing_ok=True)
