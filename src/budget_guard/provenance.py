"""Which code produced a run: git commit, exact tag and whether any code file had uncommitted edits.

evaluate.py prints this in its header and stores it in run_info.json, so the final test numbers can
be tied to the frozen commit tagged pre-test-eval. Data and results folders don't count as code.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .common import ROOT

CODE_PATHS = ("src", "scripts", "configs", "tests", "notebooks")


def _git(*args: str, root: Path = ROOT) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state(root: Path = ROOT) -> dict:
    commit = _git("rev-parse", "HEAD", root=root)
    tag = _git("describe", "--tags", "--exact-match", "HEAD", root=root)
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS, root=root)
    return {"commit": commit or "unknown", "tag": tag or None,
            "dirty_code": bool(dirty) if dirty is not None else None}


def describe(state: dict) -> str:
    tag = f", tag {state['tag']}" if state["tag"] else ", no tag on this commit"
    if state["dirty_code"]:
        return f"code commit {state['commit']}{tag}, WITH UNCOMMITTED CODE CHANGES, so not reproducible from the commit"
    if state["dirty_code"] is None:
        return f"code commit {state['commit']}{tag}, could not check for uncommitted changes"
    return f"code commit {state['commit']}{tag}, clean"
