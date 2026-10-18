"""One command per stage, the same on Windows, macOS and Linux.

  python scripts/bg.py synth          build the synthetic world in data/synthetic ($0, offline)
  python scripts/bg.py grade          grade every tier cache            [--data-dir data/synthetic]
  python scripts/bg.py features       join caches into the feature table and embed questions
  python scripts/bg.py evaluate       fit, tune, then evaluate on test, write results/ and figures
  python scripts/bg.py report         redraw figures and the README table from results/
  python scripts/bg.py test           run the offline test suite
  python scripts/bg.py synthetic      synth, grade, features, evaluate in one go
  python scripts/bg.py real           grade, features, evaluate on the real cache in data/

  python scripts/bg.py splits         build the frozen AmbigNQ splits (downloads from Hugging Face, once)
  python scripts/bg.py estimate       worst case API cost of everything still missing (spends nothing)
  python scripts/bg.py pilot-check    AUROC of each uncertainty feature and the clarify gain, pilot rows only
  python scripts/bg.py ambiguity-text free question-text ambiguity detector, trained on calib, scored on tune
  python scripts/bg.py probe-compare  optional paid probe against the free text detector, pilot questions
  python scripts/bg.py cache ...      any cache_api command, for example
                                        cache estimate --tier tier1 --pilot
                                        cache sync --tier tier2 --pilot --cap 0.25
                                        cache batch-submit --tier tier2 --cap 4.0

Anything after the stage name is passed through, for example
  python scripts/bg.py evaluate --data-dir data/synthetic --eval-split tune
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ["--data-dir", "data/synthetic"]


def py(*args: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, *args]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode:
        sys.exit(r.returncode)


def has_data_dir(extra: list[str]) -> bool:
    return any(a == "--data-dir" or a.startswith("--data-dir=") for a in extra)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    stage, extra = sys.argv[1], sys.argv[2:]
    if stage == "splits":
        py("-m", "budget_guard.split", *extra)
    elif stage == "synth":
        py("-m", "budget_guard.synthetic", *extra)
    elif stage == "grade":
        py("-m", "budget_guard.grade", *extra)
    elif stage == "features":
        py("-m", "budget_guard.features", *extra)
    elif stage == "evaluate":
        py("-m", "budget_guard.evaluate", *extra)
    elif stage == "report":
        py("-m", "budget_guard.report", *extra)
    elif stage == "test":
        py("-m", "pytest", *extra)
    elif stage == "probe-compare":
        py("-m", "budget_guard.ambiguity_probe", "compare", *extra)
    elif stage == "ambiguity-text":
        py("-m", "budget_guard.ambiguity_text", *extra)
    elif stage == "pilot-check":
        py("-m", "budget_guard.pilot_check", *extra)
    elif stage == "cache":
        py("-m", "budget_guard.cache_api", *extra)
    elif stage == "estimate":
        for tier in ("tier1", "tier2"):
            py("-m", "budget_guard.cache_api", "estimate", "--tier", tier, *extra)
    elif stage == "synthetic":
        py("-m", "budget_guard.synthetic")
        py("-m", "budget_guard.grade", *SYNTH)
        py("-m", "budget_guard.features", *SYNTH)
        py("-m", "budget_guard.evaluate", *SYNTH, *extra)
    elif stage == "real":
        if has_data_dir(extra):
            sys.exit("the real stage always reads data/")
        py("-m", "budget_guard.grade")
        py("-m", "budget_guard.features", "--embed", "minilm" if "--minilm" in extra else "hash")
        py("-m", "budget_guard.evaluate", *[a for a in extra if a != "--minilm"],
           *(["--embed", "minilm"] if "--minilm" in extra else []))
    else:
        sys.exit(f"unknown stage {stage}. Run python scripts/bg.py --help")


if __name__ == "__main__":
    main()
