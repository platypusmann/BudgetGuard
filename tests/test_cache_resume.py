"""Resume behaviour of the API cache, with a fake client. No network, no key, no money."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_script_resumes_and_never_pays_twice():
    """The original smoke test exercises sync resume, crash recovery and batch in-flight tracking."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    env.pop("OPENAI_API_KEY", None)
    out = subprocess.run([sys.executable, "tests/smoke_split_and_cache.py"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    text = out.stdout
    assert "refused overwrite OK" in text
    # first run: every 7th call fails with a 400. Resume only calls the failed ones
    line = next(l for l in text.splitlines() if l.startswith("after run1 ok:"))
    n1, n2, calls = (int(x) for x in line.replace(",", "").split() if x.isdigit())
    assert n2 > n1 and calls == n2 - n1
    assert "tier2: 0 prompts to run" in text
    assert "survived truncated line: True" in text
    assert "tier1: 0 prompts to submit" in text          # in-flight batch is not resubmitted
    assert "tier1: 1 prompts missing" in text            # the failed batch line is missing again


def test_done_ids_skips_errors_and_truncated_lines(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from budget_guard.common import JsonlAppender, done_ids

    p = tmp_path / "c.jsonl"
    with JsonlAppender(p) as w:
        w.write({"pid": "a", "error": None})
        w.write({"pid": "b", "error": "boom"})
    with p.open("a", encoding="utf-8") as f:
        f.write('{"pid": "c", "err')                        # killed mid write
    with JsonlAppender(p) as w:
        w.write({"pid": "b", "error": None})                 # retry succeeded
    assert done_ids(p) == {"a", "b"}
    assert b"\r\n" not in p.read_bytes()                     # LF on every OS, so hashes are portable


def test_latest_good_row_wins(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from budget_guard.common import JsonlAppender
    from budget_guard.grade import latest_good_rows

    p = tmp_path / "c.jsonl"
    with JsonlAppender(p) as w:
        w.write({"pid": "a", "answer": "old", "error": None})
        w.write({"pid": "a", "error": "later failure"})
        w.write({"pid": "a", "answer": "new", "error": None})
        w.write({"pid": "b", "error": "never fixed"})
    rows = list(latest_good_rows(p))
    assert [(r["pid"], r["answer"]) for r in rows] == [("a", "new")]
