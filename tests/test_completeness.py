"""evaluate.py must refuse an incomplete or inconsistent real cache."""
import json
import shutil

import pytest

from budget_guard import completeness, evaluate
from budget_guard.common import iter_jsonl


@pytest.fixture()
def real_copy(world, tmp_path):
    d = tmp_path / "real"
    shutil.copytree(world, d)
    (d / "SYNTHETIC").unlink()          # a real-looking cache
    return d


def _rewrite(path, fn):
    rows = list(iter_jsonl(path))
    rows = fn(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_complete_synthetic_cache_passes(world):
    problems, report = completeness.check_cache(world)
    assert problems == []
    assert report["tier2"]["retried"] > 0              # the synthetic cache has fixed retries, which are fine


def test_missing_pid_is_a_problem(real_copy):
    _rewrite(real_copy / "cache" / "tier1.jsonl", lambda rows: rows[1:])
    problems, _ = completeness.check_cache(real_copy)
    assert any("tier1" in p and "no row" in p for p in problems)


def test_unresolved_error_is_a_problem_but_a_fixed_retry_is_not(real_copy):
    def add_error(rows):
        last = rows[-1]
        return rows + [{"pid": last["pid"], "qid": last["qid"], "tier": "tier2", "error": "RateLimitError"}]
    _rewrite(real_copy / "cache" / "tier2.jsonl", add_error)
    problems, _ = completeness.check_cache(real_copy)
    assert any("tier2" in p and "latest row is an error" in p for p in problems)


def test_wrong_prompt_version_is_a_problem(real_copy):
    def bump(rows):
        rows[0] = {**rows[0], "prompt_version": "v2"}
        return rows
    _rewrite(real_copy / "cache" / "tier0.jsonl", bump)
    problems, _ = completeness.check_cache(real_copy)
    assert any("tier0" in p and "prompt_version" in p for p in problems)


def test_missing_tier_file_is_a_problem(real_copy):
    (real_copy / "cache" / "tier1.jsonl").unlink()
    problems, _ = completeness.check_cache(real_copy)
    assert any("tier1: no cache file" in p for p in problems)


def test_evaluate_refuses_incomplete_real_cache(real_copy, tmp_path):
    _rewrite(real_copy / "cache" / "tier1.jsonl", lambda rows: rows[1:])
    with pytest.raises(SystemExit, match="incomplete real cache"):
        evaluate.main(["--data-dir", str(real_copy), "--out", str(tmp_path / "o")])
