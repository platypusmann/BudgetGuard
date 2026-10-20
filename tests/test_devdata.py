"""The calib-and-tune-only dry-run path never reads, embeds or grades anything from test."""
import json
import shutil

import pandas as pd
import pytest

from budget_guard import completeness, devdata, evaluate
from budget_guard.common import iter_jsonl
from budget_guard.features import _text_key, load_embeddings
from budget_guard.leakage import verify_and_test_qids


@pytest.fixture()
def copy(world, tmp_path):
    d = tmp_path / "w"
    shutil.copytree(world, d, ignore=shutil.ignore_patterns("dev", "features", "graded"))
    return d


def _test_split(d):
    """Read only by the test itself, to know what must not appear."""
    test = list(iter_jsonl(d / "splits" / "test.jsonl"))
    others = {r["question"] for s in ("calib", "tune") for r in iter_jsonl(d / "splits" / f"{s}.jsonl")}
    return {r["qid"] for r in test}, {r["question"] for r in test} - others


@pytest.fixture()
def decode_spy(monkeypatch):
    """Records every JSON string decoded through json.loads while the code under test runs."""
    seen = []
    real = json.loads

    def spy(s, *a, **k):
        seen.append(s if isinstance(s, str) else s.decode("utf-8", "replace"))
        return real(s, *a, **k)
    monkeypatch.setattr(json, "loads", spy)
    return seen


def test_dev_artifacts_hold_no_test_qid_or_text(copy):
    test_qids, test_only_text = _test_split(copy)
    root = devdata.build(copy, "hash")
    f = pd.read_parquet(root / "features" / "features.parquet")
    assert set(f["split"]) == {"calib", "tune"}
    assert not (set(f["qid"]) & test_qids)
    assert not (set(f["question"]) & test_only_text)
    for tier in ("tier0", "tier1", "tier2"):
        g = pd.read_parquet(root / "graded" / f"{tier}.parquet")
        assert set(g["split"]) <= {"calib", "tune"} and not ({p.split("::")[0] for p in g["pid"]} & test_qids)
    idx = json.loads((root / "features" / "emb_hash_qids.json").read_text())
    assert not (set(idx["qids"]) & test_qids)
    assert not (set(idx["text_sha1"]) & {_text_key(t) for t in test_only_text})
    assert set(load_embeddings(root, "hash")) == set(f["qid"])
    devdata.assert_clean(copy, frozenset(test_qids))
    # and nothing was written to the final paths
    assert not (copy / "features").exists() and not (copy / "graded").exists()


def test_no_test_line_is_ever_decoded(copy, decode_spy):
    test_qids, _ = _test_split(copy)
    decode_spy.clear()
    tq = verify_and_test_qids(copy / "splits")
    assert tq == frozenset(test_qids)
    completeness.check_cache(copy, splits=devdata.DEV_SPLITS)
    devdata.build(copy, "hash")
    leaked = [s[:80] for s in decode_spy if any(q in s for q in test_qids)]
    assert decode_spy and not leaked, leaked[:3]
    # the spy can fail: the full completeness check, which belongs to the final path, does decode test rows
    completeness.check_cache(copy)
    assert any(any(q in s for q in test_qids) for s in decode_spy)


def test_dry_run_evaluate_never_decodes_test(copy, tmp_path, decode_spy):
    test_qids, _ = _test_split(copy)
    decode_spy.clear()
    out = tmp_path / "dry"
    evaluate.main(["--data-dir", str(copy), "--out", str(out), "--eval-split", "tune", "--quick", "--no-sweeps",
                   "--no-figures"])
    leaked = [s[:80] for s in decode_spy if any(q in s for q in test_qids)]
    assert not leaked, leaked[:3]
    info = json.loads((out / "run_info.json").read_text())
    assert info["mode"].startswith("dry run") and info["eval_split"] == "tune"
    assert set(info["pool_sizes"]) == {"calib", "tune"}
    from budget_guard.dry_run_report import report

    text = report(out)
    assert "Masked policies all at 0: yes" in text and "Result: COMPLETE" in text
    from budget_guard.report import write_table

    assert "DRY RUN" in write_table(out).read_text(encoding="utf-8")


def test_final_path_still_covers_test(world):
    test_qids, _ = _test_split(world)
    f = pd.read_parquet(world / "features" / "features.parquet")
    assert test_qids <= set(f["qid"])
