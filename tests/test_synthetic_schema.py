"""The synthetic cache must look exactly like the real one, or the real run breaks later."""
import re
from pathlib import Path

from budget_guard import cache_api
from budget_guard.common import iter_jsonl, load_yaml, CONFIGS

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_tier0.py"
LP_KEYS = ["lp_mean", "lp_min", "lp_sum", "p_first", "ent_first_top"]


def _good_rows(path):
    return [r for r in iter_jsonl(path) if not r.get("error")]


def test_api_rows_match_cache_row_keys(world):
    cfg = load_yaml(CONFIGS / "tiers.yaml")["tier2"]
    body = {"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": "x"},
                                       "logprobs": {"content": [{"token": "x", "logprob": -0.1, "top_logprobs": []}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    p = {"pid": "q::orig", "qid": "q", "variant": "orig", "split": "calib"}
    expected = list(cache_api.cache_row("tier2", cfg, p, body, batch=True))
    for tier in ("tier1", "tier2"):
        for r in _good_rows(world / "cache" / f"{tier}.jsonl")[:50]:
            assert list(r) == expected


def test_tier0_rows_match_notebook_keys(world):
    src = NOTEBOOK.read_text(encoding="utf-8")
    block = src[src.index("rows.append({"):src.index("})", src.index("rows.append({"))]
    keys = re.findall(r'"(\w+)":', block)
    star = block.index("**feats")
    before = len(re.findall(r'"(\w+)":', block[:star]))
    expected = keys[:before] + LP_KEYS + keys[before:]
    for r in _good_rows(world / "cache" / "tier0.jsonl")[:50]:
        assert list(r) == expected
        assert len(r["samples"]) == load_yaml(CONFIGS / "tiers.yaml")["tier0"]["n_samples"]
        assert 0 <= r["sc_agree"] <= 1


def test_error_rows_are_followed_by_good_ones(world):
    seen_err, good = set(), set()
    for r in iter_jsonl(world / "cache" / "tier2.jsonl"):
        (seen_err if r.get("error") else good).add(r["pid"])
    assert seen_err <= good


def test_accuracies_roughly_realistic(world):
    import pandas as pd

    from budget_guard.common import iter_jsonl as it

    amb = {r["qid"]: r["is_ambiguous"] for s in ("calib", "tune", "test") for r in it(world / "splits" / f"{s}.jsonl")}
    targets = {"tier0": 0.25, "tier1": 0.40, "tier2": 0.55}
    for tier, target in targets.items():
        g = pd.read_parquet(world / "graded" / f"{tier}.parquet")
        g["amb"] = g["qid"].map(amb)
        clear = g[(g["variant"] == "clar") | ~g["amb"]]["lenient"].mean()
        raw_amb = g[(g["variant"] == "orig") & g["amb"]]["lenient"].mean()
        assert abs(clear - target) < 0.08, (tier, clear)
        assert raw_amb < clear - 0.1, (tier, raw_amb, clear)


def test_splits_fixture_is_real_format(world):
    from budget_guard import split

    split.verify(world / "splits")
    row = next(iter_jsonl(world / "splits" / "calib.jsonl"))
    assert {"qid", "question", "is_ambiguous", "gold", "intended_question", "interpretations",
            "time_sensitive", "split"} <= set(row)
    p = next(iter_jsonl(world / "splits" / "prompts.jsonl"))
    assert p["pid"].endswith(("::orig", "::clar"))
