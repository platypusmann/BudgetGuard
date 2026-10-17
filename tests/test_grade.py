import pytest

from budget_guard.grade import exact_match, grade_answer, lenient_match, normalize, token_f1


@pytest.mark.parametrize("raw, norm", [
    ("The Eiffel Tower.", "eiffel tower"),
    ("  a   Tale of Two Cities ", "tale of two cities"),
    ("1939–1945", "19391945"),             # unicode en dash is punctuation too
    ("“Quoted”", "quoted"),
    ("Theodore", "theodore"),              # articles only as whole words
    ("", ""),
    (None, ""),
])
def test_normalize(raw, norm):
    assert normalize(raw) == norm


def test_exact_match_aliases_and_case():
    assert exact_match("paris", ["London", "Paris"])
    assert exact_match("The Beatles", ["Beatles"])
    assert not exact_match("Paris, France", ["Paris"])


def test_dash_variants_match():
    assert exact_match("1939-1945", ["1939–1945"])


def test_empty_answers_never_correct():
    for pred in ("", "   ", "the", "!!!", None):
        g = grade_answer(pred, ["a"])
        assert not g["em"] and not g["lenient"] and g["f1"] == 0.0


def test_gold_that_normalizes_to_empty_is_ignored():
    assert not exact_match("the", ["The"])
    assert not lenient_match("something", ["."])


def test_lenient_allows_short_extras():
    assert lenient_match("The Eiffel Tower in Paris", ["Eiffel Tower"])         # 4 tokens vs 2, ok
    assert lenient_match("It was 1969", ["1969"])
    assert lenient_match("Neil Armstrong (astronaut)", ["Neil Armstrong"])


def test_lenient_length_cap():
    gold = ["Eiffel Tower"]                                                      # 2 tokens, cap 6
    assert lenient_match("eiffel tower a b c d", gold)                          # "a" is dropped, 5 tokens
    assert lenient_match("eiffel tower x y z w", gold)                          # 6 tokens, at the cap
    assert not lenient_match("eiffel tower x y z w v", gold)                    # 7 tokens, over
    assert not lenient_match("Louvre, Notre Dame, Arc de Triomphe or the Eiffel Tower", gold)


def test_lenient_is_token_based_not_substring():
    assert not lenient_match("1990", ["1"])
    assert not lenient_match("Parisian", ["Paris"])
    assert lenient_match("in 1 day", ["1"])


def test_lenient_needs_contiguous_gold():
    assert not lenient_match("Tower Eiffel", ["Eiffel Tower"])


def test_f1():
    assert token_f1("Eiffel Tower", ["Eiffel Tower"]) == 1.0
    assert token_f1("The Eiffel Tower in Paris", ["Eiffel Tower"]) == pytest.approx(2 * (2 / 4) * 1 / (2 / 4 + 1))
    assert token_f1("Louvre", ["Eiffel Tower"]) == 0.0
    assert token_f1("tower", ["Eiffel Tower", "Tower"]) == 1.0                 # max over aliases


def test_em_implies_lenient():
    for pred, gold in [("Paris", ["paris"]), ("the beatles", ["Beatles"]), ("42", ["42"])]:
        g = grade_answer(pred, gold)
        assert g["em"] and g["lenient"]


def test_grade_cli_writes_per_tier(world):
    import pandas as pd

    for tier in ("tier0", "tier1", "tier2"):
        g = pd.read_parquet(world / "graded" / f"{tier}.parquet")
        assert {"pid", "qid", "em", "lenient", "f1", "answer_norm"} <= set(g.columns)
        assert g["pid"].is_unique                                               # error rows deduped
        assert (g["lenient"] >= g["em"]).all()


def test_grade_cli_jsonl(world, tmp_path):
    import shutil

    from budget_guard import grade
    from budget_guard.common import iter_jsonl

    shutil.copytree(world / "splits", tmp_path / "splits")
    shutil.copytree(world / "cache", tmp_path / "cache")
    grade.main(["--data-dir", str(tmp_path), "--tier", "tier0", "--format", "jsonl"])
    rows = list(iter_jsonl(tmp_path / "graded" / "tier0.jsonl"))
    assert rows and "lenient" in rows[0]
