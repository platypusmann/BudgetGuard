"""Grade cached answers against AmbigNQ gold.

Three scores per answer, all max over gold aliases:
  em        SQuAD exact match after normalization
  f1        SQuAD token F1
  lenient   normalized gold appears as a contiguous token run inside a prediction
            of at most len(gold) + LENIENT_SLACK tokens, so "the Eiffel Tower in Paris"
            passes for gold "Eiffel Tower" but a rambling list of guesses does not
  lenient_plus  lenient after two more normalizations on both sides: ordinal suffixes are
            stripped from numbers (78th is 78) and full day-month-year dates are rewritten to
            one canonical token (1 July 2017, July 1 2017 and July 1, 2017 all match).
            Anything lenient accepts also passes. Nothing else: no synonyms, no partial
            dates, no looser containment

The primary metric is pre-registered in DECISIONS.md (lenient). EM and lenient_plus are
secondary. lenient_plus was added after the pilot hand check, see DECISIONS.md.

Every pid, raw or clarified, is graded against row["gold"] for its qid. For an ambiguous
question that is the intended interpretation's answers, so answering the raw question only
counts when the model happened to answer the reading the user meant.

  python -m budget_guard.grade                      # all tiers with a cache file
  python -m budget_guard.grade --tier tier0 --format jsonl
  python -m budget_guard.grade --data-dir data/synthetic
"""
from __future__ import annotations

import argparse
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

from .common import DATA, JsonlAppender, iter_jsonl

LENIENT_SLACK = 4
TIERS = ("tier0", "tier1", "tier2")
SPLIT_NAMES = ("calib", "tune", "test")

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_ASCII_PUNCT = set(string.punctuation)


def _is_punct(ch: str) -> bool:
    # SQuAD strips string.punctuation only. We also strip unicode punctuation so that
    # "1939–1945" and "1939-1945" normalize the same way. Logged in DECISIONS.md.
    return ch in _ASCII_PUNCT or unicodedata.category(ch).startswith("P")


def normalize(s: str | None) -> str:
    """SQuAD normalization: lowercase, drop punctuation, drop articles, squash whitespace."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if not _is_punct(ch))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _tokens(s: str | None) -> list[str]:
    return normalize(s).split()


def exact_match(pred: str | None, golds: Iterable[str]) -> bool:
    # an answer that normalizes to nothing, like "the", is never correct, unlike raw SQuAD
    p = normalize(pred)
    return bool(p) and any(p == normalize(g) for g in golds)


def _f1_one(pred_toks: list[str], gold_toks: list[str]) -> float:
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_toks)
    recall = same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str | None, golds: Iterable[str]) -> float:
    pt = _tokens(pred)
    if not pt:
        return 0.0
    return max((_f1_one(pt, gt) for gt in map(_tokens, golds) if gt), default=0.0)


def _contains(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def lenient_match(pred: str | None, golds: Iterable[str], slack: int = LENIENT_SLACK) -> bool:
    pt = _tokens(pred)
    if not pt:
        return False
    for g in golds:
        gt = _tokens(g)
        if gt and len(pt) <= len(gt) + slack and _contains(pt, gt):
            return True
    return False


_ORDINAL = re.compile(r"^(\d+)(st|nd|rd|th)$")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
_MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
                "oct": 10, "nov": 11, "dec": 12})


def _day(tok: str) -> int | None:
    return int(tok) if tok.isdigit() and len(tok) <= 2 and 1 <= int(tok) <= 31 else None


def _year(tok: str) -> int | None:
    return int(tok) if tok.isdigit() and len(tok) == 4 else None


def _plus_tokens(s: str | None) -> list[str]:
    """Tokens for lenient_plus: ordinals stripped, then full dates collapsed to one token.

    Only complete dates (day, month and four-digit year, in either order) are rewritten, so
    "1963" never matches "July 1, 1963" and "July 2017" is left alone.
    """
    toks = [_ORDINAL.sub(lambda m: m.group(1), t) for t in _tokens(s)]
    out, i = [], 0
    while i < len(toks):
        if i + 2 < len(toks):
            a, b, c = toks[i:i + 3]
            y = _year(c)
            if y is not None:
                if a in _MONTHS and _day(b) is not None:
                    out.append(f"date:{y:04d}-{_MONTHS[a]:02d}-{_day(b):02d}")
                    i += 3
                    continue
                if b in _MONTHS and _day(a) is not None:
                    out.append(f"date:{y:04d}-{_MONTHS[b]:02d}-{_day(a):02d}")
                    i += 3
                    continue
        out.append(toks[i])
        i += 1
    return out


def lenient_plus_match(pred: str | None, golds: Iterable[str], slack: int = LENIENT_SLACK) -> bool:
    """lenient, or the lenient rule on lenient_plus tokens.

    The "or" keeps every lenient match. Collapsing a full date would otherwise hide its year, so
    "September 14, 1814" would stop matching a gold of "1814", which lenient accepts. It adds no
    partial date matching of its own: "1963" still fails against "July 1, 1963".
    """
    if lenient_match(pred, golds, slack):
        return True
    pt = _plus_tokens(pred)
    if not pt:
        return False
    for g in golds:
        gt = _plus_tokens(g)
        if gt and len(pt) <= len(gt) + slack and _contains(pt, gt):
            return True
    return False


def grade_answer(pred: str | None, golds: list[str]) -> dict:
    return {
        "em": exact_match(pred, golds),
        "lenient": lenient_match(pred, golds),
        "lenient_plus": lenient_plus_match(pred, golds),
        "f1": round(token_f1(pred, golds), 6),
    }


# ---------------------------------------------------------------- cli

def gold_map(splits_dir: Path) -> dict[str, list[str]]:
    """qid to gold aliases. About 14k short lists on the real data, a few MB."""
    out: dict[str, list[str]] = {}
    for name in SPLIT_NAMES:
        for row in iter_jsonl(Path(splits_dir) / f"{name}.jsonl"):
            out[row["qid"]] = row["gold"]
    return out


def latest_good_rows(cache_file: Path) -> Iterable[dict]:
    """Last non error row per pid, same rule cache_api uses to decide what is done.

    Keeps only pid to line number in memory on the first pass, then streams again.
    """
    last: dict[str, int] = {}
    for i, row in enumerate(iter_jsonl(cache_file)):
        if row.get("pid") and not row.get("error"):
            last[row["pid"]] = i
    keep = set(last.values())
    for i, row in enumerate(iter_jsonl(cache_file)):
        if i in keep:
            yield row


def grade_tier(tier: str, data_dir: Path = DATA, fmt: str = "parquet") -> Path | None:
    data_dir = Path(data_dir)
    cache_file = data_dir / "cache" / f"{tier}.jsonl"
    if not cache_file.exists():
        print(f"{tier}: no cache at {cache_file}, skipped")
        return None
    golds = gold_map(data_dir / "splits")
    out_dir = data_dir / "graded"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, missing_gold = [], 0
    for r in latest_good_rows(cache_file):
        g = golds.get(r["qid"])
        if g is None:
            missing_gold += 1
            continue
        rows.append({
            "pid": r["pid"], "qid": r["qid"], "variant": r.get("variant"), "split": r.get("split"),
            "tier": tier, "answer_norm": normalize(r.get("answer")), **grade_answer(r.get("answer"), g),
        })

    if fmt == "jsonl":
        out = out_dir / f"{tier}.jsonl"
        out.unlink(missing_ok=True)
        with JsonlAppender(out, sync_every=5000) as w:
            for row in rows:
                w.write(row)
    else:
        import pandas as pd

        out = out_dir / f"{tier}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)

    n = max(1, len(rows))
    print(f"{tier}: graded {len(rows)} rows -> {out}  "
          f"em {sum(r['em'] for r in rows) / n:.3f}  lenient {sum(r['lenient'] for r in rows) / n:.3f}  "
          f"lenient_plus {sum(r['lenient_plus'] for r in rows) / n:.3f}"
          + (f"  ({missing_gold} rows with no gold skipped)" if missing_gold else ""))
    return out


def load_graded(tier: str, data_dir: Path = DATA):
    """Read a graded file in either format as a DataFrame."""
    import pandas as pd

    base = Path(data_dir) / "graded"
    if (base / f"{tier}.parquet").exists():
        return pd.read_parquet(base / f"{tier}.parquet")
    return pd.DataFrame(list(iter_jsonl(base / f"{tier}.jsonl")))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", action="append", choices=TIERS, help="repeatable, default all three")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--format", choices=["parquet", "jsonl"], default="parquet")
    args = ap.parse_args(argv)
    for tier in args.tier or TIERS:
        grade_tier(tier, args.data_dir, args.format)


if __name__ == "__main__":
    main()
