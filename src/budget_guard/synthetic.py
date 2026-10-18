"""Synthetic AmbigNQ splits and tier caches, for development and tests only. Spends nothing.

Everything goes through the real code paths:
  splits  raw AmbigNQ-shaped rows are fed to split.build, so the fixture has the real format,
          the real intent hashing and a real MANIFEST
  tier1/2 rows come from cache_api.cache_row applied to a fake chat completion body,
          so the schema is the real one by construction
  tier0   rows copy the Kaggle notebook's row dict key for key, self consistency samples included

Answers are strings, not labels. grade.py decides what is correct, the same as on real data.

The latent model, recalibrated on 2026-10-19 to the real 300-question pilot (PILOT_TARGETS):
  skill_k = a_k - L_k * z - d_k * is_ambiguous + N_k * e_k
z is a difficulty shared by every tier. Tier 1 and 2 load fully on it, have little noise of their
own and share one success draw, because on the pilot they are right together on 42 percent of raw
questions and Tier 2 is right where Tier 1 is wrong on only 6 percent. Tier 0 barely loads on z
and has its own draw, because its confidence says nothing about the paid tiers on the pilot
(AUROC 0.42 to 0.56). a_k is solved so clear-question accuracy hits the pilot, d_k so clarified
ambiguous accuracy does (ambiguous questions are harder even once clarified), and on the raw
prompt of an ambiguous question the model lands on the intended reading with probability r_k,
the pilot's raw over clarified ratio.

Confidence is a noisy, overconfident view of skill, tuned to the pilot's AUROC for each tier's
own correctness (about 0.7). Nothing in it, and nothing in the Tier 0 self consistency samples,
depends on ambiguity beyond the harder questions, so like on the pilot the model signals carry
almost no ambiguity information. The question text does: template ambiguity rates are spread just
enough that the free text detector lands near the real 0.65.

  python -m budget_guard.synthetic                     # writes data/synthetic
  python -m budget_guard.synthetic --n-train 600 --n-val 200 --out tests/_tmp/synth
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
from pathlib import Path

import numpy as np

from . import cache_api, split
from .common import CONFIGS, DATA, JsonlAppender, iter_jsonl, load_yaml, stable_hash
from .prompts import PROMPT_VERSION, SYSTEM, user_message

SYNTH_DIR = DATA / "synthetic"
# real pilot, lenient accuracy on 300 calib questions (DECISIONS.md 61 and 66)
PILOT_TARGETS = {
    "tier0": {"clear": 0.180, "clarified": 0.109, "raw_ambiguous": 0.062},
    "tier1": {"clear": 0.570, "clarified": 0.422, "raw_ambiguous": 0.258},
    "tier2": {"clear": 0.645, "clarified": 0.578, "raw_ambiguous": 0.273},
}
LOADING = {"tier0": 0.1, "tier1": 1.0, "tier2": 1.0}      # weight on the shared difficulty z
OWN_NOISE = {"tier0": 1.0, "tier1": 0.15, "tier2": 0.15}  # tier-specific skill noise
SHARED_DRAW = ("tier1", "tier2")                          # one success draw and one reading draw for both
# share of the ambiguity difficulty that confidence does NOT see. 1 means the model is exactly as
# sure on ambiguous questions as on clear ones. Pilot: Tier 0 and 1 show no drop, Tier 2 a small one
CONF_AMB_BLIND = {"tier0": 1.0, "tier1": 1.0, "tier2": 0.5}
# confidence = gain * skill + bias + noise. bias > 0 means overconfident, as LLMs are
CONF = {"tier0": (0.8, 1.0, 0.45), "tier1": (0.9, 1.0, 0.45), "tier2": (1.0, 1.0, 0.4)}
SC_SHIFT = 2.0             # Tier 0 samples agree with probability sigmoid(conf + SC_SHIFT)
AMB_SPREAD = 0.9           # template ambiguity rates pulled toward AMB_CENTER, text detector near 0.65
AMB_CENTER = 0.475
ERROR_ROW_RATE = 0.005  # api tiers only, a failed call followed by a good retry

# question templates: (text, p_ambiguous, difficulty offset)
TEMPLATES = [
    ("who played {person} in {work}", 0.55, 0.0),
    ("when did {work} come out", 0.60, -0.2),
    ("who won the {comp} in {year}", 0.35, 0.1),
    ("how many {thing} are in {place}", 0.45, 0.3),
    ("where is {place} located", 0.25, -0.3),
    ("who wrote {work}", 0.40, 0.0),
    ("what is the population of {place}", 0.50, 0.4),
    ("who is the current {role} of {place}", 0.75, 0.6),
    ("when was the last time {team} won the {comp}", 0.70, 0.5),
    ("who sings {work}", 0.45, -0.1),
]
ROLES = ["president", "coach", "captain", "mayor", "chairman", "head", "governor"]
THINGS = ["counties", "players", "islands", "seats", "stations", "lakes", "districts"]
SYLL = ["ka", "lor", "ven", "mi", "dra", "tus", "bel", "or", "an", "sik", "ru", "mae", "tor",
        "li", "quen", "zar", "pe", "dol", "ri", "fa", "nor", "ex", "ul", "vi"]


def _name(rng, parts=2, syl=(2, 3)) -> str:
    words = []
    for _ in range(parts):
        n = int(rng.integers(syl[0], syl[1] + 1))
        words.append("".join(rng.choice(SYLL, n)).capitalize())
    return " ".join(words)


def _fill(template: str, rng) -> str:
    return template.format(
        person=_name(rng), work=_name(rng, int(rng.integers(1, 4))), comp=_name(rng, 1) + " Cup",
        year=int(rng.integers(1950, 2024)), thing=rng.choice(THINGS), place=_name(rng, 1),
        role=rng.choice(ROLES), team=_name(rng, 1),
    )


def _answer(rng) -> list[str]:
    """Gold aliases. Sometimes a second alias, like a short form."""
    kind = rng.random()
    if kind < 0.25:
        main = str(int(rng.integers(1900, 2024)))
        return [main]
    if kind < 0.40:
        return [str(int(rng.integers(2, 900)))]
    main = _name(rng, int(rng.integers(1, 4)))
    if " " in main and rng.random() < 0.3:
        return [main, main.split()[-1]]
    return [main]


def amb_rate(p_template: float) -> float:
    return AMB_CENTER + AMB_SPREAD * (p_template - 0.5)


def _qrng(seed: int, qid: str, tag: str):
    return np.random.default_rng([seed, stable_hash(qid, f"synth-{tag}") % (2 ** 32)])


# ---------------------------------------------------------------- raw rows

def raw_rows(hf_split: str, n: int, seed: int, latent: dict) -> list[dict]:
    rows = []
    prefix = "str" if hf_split == "train" else "sva"
    for i in range(n):
        qid = f"{prefix}{seed}_{i}"
        rng = _qrng(seed, qid, "raw")
        t = int(rng.integers(len(TEMPLATES)))
        tmpl, p_amb, d_off = TEMPLATES[t]
        question = _fill(tmpl, rng)
        z = float(rng.normal(d_off, 1.0))
        if rng.random() < amb_rate(p_amb):
            k = int(rng.choice([2, 2, 2, 3, 3, 4]))
            qs = [f"{question} ({_name(rng, 1).lower()} reading {j + 1})" for j in range(k)]
            ans = [_answer(rng) for _ in range(k)]
            ann = {"type": ["multipleQAs"], "answer": [[]],
                   "qaPairs": [{"question": qs, "answer": ans}]}
        else:
            ann = {"type": ["singleAnswer"], "answer": [_answer(rng)],
                   "qaPairs": [{"question": [], "answer": []}]}
        latent[qid] = {"z": z, "template": t}
        rows.append({"id": qid, "question": question, "annotations": ann})
    return rows


# ---------------------------------------------------------------- answers and features

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _bisect(f, target: float, lo: float = -8.0, hi: float = 8.0, increasing: bool = True) -> float:
    for _ in range(60):
        mid = (lo + hi) / 2
        low = f(mid) < target if increasing else f(mid) > target
        lo, hi = (mid, hi) if low else (lo, mid)
    return (lo + hi) / 2


def solve_abilities(seed: int = 0, n: int = 60000) -> dict[str, tuple[float, float, float]]:
    """Per tier (a, d, r): a hits clear accuracy, d clarified ambiguous accuracy, r the raw ratio."""
    rng = np.random.default_rng([seed, 99])
    t = rng.integers(len(TEMPLATES), size=n)
    offs = np.array([o for _, _, o in TEMPLATES])[t]
    amb = rng.random(n) < np.array([amb_rate(pa) for _, pa, _ in TEMPLATES])[t]
    z = rng.normal(offs, 1.0)
    out = {}
    for tier, tg in PILOT_TARGETS.items():
        e = rng.normal(0, 1, n)
        base = -LOADING[tier] * z + OWN_NOISE[tier] * e
        a = _bisect(lambda x: np.mean(1 / (1 + np.exp(-(x + base[~amb])))), tg["clear"])
        d = _bisect(lambda x: np.mean(1 / (1 + np.exp(-(a - x + base[amb])))), tg["clarified"], -4, 8, increasing=False)
        out[tier] = (a, d, tg["raw_ambiguous"] / tg["clarified"])
    return out


def _render_correct(gold: str, rng) -> str:
    """A correct answer as a chat model might phrase it. All pass lenient, some fail EM."""
    u = rng.random()
    if u < 0.68:
        return gold
    if u < 0.78:
        return "The " + gold
    if u < 0.86:
        return f"{gold} ({_name(rng, 1)})"
    if u < 0.92:
        return gold.upper() if len(gold) < 12 else gold.lower()
    return f"It was {gold}"


def _distractor(rng, avoid: set[str]) -> str:
    while True:
        a = _answer(rng)[0]
        if a.lower() not in avoid:
            return a


def _tok_pieces(text: str) -> list[str]:
    """Rough tokens, about four characters each, at least one."""
    return [text[i:i + 4] for i in range(0, max(1, len(text)), 4)] or [text]


def _logprob_content(answer: str, conf: float, rng) -> list[dict]:
    """Fake OpenAI logprobs.content for an answer given a confidence logit."""
    pieces = _tok_pieces(answer)
    p1 = min(max(_sigmoid(conf), 0.02), 0.999)
    rest = rng.dirichlet(np.ones(4)) * (1 - p1)
    top = [{"token": pieces[0], "logprob": math.log(p1)}] + [
        {"token": f"alt{j}", "logprob": math.log(max(float(r), 1e-9))} for j, r in enumerate(rest)]
    content = [{"token": pieces[0], "logprob": math.log(p1), "top_logprobs": top}]
    for piece in pieces[1:]:
        p = min(max(_sigmoid(conf + 2.5 + rng.normal(0, 0.8)), 0.05), 0.9999)
        content.append({"token": piece, "logprob": math.log(p), "top_logprobs": []})
    return content


def _nb_normalize(s: str) -> str:
    """Copy of the Kaggle notebook's normalize, used only for self consistency agreement."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _n_in(text: str, tier: str) -> int:
    chars = len(SYSTEM) + len(user_message(text))
    # Qwen's chat template adds more scaffolding tokens than the OpenAI chat format
    return math.ceil(chars / 4.3) + (19 if tier == "tier0" else 7)


def simulate_answer(row: dict, variant: str, tier: str, abilities: dict, latent: dict, seed: int) -> dict:
    """Decide what tier says for one prompt. Shared randomness ties orig and clar together."""
    qid = row["qid"]
    lat = latent[qid]
    a, d, r = abilities[tier]
    rq = _qrng(seed, qid, tier)                       # same draws for orig and clar of this tier
    skill = a - LOADING[tier] * lat["z"] - d * row["is_ambiguous"] + OWN_NOISE[tier] * rq.normal()
    u_success = _qrng(seed, qid, "shared").random() if tier in SHARED_DRAW else rq.random()
    gain, bias, noise = CONF[tier]
    seen = skill + CONF_AMB_BLIND[tier] * d * row["is_ambiguous"]
    conf = gain * seen + bias + rq.normal(0, noise)
    rv = _qrng(seed, f"{qid}::{variant}", tier)       # per prompt draws for wording
    gold = row["gold"]
    avoid = {g.lower() for g in gold}

    knows = u_success < _sigmoid(skill)
    reading_ok = True
    other_answers: list[str] = []
    if row["is_ambiguous"]:
        other_answers = [it["answers"][0] for j, it in enumerate(row["interpretations"]) if j != row["intended_index"]]
        if variant == "orig":
            u_read = _qrng(seed, qid, "reading").random() if tier in SHARED_DRAW else rv.random()
            reading_ok = u_read < r

    if knows and reading_ok:
        answer = _render_correct(str(rv.choice(gold)), rv)
    elif knows and other_answers:
        answer = str(rv.choice(other_answers))        # right fact, wrong reading
    else:
        answer = _distractor(rv, avoid)
    return {"answer": answer, "conf": conf, "rv": rv, "other_answers": other_answers, "avoid": avoid,
            "knows": knows, "reading_ok": reading_ok}


def api_row(tier: str, cfg: dict, prompt: dict, sim: dict, batch: bool) -> dict:
    """Build a fake chat completion body and let cache_api turn it into the real row format."""
    answer = sim["answer"]
    content = _logprob_content(answer, sim["conf"], sim["rv"])
    body = {
        "model": f"{cfg['model']}-2026-09-01",
        "choices": [{"finish_reason": "stop", "message": {"content": answer},
                     "logprobs": {"content": content}}],
        "usage": {"prompt_tokens": _n_in(prompt["text"], tier),
                  "completion_tokens": min(len(content), cfg["max_completion_tokens"])},
    }
    return cache_api.cache_row(tier, cfg, prompt, body, batch=batch)


def tier0_row(cfg: dict, prompt: dict, sim: dict) -> dict:
    """Same keys, same order and same formulas as notebooks/kaggle_tier0.py run_batch."""
    rv = sim["rv"]
    answer = sim["answer"]
    content = _logprob_content(answer, sim["conf"], rv)
    feats = cache_api.logprob_features(content)
    n_in, n_out = _n_in(prompt["text"], "tier0"), min(len(content), cfg["max_new_tokens"])
    samples = []
    # no ambiguity term: on the pilot self consistency carried no ambiguity signal (AUROC 0.52)
    agree_p = _sigmoid(sim["conf"] + SC_SHIFT)
    for _ in range(cfg["n_samples"]):
        u = rv.random()
        if u < agree_p:
            samples.append(answer)
        else:
            samples.append(_distractor(rv, sim["avoid"]))
    norm = _nb_normalize(answer)
    agree = sum(_nb_normalize(x) == norm for x in samples) / len(samples)
    distinct = len({_nb_normalize(x) for x in samples})
    return {
        "pid": prompt["pid"], "qid": prompt["qid"], "variant": prompt["variant"], "split": prompt["split"],
        "tier": "tier0", "model": cfg["model"], "model_returned": cfg["model"], "prompt_version": PROMPT_VERSION,
        "mode": "kaggle", "answer": answer, "finish_reason": "length" if n_out >= cfg["max_new_tokens"] else "stop",
        "n_in": int(n_in), "n_out": n_out,
        "usd": round((n_in * cfg["nominal_price_in"] + n_out * cfg["nominal_price_out"]) / 1e6, 10),
        **feats, "sc_agree": agree, "sc_distinct": distinct, "samples": samples,
        "ts": "2026-10-17T00:00:00Z", "error": None,
    }


# ---------------------------------------------------------------- build

def build_world(out_dir: Path = SYNTH_DIR, n_train: int = 10036, n_val: int = 2002, seed: int = 0,
                pilot_n: int = split.PILOT_N, tiers: tuple[str, ...] = ("tier0", "tier1", "tier2")) -> dict:
    out_dir = Path(out_dir)
    splits_dir, cache_dir = out_dir / "splits", out_dir / "cache"
    latent: dict[str, dict] = {}
    raw = {"train": raw_rows("train", n_train, seed, latent),
           "validation": raw_rows("validation", n_val, seed, latent)}
    manifest = split.build(out_dir=splits_dir, source=lambda s: iter(raw[s]), force=True,
                           pilot_n=min(pilot_n, int(n_train * 0.7 * 0.5)))
    del raw

    cfgs = load_yaml(CONFIGS / "tiers.yaml")
    abilities = solve_abilities(seed)
    rows_by_qid = {}
    for name in ("calib", "tune", "test"):
        for r in iter_jsonl(splits_dir / f"{name}.jsonl"):
            rows_by_qid[r["qid"]] = r

    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for tier in tiers:
        path = cache_dir / f"{tier}.jsonl"
        path.unlink(missing_ok=True)
        cfg = cfgs[tier]
        n = 0
        with JsonlAppender(path, sync_every=5000) as w:
            for p in iter_jsonl(splits_dir / "prompts.jsonl"):
                row = rows_by_qid[p["qid"]]
                sim = simulate_answer(row, p["variant"], tier, abilities, latent, seed)
                if tier == "tier0":
                    w.write(tier0_row(cfg, p, sim))
                else:
                    if sim["rv"].random() < ERROR_ROW_RATE:
                        w.write({"pid": p["pid"], "qid": p["qid"], "tier": tier,
                                 "ts": cache_api.now(), "error": "RateLimitError: synthetic"})
                    w.write(api_row(tier, cfg, p, sim, batch=not p["pilot"]))
                n += 1
        counts[tier] = n

    info = {"synthetic": True, "seed": seed, "n_train": n_train, "n_val": n_val,
            "abilities": {k: {"a": v[0], "d": v[1], "r": v[2]} for k, v in abilities.items()},
            "pilot_targets": PILOT_TARGETS, "rows": counts,
            "split_stats": manifest["stats"]}
    (out_dir / "SYNTHETIC").write_text(json.dumps(info, indent=2))
    return info


def is_synthetic(data_dir: Path) -> bool:
    # is_file, not exists: on Windows data/SYNTHETIC case-insensitively matches the data/synthetic
    # folder, which would stamp real results as synthetic
    return (Path(data_dir) / "SYNTHETIC").is_file()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=SYNTH_DIR)
    ap.add_argument("--n-train", type=int, default=10036)   # AmbigNQ light train size
    ap.add_argument("--n-val", type=int, default=2002)      # AmbigNQ light validation size
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot-n", type=int, default=split.PILOT_N)
    args = ap.parse_args(argv)
    info = build_world(args.out, args.n_train, args.n_val, args.seed, args.pilot_n)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
