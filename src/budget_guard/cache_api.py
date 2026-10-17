"""Answer every prompt once per API tier and cache it forever. Resumable, budget guarded.

Commands
  estimate       count missing prompts and print worst case cost. Spends nothing.
  sync           call the API directly (standard price). Use for the pilot.
  batch-submit   write missing prompts to Batch API files and submit (50 percent price).
  batch-collect  poll submitted batches and merge finished results into the cache.

Examples
  python -m budget_guard.cache_api estimate --tier tier2 --pilot
  python -m budget_guard.cache_api sync --tier tier1 --pilot
  python -m budget_guard.cache_api batch-submit --tier tier2 --splits calib,tune,test
  python -m budget_guard.cache_api batch-collect --tier tier2

Every row lands in data/cache/<tier>.jsonl keyed by pid. Re running any command skips pids that
already have a good row and pids sitting in an uncollected batch, so nothing is paid for twice.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .common import CACHE, CONFIGS, SPLITS, JsonlAppender, done_ids, iter_jsonl, load_yaml
from .prompts import PROMPT_VERSION, messages

BATCH_DIR = CACHE / "batches"
LEDGER = CACHE / "ledger.jsonl"
MAX_BATCH_LINES = 40_000  # Batch API cap is 50k requests per file


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- selection

def select_prompts(splits: list[str], pilot_only: bool, variants: list[str], limit: int | None,
                   prompts_path: Path = SPLITS / "prompts.jsonl") -> Iterator[dict]:
    n = 0
    for p in iter_jsonl(prompts_path):
        if p["split"] not in splits or p["variant"] not in variants:
            continue
        if pilot_only and not p["pilot"]:
            continue
        yield p
        n += 1
        if limit and n >= limit:
            return


def cache_path(tier: str) -> Path:
    return CACHE / f"{tier}.jsonl"


def batch_log(tier: str) -> Path:
    return BATCH_DIR / f"{tier}_batches.jsonl"


def batch_states(tier: str) -> dict[str, dict]:
    """Append only log, last line per batch id wins."""
    state: dict[str, dict] = {}
    for rec in iter_jsonl(batch_log(tier)):
        state[rec["batch_id"]] = {**state.get(rec["batch_id"], {}), **rec}
    return state


def in_flight_pids(tier: str) -> set[str]:
    pids: set[str] = set()
    for rec in batch_states(tier).values():
        if rec.get("collected"):
            continue
        for line in iter_jsonl(Path(rec["input_file"])):
            pids.add(line["custom_id"])
    return pids


def missing(tier: str, args) -> Iterator[dict]:
    skip = done_ids(cache_path(tier)) | in_flight_pids(tier)
    for p in select_prompts(args.splits, args.pilot, args.variants, args.limit):
        if p["pid"] not in skip:
            yield p


# ---------------------------------------------------------------- cost

def est_in_tokens(text: str) -> int:
    from .prompts import SYSTEM, user_message

    # 4 chars per token plus chat formatting overhead, deliberately on the high side
    return math.ceil((len(SYSTEM) + len(user_message(text))) / 4) + 16


def price(cfg: dict, batch: bool) -> tuple[float, float]:
    d = cfg.get("batch_discount", 1.0) if batch else 1.0
    return cfg["price_in"] * d / 1e6, cfg["price_out"] * d / 1e6


def worst_case_cost(cfg: dict, prompts: list[dict], batch: bool) -> float:
    pin, pout = price(cfg, batch)
    return sum(est_in_tokens(p["text"]) * pin + cfg["max_completion_tokens"] * pout for p in prompts)


def actual_cost(cfg: dict, usage: dict, batch: bool) -> float:
    pin, pout = price(cfg, batch)
    return usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout


def spent_so_far() -> float:
    return sum(r.get("usd", 0.0) for r in iter_jsonl(LEDGER))


def guard(cost: float, cap: float, force: bool) -> None:
    print(f"worst case cost for this command: ${cost:.4f}  (cap ${cap:.2f})")
    print(f"recorded spend so far, all commands: ${spent_so_far():.4f}")
    if cost > cap and not force:
        sys.exit("over the per command cap. Narrow the selection or pass --over-cap.")


# ---------------------------------------------------------------- request and parsing

def request_body(cfg: dict, text: str) -> dict:
    body = {
        "model": cfg["model"],
        "messages": messages(text),
        "max_completion_tokens": cfg["max_completion_tokens"],
        "reasoning_effort": cfg["reasoning_effort"],
    }
    if cfg["reasoning_effort"] == "none":
        # sampling and logprob params are only accepted with effort none on GPT-6
        body.update(temperature=0, logprobs=True, top_logprobs=cfg.get("top_logprobs", 5))
    return body


def logprob_features(content: list[dict] | None) -> dict:
    """Uncertainty features from per token logprobs. Small floats only, raw tokens are not kept."""
    if not content:
        return {"lp_mean": None, "lp_min": None, "lp_sum": None, "p_first": None, "ent_first_top": None}
    lps = [t["logprob"] for t in content]
    first_top = [math.exp(t["logprob"]) for t in content[0].get("top_logprobs") or []]
    z = sum(first_top) or 1.0
    ent = -sum((p / z) * math.log(p / z) for p in first_top if p > 0)
    return {
        "lp_mean": sum(lps) / len(lps),
        "lp_min": min(lps),
        "lp_sum": sum(lps),
        "p_first": math.exp(lps[0]),
        "ent_first_top": ent,
    }


def to_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    return obj.model_dump() if hasattr(obj, "model_dump") else json.loads(obj.to_json())


def parse_completion(body: dict) -> tuple[str, dict, dict, str]:
    choice = body["choices"][0]
    answer = (choice["message"].get("content") or "").strip()
    lp = (choice.get("logprobs") or {}).get("content")
    return answer, logprob_features(lp), body.get("usage") or {}, choice.get("finish_reason", "")


def cache_row(tier: str, cfg: dict, p: dict, body: dict, batch: bool) -> dict:
    answer, feats, usage, finish = parse_completion(body)
    return {
        "pid": p["pid"], "qid": p["qid"], "variant": p["variant"], "split": p["split"],
        "tier": tier, "model": cfg["model"], "model_returned": body.get("model"),
        "prompt_version": PROMPT_VERSION, "mode": "batch" if batch else "sync",
        "answer": answer, "finish_reason": finish,
        "n_in": usage.get("prompt_tokens"), "n_out": usage.get("completion_tokens"),
        "usd": round(actual_cost(cfg, usage, batch), 8),
        **feats, "ts": now(), "error": None,
    }


# ---------------------------------------------------------------- sync mode

def call_with_retry(client, body: dict, tries: int = 6) -> dict:
    delay = 2.0
    for attempt in range(tries):
        try:
            return to_dict(client.chat.completions.create(**body))
        except Exception as e:  # rate limits, 5xx, timeouts
            if attempt == tries - 1:
                raise
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                raise  # bad request will not fix itself, do not burn retries
            time.sleep(delay)
            delay = min(delay * 2, 60)


def run_sync(tier: str, cfg: dict, args, client) -> None:
    todo = list(missing(tier, args))
    print(f"{tier}: {len(todo)} prompts to run")
    if not todo:
        return
    guard(worst_case_cost(cfg, todo, batch=False), args.cap, args.over_cap)
    if args.dry_run:
        return

    spent, ok, failed = 0.0, 0, 0
    with JsonlAppender(cache_path(tier), sync_every=args.sync_every) as out, \
            JsonlAppender(LEDGER, sync_every=1) as ledger:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(todo), 200):  # bounded number of futures in memory
                chunk = todo[start:start + 200]
                futs = {pool.submit(call_with_retry, client, request_body(cfg, p["text"])): p for p in chunk}
                chunk_usd = 0.0
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        row = cache_row(tier, cfg, p, fut.result(), batch=False)
                        ok += 1
                    except Exception as e:
                        row = {"pid": p["pid"], "qid": p["qid"], "tier": tier, "ts": now(),
                               "error": f"{type(e).__name__}: {e}"[:300]}
                        failed += 1
                    out.write(row)
                    chunk_usd += row.get("usd", 0.0)
                spent += chunk_usd
                ledger.write({"ts": now(), "tier": tier, "mode": "sync", "usd": round(chunk_usd, 8),
                              "n": len(chunk)})
                print(f"  {start + len(chunk)}/{len(todo)} done, ${spent:.4f} spent")
                if spent > args.cap:
                    print("hit the per command cap, stopping. Re run to continue.")
                    break
    print(f"{tier}: ok {ok}, failed {failed}, spent ${spent:.4f}. Failed rows retry on the next run.")


# ---------------------------------------------------------------- batch mode

def run_batch_submit(tier: str, cfg: dict, args, client) -> None:
    todo = list(missing(tier, args))
    print(f"{tier}: {len(todo)} prompts to submit")
    if not todo:
        return
    guard(worst_case_cost(cfg, todo, batch=True), args.cap, args.over_cap)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    with JsonlAppender(batch_log(tier)) as log:
        for start in range(0, len(todo), MAX_BATCH_LINES):
            chunk = todo[start:start + MAX_BATCH_LINES]
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            path = BATCH_DIR / f"{tier}_{stamp}_{start}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as f:
                for p in chunk:
                    f.write(json.dumps({"custom_id": p["pid"], "method": "POST", "url": "/v1/chat/completions",
                                        "body": request_body(cfg, p["text"])}) + "\n")
            meta = {p["pid"]: {"qid": p["qid"], "variant": p["variant"], "split": p["split"]} for p in chunk}
            (path.with_suffix(".meta.json")).write_text(json.dumps(meta))
            if args.dry_run:
                print(f"  dry run, wrote {path} with {len(chunk)} lines, not submitted")
                path.unlink()
                path.with_suffix(".meta.json").unlink()
                continue
            up = client.files.create(file=path.open("rb"), purpose="batch")
            b = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                      completion_window="24h", metadata={"project": "budget-guard", "tier": tier})
            log.write({"batch_id": b.id, "tier": tier, "input_file": str(path), "n": len(chunk),
                       "submitted": now(), "status": b.status, "collected": False})
            print(f"  submitted {b.id} with {len(chunk)} requests")


def run_batch_collect(tier: str, cfg: dict, args, client) -> None:
    states = batch_states(tier)
    open_batches = [s for s in states.values() if not s.get("collected")]
    if not open_batches:
        print(f"{tier}: nothing to collect")
        return
    with JsonlAppender(batch_log(tier)) as log, JsonlAppender(cache_path(tier), sync_every=args.sync_every) as out, \
            JsonlAppender(LEDGER) as ledger:
        for s in open_batches:
            b = client.batches.retrieve(s["batch_id"])
            print(f"  {b.id}: {b.status}  {getattr(b, 'request_counts', '')}")
            if b.status not in ("completed", "expired", "cancelled", "failed"):
                log.write({"batch_id": b.id, "status": b.status, "checked": now()})
                continue
            meta = json.loads(Path(s["input_file"]).with_suffix(".meta.json").read_text())
            ok = err = 0
            usd = 0.0
            if getattr(b, "output_file_id", None):
                for line in client.files.content(b.output_file_id).text.splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    p = {"pid": rec["custom_id"], **meta[rec["custom_id"]]}
                    resp = rec.get("response") or {}
                    if resp.get("status_code") == 200:
                        row = cache_row(tier, cfg, p, resp["body"], batch=True)
                        ok += 1
                        usd += row["usd"]
                    else:
                        row = {"pid": p["pid"], "qid": p["qid"], "tier": tier, "ts": now(),
                               "error": json.dumps(rec.get("error") or resp)[:300]}
                        err += 1
                    out.write(row)
            # requests in the error file or never run simply stay missing and get resubmitted next time
            ledger.write({"ts": now(), "tier": tier, "mode": "batch", "batch_id": b.id, "usd": round(usd, 8), "n": ok})
            log.write({"batch_id": b.id, "status": b.status, "collected": True, "collected_at": now(),
                       "ok": ok, "errors": err, "usd": round(usd, 6)})
            print(f"  merged {ok} rows, {err} errors, ${usd:.4f}")


# ---------------------------------------------------------------- cli

def make_client():
    from openai import OpenAI

    return OpenAI(max_retries=0, timeout=60)  # our own retry loop handles backoff


def main(argv=None, client=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["estimate", "sync", "batch-submit", "batch-collect"])
    ap.add_argument("--tier", required=True, choices=["tier1", "tier2"])
    ap.add_argument("--splits", default="calib,tune,test", type=lambda s: s.split(","))
    ap.add_argument("--variants", default="orig,clar", type=lambda s: s.split(","))
    ap.add_argument("--pilot", action="store_true", help="only the 300 pilot questions")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sync-every", type=int, default=20, help="fsync the cache every N rows")
    ap.add_argument("--cap", type=float, default=None, help="USD cap for this command")
    ap.add_argument("--over-cap", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    tiers = load_yaml(CONFIGS / "tiers.yaml")
    cfg = tiers[args.tier]
    args.cap = args.cap if args.cap is not None else tiers.get("max_usd_per_command", 5.0)

    if args.command == "estimate":
        todo = list(missing(args.tier, args))
        print(f"{args.tier}: {len(todo)} prompts missing")
        print(f"  sync  worst case ${worst_case_cost(cfg, todo, False):.4f}")
        print(f"  batch worst case ${worst_case_cost(cfg, todo, True):.4f}")
        print(f"  recorded spend so far ${spent_so_far():.4f}")
        return

    client = client or make_client()
    {"sync": run_sync, "batch-submit": run_batch_submit, "batch-collect": run_batch_collect}[args.command](
        args.tier, cfg, args, client)


if __name__ == "__main__":
    main()
