# %% [markdown]
# Budget Guard Tier 0 on Kaggle
# Settings: Accelerator GPU T4 x2 (uses one T4), Internet on, Persistence Files only.
# Inputs: attach your dataset budget-guard-prompts containing prompts.jsonl.
# Optional: to resume after a crashed commit, attach this notebook's previous output as an input too.
# Optional: add a Kaggle secret HF_TOKEN to mirror checkpoints to a private HF dataset.

# %% Cell 1. config
import os, json, math, re, string, time, glob, hashlib

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PROMPTS_GLOB = "/kaggle/input/*/prompts.jsonl"
OUT = "/kaggle/working/tier0.jsonl"
PRIOR_OUTPUT_GLOB = "/kaggle/input/*/tier0.jsonl"      # previous run output, if attached
HF_MIRROR_REPO = None          # e.g. "ranveerbhasin/budget-guard-tier0", private dataset. None disables.
PILOT_ONLY = True              # first run True, check results, then False for everything
SPLITS = {"calib", "tune", "test"}
BATCH_SIZE = 48                # prompts per greedy batch
N_SAMPLES = 5                  # sampled answers per prompt for self consistency
SAMPLE_TEMPERATURE = 0.7
MAX_NEW_TOKENS = 24
CHECKPOINT_EVERY = 240         # rows between fsync plus optional HF mirror push
TIME_LIMIT_HOURS = 11.0        # stop cleanly before Kaggle's 12h kill so the commit saves output
NUM_SHARDS, SHARD_ID = 1, 0    # set 2 and 0 or 1 to split work across two notebooks
NOMINAL_PRICE_IN, NOMINAL_PRICE_OUT = 0.02, 0.08   # USD per 1M tokens, must match configs/tiers.yaml
PROMPT_VERSION = "v1"
SYSTEM = (
    "You answer trivia questions. Reply with the short answer only, "
    "such as a name, a date, a number or a short phrase. "
    "No explanation, no full sentence, no punctuation at the end."
)
START = time.time()


# %% Cell 2. checkpointed jsonl helpers and resume
def iter_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass  # truncated last line from a crash


def shard_of(pid):
    return int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16) % NUM_SHARDS


def restore_prior():
    """Merge rows from the attached previous output and the HF mirror into OUT, deduplicated."""
    have = {r["pid"] for r in iter_jsonl(OUT)}
    sources = glob.glob(PRIOR_OUTPUT_GLOB)
    if HF_MIRROR_REPO and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import hf_hub_download
            sources.append(hf_hub_download(HF_MIRROR_REPO, "tier0.jsonl", repo_type="dataset",
                                           token=os.environ["HF_TOKEN"], local_dir="/kaggle/tmp_mirror"))
        except Exception as e:
            print("no mirror yet:", type(e).__name__)
    added = 0
    with open(OUT, "a", encoding="utf-8") as out:
        for src in sources:
            if os.path.abspath(src) == os.path.abspath(OUT):
                continue
            for r in iter_jsonl(src):
                if r.get("pid") and not r.get("error") and r["pid"] not in have:
                    out.write(json.dumps(r) + "\n")
                    have.add(r["pid"])
                    added += 1
    print(f"restored {added} rows from prior runs, {len(have)} done in total")
    return have


class Checkpointer:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.n = 0

    def write(self, row):
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.n += 1
        if self.n % CHECKPOINT_EVERY == 0:
            self.flush(push=True)

    def flush(self, push=False):
        self.f.flush()
        os.fsync(self.f.fileno())
        if push and HF_MIRROR_REPO and os.environ.get("HF_TOKEN"):
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=os.environ["HF_TOKEN"])
                api.create_repo(HF_MIRROR_REPO, repo_type="dataset", private=True, exist_ok=True)
                api.upload_file(path_or_fileobj=OUT, path_in_repo="tier0.jsonl",
                                repo_id=HF_MIRROR_REPO, repo_type="dataset")
            except Exception as e:
                print("mirror push failed, local checkpoint still fine:", type(e).__name__)

    def close(self):
        self.flush(push=True)
        self.f.close()


try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

done = restore_prior()
prompts_path = sorted(glob.glob(PROMPTS_GLOB))[0]
todo = [p for p in iter_jsonl(prompts_path)
        if p["split"] in SPLITS and (p["pilot"] or not PILOT_ONLY)
        and p["pid"] not in done and shard_of(p["pid"]) == SHARD_ID]
todo.sort(key=lambda p: len(p["text"]))   # similar lengths per batch means less padding
print(f"{len(todo)} prompts to run from {prompts_path}")


# %% Cell 3. load model, fp16 because T4 has no bf16
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map={"": 0})
model.eval()
print(torch.cuda.get_device_name(0), f"{torch.cuda.memory_allocated() / 1e9:.1f} GB used")


def render(text):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {text.strip()}\nAnswer:"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# %% Cell 4. generation with logprob features and self consistency
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def features_from_scores(seq_tokens, step_logprobs, eos_ids):
    """seq_tokens: generated ids for one row. step_logprobs: list of [vocab] log softmax tensors."""
    lps, first_top = [], None
    for t, (tid, lp) in enumerate(zip(seq_tokens, step_logprobs)):
        tid = int(tid)
        if tid in eos_ids or tid == tok.pad_token_id:
            break
        lps.append(float(lp[tid]))
        if t == 0:
            first_top = torch.topk(lp, 5).values.float().exp()
    if not lps:
        return {"lp_mean": None, "lp_min": None, "lp_sum": None, "p_first": None, "ent_first_top": None}, 0
    p = first_top / first_top.sum()
    ent = float(-(p * p.clamp_min(1e-12).log()).sum())
    return {"lp_mean": sum(lps) / len(lps), "lp_min": min(lps), "lp_sum": sum(lps),
            "p_first": math.exp(lps[0]), "ent_first_top": ent}, len(lps)


@torch.inference_mode()
def run_batch(batch):
    texts = [render(p["text"]) for p in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    n_in = enc["attention_mask"].sum(dim=1).tolist()
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

    g = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                       output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
    gen = g.sequences[:, enc["input_ids"].shape[1]:]
    logp = [torch.log_softmax(s.float(), dim=-1) for s in g.scores]     # one [B, vocab] per step
    greedy = tok.batch_decode(gen, skip_special_tokens=True)

    samples = [[] for _ in batch]
    if N_SAMPLES:
        s = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=SAMPLE_TEMPERATURE,
                           top_p=0.8, top_k=20, num_return_sequences=N_SAMPLES, pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(s[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i in range(len(batch)):
            samples[i] = [d.strip() for d in dec[i * N_SAMPLES:(i + 1) * N_SAMPLES]]

    rows = []
    for i, p in enumerate(batch):
        feats, n_out = features_from_scores(gen[i].tolist(), [lp[i] for lp in logp], eos_ids)
        ans = greedy[i].strip()
        norm = normalize(ans)
        agree = sum(normalize(x) == norm for x in samples[i]) / len(samples[i]) if samples[i] else None
        distinct = len({normalize(x) for x in samples[i]}) if samples[i] else None
        rows.append({
            "pid": p["pid"], "qid": p["qid"], "variant": p["variant"], "split": p["split"],
            "tier": "tier0", "model": MODEL_ID, "model_returned": MODEL_ID, "prompt_version": PROMPT_VERSION,
            "mode": "kaggle", "answer": ans, "finish_reason": "length" if n_out >= MAX_NEW_TOKENS else "stop",
            "n_in": int(n_in[i]), "n_out": n_out,
            "usd": round((n_in[i] * NOMINAL_PRICE_IN + n_out * NOMINAL_PRICE_OUT) / 1e6, 10),
            **feats, "sc_agree": agree, "sc_distinct": distinct, "samples": samples[i],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "error": None,
        })
    return rows


# %% Cell 5. main loop, checkpoint every CHECKPOINT_EVERY rows, OOM safe, time safe
ck = Checkpointer(OUT)
i, bs = 0, BATCH_SIZE
try:
    while i < len(todo):
        if (time.time() - START) / 3600 > TIME_LIMIT_HOURS:
            print("time limit reached, stopping cleanly. Re run to resume.")
            break
        batch = todo[i:i + bs]
        try:
            rows = run_batch(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print("OOM, batch size now", bs)
            continue
        for r in rows:
            ck.write(r)
        i += len(batch)
        if (i // len(batch)) % 10 == 0:
            rate = i / max(1e-9, time.time() - START)
            print(f"{i}/{len(todo)}  {rate:.1f} prompts/s  eta {(len(todo) - i) / max(rate, 1e-9) / 60:.0f} min")
finally:
    ck.close()

n_done = sum(1 for _ in iter_jsonl(OUT))
print(f"finished this session. {n_done} rows in {OUT}")


# %% Cell 6. quick sanity check, no grading here, grading happens locally
import collections
rows = list(iter_jsonl(OUT))[-2000:]
print("sample answers:", [r["answer"] for r in rows[:15]])
print("hit max tokens:", sum(r["finish_reason"] == "length" for r in rows), "of", len(rows))
print("mean sc_agree:", sum(r["sc_agree"] or 0 for r in rows) / max(1, len(rows)))
print(collections.Counter(r["variant"] for r in rows))
