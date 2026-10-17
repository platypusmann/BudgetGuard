import os, sys, json, random, shutil, types
os.makedirs("tests/_tmp", exist_ok=True)
ROOT = os.path.abspath("tests/_tmp/root"); shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(ROOT + "/configs"); shutil.copy("configs/tiers.yaml", ROOT + "/configs/")
os.environ["BG_ROOT"] = ROOT
sys.path.insert(0, os.path.abspath("src"))
from budget_guard import split, cache_api, common
import pyarrow as pa, pyarrow.parquet as pq

# synthetic AmbigNQ rows in the HF parquet shape, written to real parquet to test the streaming reader
def mk(i, hf):
    if i % 2:
        ann = {"type": ["multipleQAs"], "answer": [[]],
               "qaPairs": [{"question": [f"q{i} reading A?", f"q{i} reading B?"], "answer": [[f"A{i}"], [f"B{i}"]]}]}
    else:
        ann = {"type": ["singleAnswer"], "answer": [[f"ans{i}", ""]], "qaPairs": [{"question": [], "answer": []}]}
    return {"id": f"{hf}{i}", "question": f"What is thing {i}?" if i != 7 else "who is the current champion", "annotations": ann}
rows = {"train": [mk(i, "tr") for i in range(1000)] + [{"id": "bad", "question": "x?", "annotations": {"type": ["singleAnswer"], "answer": [[]], "qaPairs": [{"question": [], "answer": []}]}}],
        "validation": [mk(i, "va") for i in range(200)]}
paths = {}
for k, v in rows.items():
    paths[k] = f"tests/_tmp/{k}.parquet"; pq.write_table(pa.Table.from_pylist(v), paths[k])
src = lambda s: split.iter_parquet(paths[s], batch_size=64)
m = split.build(source=src, pilot_n=50)
print(json.dumps(m["stats"]))
split.verify()
try:
    split.build(source=src)
except SystemExit as e: print("refused overwrite OK:", str(e)[:40])
r = next(r for r in common.iter_jsonl(common.SPLITS/"calib.jsonl") if r["is_ambiguous"])
print("amb row:", r["question"], r["intended_question"], r["gold"])
print("time flag:", [x["time_sensitive"] for x in common.iter_jsonl(common.SPLITS/"calib.jsonl") if x["qid"]=="tr7"] or [x["time_sensitive"] for x in common.iter_jsonl(common.SPLITS/"tune.jsonl") if x["qid"]=="tr7"])

# fake OpenAI client
class Fake:
    def __init__(self, fail_every=0): self.n = 0; self.fail_every = fail_every
    def _resp(self, body):
        ans = "Paris"
        return {"model": body["model"] + "-2026", "choices": [{"finish_reason": "stop", "message": {"content": ans},
                "logprobs": {"content": [{"token": "Par", "logprob": -0.1, "top_logprobs": [{"token": "Par", "logprob": -0.1}, {"token": "Lon", "logprob": -2.5}]},
                                          {"token": "is", "logprob": -0.01, "top_logprobs": []}]}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 3}}
    @property
    def chat(self):
        outer = self
        class C:
            class completions:
                @staticmethod
                def create(**body):
                    outer.n += 1
                    if outer.fail_every and outer.n % outer.fail_every == 0:
                        e = Exception("boom"); e.status_code = 400; raise e
                    return outer._resp(body)
        return C

fake = Fake(fail_every=7)
cache_api.main(["estimate", "--tier", "tier2", "--pilot"])
cache_api.main(["sync", "--tier", "tier2", "--pilot", "--workers", "4"], client=fake)
n1 = len(common.done_ids(cache_api.cache_path("tier2")))
fake2 = Fake()
cache_api.main(["sync", "--tier", "tier2", "--pilot"], client=fake2)
n2 = len(common.done_ids(cache_api.cache_path("tier2")))
print("after run1 ok:", n1, "after resume ok:", n2, "calls in resume:", fake2.n)
cache_api.main(["sync", "--tier", "tier2", "--pilot"], client=Fake())  # nothing left
row = next(r for r in common.iter_jsonl(cache_api.cache_path("tier2")) if not r.get("error"))  # threads finish in any order
print("row:", {k: row[k] for k in ("pid","answer","usd","lp_mean","p_first","ent_first_top","model_returned")})

# simulate crash mid line, then append works
with open(cache_api.cache_path("tier2"), "a") as f: f.write('{"pid": "broken')
with common.JsonlAppender(cache_api.cache_path("tier2")) as w: w.write({"pid": "after_crash"})
print("survived truncated line:", "after_crash" in common.done_ids(cache_api.cache_path("tier2")))

# batch mode with fake client
class FakeBatch(Fake):
    def __init__(self): super().__init__(); self.store = {}; self.status = "in_progress"
    @property
    def files(self):
        o = self
        class F:
            @staticmethod
            def create(file, purpose): o.store["in"] = file.read().decode(); return types.SimpleNamespace(id="file-in")
            @staticmethod
            def content(fid):
                lines = []
                for i, l in enumerate(o.store["in"].splitlines()):
                    r = json.loads(l)
                    if i == 0: lines.append(json.dumps({"custom_id": r["custom_id"], "response": {"status_code": 500, "body": {}}}))
                    else: lines.append(json.dumps({"custom_id": r["custom_id"], "response": {"status_code": 200, "body": o._resp(r["body"])}}))
                return types.SimpleNamespace(text="\n".join(lines))
        return F
    @property
    def batches(self):
        o = self
        class B:
            @staticmethod
            def create(**kw): return types.SimpleNamespace(id="batch_1", status="validating")
            @staticmethod
            def retrieve(bid): return types.SimpleNamespace(id=bid, status=o.status, output_file_id="file-out", request_counts="")
        return B
fb = FakeBatch()
cache_api.main(["batch-submit", "--tier", "tier1", "--splits", "test", "--variants", "orig"], client=fb)
cache_api.main(["batch-submit", "--tier", "tier1", "--splits", "test", "--variants", "orig"], client=fb)  # in flight, must submit 0
cache_api.main(["batch-collect", "--tier", "tier1"], client=fb)
fb.status = "completed"
cache_api.main(["batch-collect", "--tier", "tier1"], client=fb)
print("tier1 cached:", len(common.done_ids(cache_api.cache_path("tier1"))))
cache_api.main(["estimate", "--tier", "tier1", "--splits", "test", "--variants", "orig"])
print("ledger total:", round(cache_api.spent_so_far(), 6))
