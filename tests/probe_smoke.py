"""Run the ambiguity probe end to end against a fake client in an isolated BG_ROOT. No network.

Invoked by tests/test_ambiguity_probe.py in a subprocess, because cache_api reads its paths
from BG_ROOT at import time.
"""
import json
import math
import os
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(os.environ["BG_ROOT"])
REPO = Path(__file__).resolve().parents[1]
shutil.copytree(REPO / "configs", ROOT / "configs", dirs_exist_ok=True)
sys.path.insert(0, str(REPO / "src"))

from budget_guard import ambiguity_probe, cache_api, common, synthetic  # noqa: E402

synthetic.build_world(ROOT / "data", n_train=600, n_val=200, seed=5, pilot_n=60)
amb = {r["qid"]: r["is_ambiguous"] for r in common.iter_jsonl(ROOT / "data" / "splits" / "pilot.jsonl")}


class Fake:
    """Says Yes with logprob tied to the true label plus noise, so the probe has real signal."""

    def __init__(self):
        self.n = 0
        self.bodies = []

    def _resp(self, body):
        assert body["max_completion_tokens"] == ambiguity_probe.MAX_OUT and body["logprobs"]
        assert body["messages"][0]["content"] == ambiguity_probe.SYSTEM
        q = body["messages"][1]["content"]
        h = (sum(map(ord, q)) % 100) / 100
        qid = next(k for k, v in QTEXT.items() if v in q)
        py = min(max(0.75 * amb[qid] + 0.25 * h, 0.02), 0.98)
        top = [{"token": "Yes", "logprob": math.log(py)}, {"token": "No", "logprob": math.log(1 - py)},
               {"token": "Maybe", "logprob": math.log(0.001)}]
        word = "Yes" if py > 0.5 else "No"
        return {"model": "gpt-6-luna", "usage": {"prompt_tokens": 80, "completion_tokens": 1},
                "choices": [{"finish_reason": "stop", "message": {"content": word},
                             "logprobs": {"content": [{"token": word, "logprob": math.log(max(py, 1 - py)),
                                                       "top_logprobs": top}]}}]}

    @property
    def chat(self):
        outer = self

        class C:
            class completions:
                @staticmethod
                def create(**body):
                    outer.n += 1
                    outer.bodies.append(body)
                    return outer._resp(body)
        return C


QTEXT = {p["qid"]: p["text"] for p in common.iter_jsonl(ROOT / "data" / "splits" / "prompts.jsonl")
         if p["pilot"] and p["variant"] == "orig"}

# sync, then resume does nothing
f = Fake()
cache_api.main(["sync", "--task", "ambiguity", "--tier", "tier1", "--pilot"], client=f)
rows = [r for r in common.iter_jsonl(cache_api.cache_path("tier1", "ambiguity"))]
assert len(rows) == len(amb) and f.n == len(amb), (len(rows), f.n)
assert all(r["pid"].endswith("::ambprobe") and r["probe_version"] == ambiguity_probe.PROBE_VERSION for r in rows)
assert all(0 <= r["p_yes"] <= 1 for r in rows)
f2 = Fake()
cache_api.main(["sync", "--task", "ambiguity", "--tier", "tier1", "--pilot"], client=f2)
assert f2.n == 0
# the answer cache was never touched by the probe
assert not cache_api.cache_path("tier1").read_text(encoding="utf-8").count("::ambprobe")

# batch path: wipe the probe cache, submit, collect
cache_api.cache_path("tier1", "ambiguity").unlink()


class FakeBatch(Fake):
    def __init__(self):
        super().__init__()
        self.store = {}

    @property
    def files(self):
        o = self

        class F:
            @staticmethod
            def create(file, purpose):
                o.store["in"] = file.read().decode()
                return types.SimpleNamespace(id="file-in")

            @staticmethod
            def content(fid):
                lines = [json.dumps({"custom_id": json.loads(l)["custom_id"],
                                     "response": {"status_code": 200, "body": o._resp(json.loads(l)["body"])}})
                         for l in o.store["in"].splitlines()]
                return types.SimpleNamespace(text="\n".join(lines))
        return F

    @property
    def batches(self):
        class B:
            @staticmethod
            def create(**kw):
                assert kw["metadata"]["task"] == "ambiguity"
                return types.SimpleNamespace(id="batch_p", status="validating")

            @staticmethod
            def retrieve(bid):
                return types.SimpleNamespace(id=bid, status="completed", output_file_id="file-out", request_counts="")
        return B


fb = FakeBatch()
cache_api.main(["batch-submit", "--task", "ambiguity", "--tier", "tier1", "--pilot"], client=fb)
assert cache_api.batch_log("tier1", "ambiguity").exists() and not cache_api.batch_log("tier1").exists()
cache_api.main(["batch-collect", "--task", "ambiguity", "--tier", "tier1", "--pilot"], client=fb)
rows = [r for r in common.iter_jsonl(cache_api.cache_path("tier1", "ambiguity")) if not r.get("error")]
assert len(rows) == len(amb) and all(r["mode"] == "batch" for r in rows)

# refusing outside the pilot or on another tier
for bad in (["estimate", "--task", "ambiguity", "--tier", "tier1"], ["estimate", "--task", "ambiguity", "--tier", "tier2", "--pilot"]):
    try:
        cache_api.main(bad)
        raise AssertionError("should have refused")
    except SystemExit:
        pass

df = ambiguity_probe.compare(ROOT / "data", ROOT / "results", backend="hash", n_boot=200)
assert set(df["score"]) == {"llm_probe", "text_detector_oof", "probe_minus_text"}
assert df.set_index("score").loc["llm_probe", "auroc"] > 0.8      # the fake probe is built to be good
print("PROBE SMOKE OK", df["decision"].iloc[0])
