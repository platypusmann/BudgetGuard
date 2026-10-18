"""The optional paid ambiguity probe, tested offline with a fake client."""
import math
import os
import subprocess
import sys
from pathlib import Path

from budget_guard import ambiguity_probe as probe
from budget_guard import prompts

ROOT = Path(__file__).resolve().parents[1]


def test_p_yes_from_top_logprobs():
    top = [{"token": "Yes", "logprob": math.log(0.6)}, {"token": " no", "logprob": math.log(0.2)},
           {"token": "Maybe", "logprob": math.log(0.2)}]
    assert abs(probe.p_yes([{"token": "Yes", "logprob": math.log(0.6), "top_logprobs": top}]) - 0.75) < 1e-9
    assert probe.p_yes([{"token": "Hmm", "logprob": -0.1, "top_logprobs": [{"token": "Hmm", "logprob": -0.1}]}]) is None
    assert probe.p_yes(None) is None


def test_probe_prompt_is_separate_from_the_frozen_answer_prompt():
    assert probe.SYSTEM != prompts.SYSTEM
    body = probe.request_body({"model": "m", "reasoning_effort": "none", "top_logprobs": 5}, "who won?")
    assert body["max_completion_tokens"] == probe.MAX_OUT and body["logprobs"] and body["temperature"] == 0
    assert probe.probe_pid("q1") == "q1::ambprobe"


def test_probe_end_to_end_with_fake_client(tmp_path):
    env = {**os.environ, "BG_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT / "src")}
    env.pop("OPENAI_API_KEY", None)
    out = subprocess.run([sys.executable, str(ROOT / "tests" / "probe_smoke.py")], env=env, cwd=ROOT,
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    assert "PROBE SMOKE OK" in out.stdout
