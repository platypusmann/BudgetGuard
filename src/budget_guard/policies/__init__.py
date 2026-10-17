"""All routing policies. build_all returns them in fitting order, dependencies first."""
from __future__ import annotations

from .base import FitContext, Policy
from .cascade import FrugalCascade
from .dp import DPOracle
from .learned import LearnedRouter, RouteLLM
from .oracle import HindsightOracle
from .simple import AlwaysTier, RandomMatched
from .threshold import FixedThreshold, SlackSchedule, TunedThreshold


def build_all(include_routellm: bool = True) -> list[Policy]:
    out: list[Policy] = [AlwaysTier(0), AlwaysTier(1), AlwaysTier(2), FixedThreshold(), TunedThreshold(),
                         RandomMatched(), FrugalCascade()]
    if include_routellm:
        out.append(RouteLLM())
    out += [DPOracle(), SlackSchedule(), LearnedRouter(), HindsightOracle()]
    return out


def fit_all(policies: list[Policy], ctx: FitContext, verbose: bool = True) -> list[Policy]:
    import time

    for p in policies:
        t = time.time()
        p.fit(ctx)
        ctx.fitted[p.name] = p
        if verbose:
            print(f"    fitted {p.name:18s} {time.time() - t:5.1f}s")
    return policies


__all__ = ["build_all", "fit_all", "FitContext", "Policy", "AlwaysTier", "RandomMatched", "FixedThreshold",
           "TunedThreshold", "SlackSchedule", "FrugalCascade", "DPOracle", "LearnedRouter", "RouteLLM",
           "HindsightOracle"]
