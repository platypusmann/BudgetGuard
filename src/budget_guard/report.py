"""Figures and the README table from the CSVs evaluate.py wrote. Rerunnable on its own:

  python -m budget_guard.report --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import ROOT

# categorical slots 1 to 7 of the reference palette, in fixed order, one per routing policy.
# floors, controls and bounds are gray so the routing policies carry the color
COLORS = {
    "dp_oracle": "#2a78d6", "learned_router": "#eb6834", "slack_schedule": "#1baf7a",
    "threshold_tuned": "#eda100", "cascade_frugal": "#e87ba4", "routellm": "#008300",
    "threshold_fixed": "#6250d6", "threshold_three_tier": "#e34948",
}
GRAY = {"always_tier0": "#9a9893", "always_tier1": "#7c7a75", "always_tier2": "#5d5c58",
        "random_matched": "#b5b3ad", "hindsight_oracle": "#3f3e3b", "clarify_all": "#c9c7c1",
        "ambiguity_oracle": "#85837d"}
LABEL = {
    "dp_oracle": "DP oracle", "learned_router": "Learned router", "slack_schedule": "Slack schedule",
    "threshold_tuned": "Threshold tuned per budget", "cascade_frugal": "FrugalGPT cascade",
    "threshold_three_tier": "Three-tier threshold tuned per budget", "best_simple_tier1": "Best simple with Tier 1",
    "routellm": "RouteLLM-style", "threshold_fixed": "Fixed threshold (ReDAct)",
    "always_tier0": "Always Tier 0", "always_tier1": "Always Tier 1", "always_tier2": "Always Tier 2",
    "random_matched": "Random, matched spend", "hindsight_oracle": "Hindsight oracle",
    "clarify_all": "Clarify every question (blind)", "ambiguity_oracle": "Ambiguity oracle (clarify iff ambiguous)",
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TABLE_POLICIES = ["always_tier0", "always_tier1", "always_tier2", "random_matched", "threshold_fixed",
                  "threshold_tuned", "threshold_three_tier", "cascade_frugal", "routellm", "slack_schedule", "learned_router",
                  "dp_oracle", "clarify_all", "ambiguity_oracle", "hindsight_oracle"]


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


STAMPS = {"SYNTHETIC": "SYNTHETIC DATA, not real model results",
          "DRY RUN": "DRY RUN on tune, debugging only, not results"}


def _tag(s) -> str:
    """DRY RUN for any run evaluated on tune (real or synthetic), else SYNTHETIC or REAL."""
    if (s["eval_split"] == "tune").all():
        return "DRY RUN"
    return "SYNTHETIC" if s["data"].iloc[0] == "SYNTHETIC" else "REAL"


def _stamp(fig, tag, bottom: float = 0.0):
    """Mark synthetic and dry-run figures in the top right corner, then lay out around it."""
    if tag in STAMPS:
        fig.text(0.99, 0.995, STAMPS[tag], ha="right", va="top", fontsize=9, color="#e34948", weight="bold")
    fig.tight_layout(rect=(0, bottom, 1, 0.95))


def _policy_line(ax, d, name, x, y="accuracy", ci=True):
    d = d.sort_values(x)
    color = COLORS.get(name, GRAY.get(name, INK2))
    main = name in COLORS
    style = "--" if name in ("hindsight_oracle", "ambiguity_oracle", "clarify_all") else "-"
    ax.plot(d[x], d[y], style, color=color, linewidth=2 if main else 1.5, marker="o", markersize=5 if main else 4,
            markeredgecolor=SURFACE, markeredgewidth=1, label=LABEL.get(name, name), zorder=3 if main else 2)
    if ci and "acc_ci_lo" in d and y == "accuracy":
        ax.fill_between(d[x], d["acc_ci_lo"], d["acc_ci_hi"], color=color, alpha=0.12 if main else 0.06,
                        linewidth=0, zorder=1)


def make_figures(out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    out = Path(out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(out / "summary.csv")
    tag = _tag(s)
    head = s[s["headline"] & s["masked"]]
    soft = pd.read_csv(out / "soft_frontier.csv") if (out / "soft_frontier.csv").exists() else None
    if soft is not None and len(soft) == 0:
        soft = None
    written = []

    # 1. headline: cost versus accuracy, every policy under the hard cap, plus the uncapped soft frontier
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    for name in list(GRAY) + list(COLORS)[::-1]:
        d = head[head["policy"] == name]
        if len(d):
            _policy_line(ax, d, name, "spend_per_q")
    if soft is not None:
        sf = soft.sort_values("spend_per_q")
        ax.plot(sf["spend_per_q"], sf["accuracy"], ":", color=INK2, linewidth=1.5, marker="s", markersize=4,
                label="Soft frontier (fixed threshold, no cap)", zorder=2)
    ax.set_xscale("log")
    _style(ax, f"Cost versus accuracy under a hard cap ({head['K'].iloc[0]} questions per episode)",
           "Mean spend per question, USD (log scale)", f"Accuracy ({head['metric'].iloc[0]})")
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    _stamp(fig, tag)
    p = figs / "headline_cost_vs_accuracy.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 2. accuracy against the budget itself
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    for name in list(GRAY) + list(COLORS)[::-1]:
        d = head[head["policy"] == name]
        if len(d):
            _policy_line(ax, d, name, "budget_u2_per_q")
    ax.set_xscale("log")
    _style(ax, "Accuracy by hard budget", "Budget per question, in mean Tier 2 calls (log scale)",
           f"Accuracy ({head['metric'].iloc[0]})")
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    _stamp(fig, tag)
    p = figs / "accuracy_vs_budget.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 3. what the cap is worth: budget-blind policies run without the guard. Two panels, one measure each
    un = s[s["headline"] & ~s["masked"]]
    if len(un):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
        for name in ["threshold_fixed", "threshold_tuned", "cascade_frugal", "routellm", "always_tier2"]:
            d = un[un["policy"] == name].sort_values("budget_u2_per_q")
            if not len(d):
                continue
            color = COLORS.get(name, GRAY.get(name))
            axes[0].plot(d["budget_u2_per_q"], d["violation_rate"], "-o", color=color, linewidth=2, markersize=5,
                         markeredgecolor=SURFACE, label=LABEL[name])
            axes[1].plot(d["budget_u2_per_q"], d["spend_p95"] / d["budget_usd"], "-o", color=color, linewidth=2,
                         markersize=5, markeredgecolor=SURFACE, label=LABEL[name])
        axes[1].axhline(1.0, color=INK2, linewidth=1, linestyle="--")
        for a in axes:
            a.set_xscale("log")
        # x in axes fraction, y in data units, so the label sits on the cap line whatever the x scale
        axes[1].text(0.01, 1.02, "the cap", color=INK2, fontsize=8, va="bottom", transform=axes[1].get_yaxis_transform())
        _style(axes[0], "Episodes over the cap without the guard", "Budget per question (u2, log)", "Violation rate")
        _style(axes[1], "95th percentile spend without the guard", "Budget per question (u2, log)", "p95 spend / budget")
        axes[1].legend(fontsize=8, frameon=False, loc="upper right")
        _stamp(fig, tag)
        p = figs / "unguarded_violations.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # 4. sweeps: clarify fee and Tier 0 price, small multiples per sweep budget
    sw = s[s["masked"]]
    key = ["threshold_fixed", "threshold_tuned", "cascade_frugal", "slack_schedule", "learned_router", "dp_oracle"]
    for dim, xcol, fixed_col, fixed_val, fname, xlabel in (
            ("fee", "fee_mult", "tier0_price", "0.02/0.08", "sweep_clarify_fee.png", "Clarify fee (mean Tier 2 calls)"),
            ("price", "tier0_price", "fee_mult", 1.0, "sweep_tier0_price.png", "Tier 0 nominal price (USD per 1M in/out)")):
        d = sw[sw[fixed_col] == fixed_val]
        budgets = sorted(set(d[~d["headline"]]["budget_u2_per_q"]))
        if not budgets or d[xcol].nunique() < 2:
            continue
        fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 4), facecolor=SURFACE, squeeze=False)
        for ax, b in zip(axes[0], budgets):
            db = d[np.isclose(d["budget_u2_per_q"], b)]
            order = sorted(db[xcol].unique(), key=lambda v: float(str(v).split("/")[0]))
            for name in key:
                dp = db[db["policy"] == name].set_index(xcol).reindex(order)
                ax.plot(range(len(order)), dp["accuracy"], "-o", color=COLORS[name], linewidth=2, markersize=5,
                        markeredgecolor=SURFACE, label=LABEL[name])
            ax.set_xticks(range(len(order)), [str(o) for o in order])
            _style(ax, f"Budget {b:g} u2 per question", xlabel, "Accuracy")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
        _stamp(fig, tag, bottom=0.08)
        p = figs / fname
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # 5. decisions by slack, like Experiment 3 of the toy
    logs = sorted(out.glob("decisions_*.parquet"))
    if logs:
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
        u2 = s["u2_usd"].iloc[0]
        edges = [0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, np.inf]
        labels = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.35", "0.35-0.6", "0.6-1", ">1"]
        for path in logs:
            lg = pd.read_parquet(path)
            name = lg["policy"].iloc[0]
            K = lg["t"].max() + 1
            slack = lg["budget_before"] / ((K - lg["t"]) * u2)
            b = pd.cut(slack, edges, labels=labels, right=False)
            spend_t2 = lg["final"].isin(["T2", "C2"])
            rate = spend_t2.groupby(b, observed=False).mean()
            ax.plot(range(len(labels)), rate.reindex(labels).to_numpy(), "-o", color=COLORS.get(name, INK2),
                    linewidth=2, markersize=5, markeredgecolor=SURFACE, label=LABEL.get(name, name))
        ax.set_xticks(range(len(labels)), labels)
        _style(ax, "Tier 2 use by remaining budget per remaining question",
               "Slack, remaining budget per remaining question (u2)", "Share answered by Tier 2")
        ax.legend(fontsize=8, frameon=False)
        _stamp(fig, tag)
        p = figs / "tier2_use_by_slack.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    write_table(out)
    for col in s.columns:
        if col.startswith("accuracy_"):
            write_table(out, col[len("accuracy_"):])
    print("figures:", ", ".join(str(w) for w in written))
    return written


REF_LABEL = {"threshold_fixed": "Fixed two-tier threshold (pre-registered reference)",
             "threshold_three_tier": "Three-tier threshold tuned per budget",
             "best_simple_tier1": "Best simple with Tier 1, picked on tune"}


def _dp_versus_references(out: Path, pick, metric: str | None) -> list[str]:
    """DP minus each headline reference, paired over episodes, primary metric only."""
    import pandas as pd

    path = Path(out) / "paired.csv"
    if metric is not None or not path.exists():
        return []
    p = pd.read_csv(path)
    p = p[p["headline"] & (p["policy"] == "dp_oracle") & p["reference"].isin(REF_LABEL)]
    if p.empty:
        return []
    lines = ["", "DP minus each reference, accuracy points, paired bootstrap 95 percent CI over the same episodes.", "",
             "| DP against | " + " | ".join(f"budget {b:g} u2" for b in pick) + " |", "|---|" + "---|" * len(pick)]
    for ref, label in REF_LABEL.items():
        cells = []
        for b in pick:
            r = p[(p["reference"] == ref) & np.isclose(p["budget_u2_per_q"], b)]
            cells.append("" if not len(r) else
                         f"{r.iloc[0]['diff_mean'] * 100:+.1f} ({r.iloc[0]['diff_ci_lo'] * 100:+.1f} to "
                         f"{r.iloc[0]['diff_ci_hi'] * 100:+.1f})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def write_table(out: Path, metric: str | None = None) -> Path:
    """Markdown table of the headline config at three budgets, for the README.

    metric None is the primary metric. A secondary metric name reads its rescored columns.
    """
    import pandas as pd

    s = pd.read_csv(Path(out) / "summary.csv")
    tag = _tag(s)
    head = s[s["headline"] & s["masked"]].copy()
    label = head["metric"].iloc[0]
    if metric is not None:
        head["accuracy"] = head[f"accuracy_{metric}"]
        head["acc_ci_lo"] = head[f"acc_{metric}_ci_lo"]
        head["acc_ci_hi"] = head[f"acc_{metric}_ci_hi"]
        label = f"{metric}, secondary, same decisions rescored"
    bs = sorted(head["budget_u2_per_q"].unique())
    pick = [bs[0], bs[len(bs) // 2], bs[-1]] if len(bs) >= 3 else bs
    sha = head["design_sha256"].iloc[0] if "design_sha256" in head else "unknown"
    info_path = Path(out) / "run_info.json"
    code = json.loads(info_path.read_text()).get("code") if info_path.exists() else None
    lines = (["DRY RUN on the tune split: debugging numbers, not results. Keep out of the README.", ""]
             if tag == "DRY RUN" else [])
    commit = ""
    if code:
        tag = f" ({code['tag']})" if code.get("tag") else ""
        dirty = ", with uncommitted code changes" if code.get("dirty_code") else ""
        commit = f" Code commit {code['commit']}{tag}{dirty}."
    lines += [f"Experiment design sha256 {sha}.{commit}", "",
             f"Accuracy ({label}) with bootstrap 95 percent CI, {s['data'].iloc[0]} data, "
             f"{head['eval_split'].iloc[0]} split, {int(head['n_episodes'].iloc[0])} episodes of "
             f"{int(head['K'].iloc[0])} questions, clarify fee 1x u2, Tier 0 at 0.02/0.08. "
             "Budget is per question in u2, the mean Tier 2 call cost. Violation rate is 0 for every row.", "",
             "| Policy | " + " | ".join(f"budget {b:g} u2" for b in pick) + " |",
             "|---|" + "---|" * len(pick)]
    for name in TABLE_POLICIES:
        cells = []
        for b in pick:
            r = head[(head["policy"] == name) & np.isclose(head["budget_u2_per_q"], b)]
            if not len(r):
                cells.append("")
                continue
            r = r.iloc[0]
            cells.append(f"{r['accuracy']:.3f} ({r['acc_ci_lo']:.3f} to {r['acc_ci_hi']:.3f})")
        lines.append(f"| {LABEL.get(name, name)} | " + " | ".join(cells) + " |")
    best_path = Path(out) / "best_simple_tier1.csv"
    if best_path.exists():
        best = pd.read_csv(best_path)
        best = best[best["headline"]] if "headline" in best else best
        cells = []
        for b in pick:
            row = best[np.isclose(best["budget_u2_per_q"], b)]
            if not len(row):
                cells.append("")
                continue
            chosen = row.iloc[0]["picked"]
            r = head[(head["policy"] == chosen) & np.isclose(head["budget_u2_per_q"], b)].iloc[0]
            cells.append(f"{r['accuracy']:.3f} ({r['acc_ci_lo']:.3f} to {r['acc_ci_hi']:.3f}), {LABEL.get(chosen, chosen)}")
        lines.append("| Best simple with Tier 1, picked on tune | " + " | ".join(cells) + " |")
    assert (head["violation_rate"] == 0).all(), "a masked policy broke the cap"
    lines += _dp_versus_references(out, pick, metric)
    p = Path(out) / ("results_table.md" if metric is None else f"results_table_{metric}.md")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)
    make_figures(args.out)


if __name__ == "__main__":
    main()
