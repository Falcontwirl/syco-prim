"""Figures for WRITEUP.md: headline results under majority-of-three judge labels, plus the judge
comparison that motivated them. Reads stored judge records only (no API calls).

    python -m src.figures_writeup --config config/main_gpt4o.yaml

Writes results/figures/<experiment>/writeup/fig{1..4}_*.png. Colour roles follow analyze.py:
blue/orange always mean primed/unprimed; per-judge series are small multiples or dot plots in ink,
so no extra categorical hues are introduced.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import analyze as A
from src.config import load_config
from src.generate_filler import select_questions

NEGATIONS = {"not", "can't", "cannot", "no", "don't", "doesn't", "isn't", "won't", "wouldn't", "never"}


def is_negated(premise: str) -> bool:
    return any(w in NEGATIONS for w in premise.lower().split())


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_rates(boot: pd.DataFrame, lengths: list[int], path) -> None:
    """Sycophancy rate vs L, primed vs unprimed, majority labels (OR rule)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.arange(len(lengths))
    for cond, dx in (("unprimed", -0.05), ("primed", 0.05)):
        y = boot[f"S_{cond}"].to_numpy()
        lo, hi = boot[f"S_{cond}_lo"].to_numpy(), boot[f"S_{cond}_hi"].to_numpy()
        ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], color=A.COLOR[cond], lw=2, marker="o", ms=8, capsize=3,
                    label=cond, markeredgecolor="white", markeredgewidth=2)
    ax.set_xticks(x, A._xlabels(lengths))
    ax.set_xlabel("Intervening conversation (target tokens)", color=A.INK2)
    ax.set_ylim(0, 0.35)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="upper right")
    A._style(ax, "Sycophancy rate by context length (majority-of-three judges)", "Sycophantic trials (95% CI)")
    _save(fig, path)


def fig_benefit_by_judge(boots: dict[str, pd.DataFrame], lengths: list[int], path) -> None:
    """B(L) under each judge view, as small multiples on one shared y-scale."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(boots), figsize=(10, 3.8), sharey=True)
    x = np.arange(len(lengths))
    for ax, (title, b) in zip(axes, boots.items()):
        y, lo, hi = b["B"].to_numpy(), b["B_lo"].to_numpy(), b["B_hi"].to_numpy()
        ax.axhline(0, color=A.INK2, lw=1)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=A.INK, lw=2, marker="o", ms=8, capsize=3,
                    markeredgecolor="white", markeredgewidth=2)
        ax.set_xticks(x, A._xlabels(lengths))
        ax.set_xlabel("Intervening tokens", color=A.INK2)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f} pp")
        A._style(ax, title, "B(L) = S_unprimed − S_primed" if ax is axes[0] else "")
    fig.suptitle("Priming benefit by context length, under each judge (OR rule, 38 items, 95% CI)",
                 x=0.01, ha="left", fontsize=11, color=A.INK)
    _save(fig, path)


def fig_l0_vs_paper(rows: list[tuple[str, float, float]], path) -> None:
    """L = 0, 80 SYCON items: unprimed and primed rates for the paper, our run with its judge, and majority."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    yy = np.arange(len(rows))[::-1]
    for y, (name, u, p) in zip(yy, rows):
        ax.plot([p, u], [y, y], color=A.GRID, lw=3, zorder=1)
        ax.scatter([u], [y], s=80, color=A.COLOR["unprimed"], edgecolor="white", linewidth=2, zorder=3,
                   label="unprimed" if y == yy[0] else None)
        ax.scatter([p], [y], s=80, color=A.COLOR["primed"], edgecolor="white", linewidth=2, zorder=3,
                   label="primed" if y == yy[0] else None)
        ax.annotate(f"{u:.0%}", (u, y), xytext=(8, 0), textcoords="offset points", va="center", color=A.INK2, fontsize=9)
        ax.annotate(f"{p:.0%}", (p, y), xytext=(-8, 0), textcoords="offset points", va="center", ha="right",
                    color=A.INK2, fontsize=9)
    ax.set_yticks(yy, [r[0] for r in rows])
    ax.set_xlim(0, 0.55)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Sycophantic trials, OR rule (GPT-4o, no filler, 80 SYCON items)", color=A.INK2)
    ax.legend(frameon=False, loc="lower right")
    A._style(ax, "The paper's L = 0 effect shrinks once the judge is checked", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=A.GRID, linewidth=0.8)
    _save(fig, path)


def fig_judges(rates: pd.DataFrame, sided: dict[str, tuple[int, int, int]], path) -> None:
    """(a) OR-rule rate per judge by premise wording (dot plot); (b) whom the tie-break sided with."""
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    judges = list(rates.index)
    yy = np.arange(len(judges))[::-1]
    for col, marker in (("negated", "o"), ("affirmative", "s")):
        a1.scatter(rates[col], yy, s=70, marker=marker, color=A.INK if col == "negated" else A.NEUTRAL[1],
                   edgecolor="white", linewidth=2, zorder=3, label=f"{col} premise")
        for v, y, other in zip(rates[col], yy, rates["affirmative" if col == "negated" else "negated"]):
            right = v >= other  # label away from the row's other marker so close pairs never collide
            a1.annotate(f"{v:.0%}", (v, y), xytext=(9 if right else -9, 0), textcoords="offset points",
                        ha="left" if right else "right", va="center", color=A.INK2, fontsize=8)
    a1.set_yticks(yy, judges)
    a1.set_xlim(0, 0.55)
    a1.set_ylim(-0.6, len(judges) - 0.3)
    a1.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a1.set_xlabel("Sycophantic trials, OR rule (all 828 trials)", color=A.INK2)
    a1.legend(frameon=False, loc="lower right", fontsize=8)
    A._style(a1, "(a) Sycophancy rate by judge and premise wording", "")
    a1.grid(axis="y", visible=False)
    a1.grid(axis="x", color=A.GRID, linewidth=0.8)

    names = ["original judge\n(gpt-4o-mini)", "second judge\n(gpt-4.1-mini)", "neither"]
    outcomes = list(sided)
    w = 0.36
    x = np.arange(len(names))
    for k, (o, shade) in enumerate(zip(outcomes, (A.INK, A.NEUTRAL[1]))):
        vals = np.array(sided[o], dtype=float)
        tot = vals.sum()
        bars = a2.bar(x + (k - 0.5) * (w + 0.02), vals / tot, width=w, color=shade, label=f"{o} (n = {int(tot)})")
        for b_, v in zip(bars, vals):
            if v:
                a2.annotate(f"{int(v)}", (b_.get_x() + b_.get_width() / 2, b_.get_height()), xytext=(0, 3),
                            textcoords="offset points", ha="center", color=A.INK2, fontsize=8)
    a2.set_xticks(x, names, fontsize=8)
    a2.set_ylim(0, 1.05)
    a2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a2.set_xlabel("gpt-5.5 sided with", color=A.INK2)
    a2.legend(frameon=False, loc="upper left", fontsize=8)
    A._style(a2, "(b) gpt-5.5 tie-break on the disputed trials", "Share of disputed trials")
    _save(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not (cfg.second_judge_model and cfg.tiebreak_judge_model):
        raise SystemExit("config needs second_judge_model and tiebreak_judge_model")

    a = A.load_frame(cfg)
    b = A.load_frame(cfg, cfg.second_judge_model)
    c = A.load_frame(cfg, cfg.tiebreak_judge_model.model)
    maj, tsum = A.majority_frame(a, b, c[c["judged"]])
    lengths = cfg.context_lengths
    core_ids = set(cfg.filler_question_ids())
    core = {name: d[d["question_id"].isin(core_ids)].reset_index(drop=True)
            for name, d in (("original", a), ("second", b), ("majority", maj))}
    out = cfg.results_path / "figures" / cfg.experiment / "writeup"
    out.mkdir(parents=True, exist_ok=True)

    boot = {k: A.cluster_bootstrap(d, "or_rule", lengths) for k, d in core.items()}
    fig_rates(boot["majority"], lengths, out / "fig1_rates_majority.png")
    fig_benefit_by_judge({f"Original judge\n({cfg.judge_model.model})": boot["original"],
                          f"Second judge\n({cfg.second_judge_model})": boot["second"],
                          f"Majority of three\n(+ {cfg.tiebreak_judge_model.model} tie-break)": boot["majority"]},
                         lengths, out / "fig2_benefit_by_judge.png")

    def sycon(df: pd.DataFrame, cond: str) -> float:
        r = A.l0_upstream_comparison(df, cfg)
        return float(r[(r["source"] == "sycon") & (r["condition"] == cond)]["or_rule_ours"].iloc[0])

    up = A.l0_upstream_comparison(a, cfg)
    up = up[up["source"] == "sycon"].set_index("condition")
    fig_l0_vs_paper([("Paper's run\n(its judge)", up.loc["unprimed", "or_rule_original"], up.loc["primed", "or_rule_original"]),
                     ("Our run,\npaper's judge", sycon(a, "unprimed"), sycon(a, "primed")),
                     ("Our run,\nmajority of three", sycon(maj, "unprimed"), sycon(maj, "primed"))],
                    out / "fig3_l0_vs_paper.png")

    neg = {q.id: is_negated(q.false_premise) for q in select_questions(cfg)}
    rows = {}
    for name, d in ((cfg.judge_model.model + " (original)", a), (cfg.second_judge_model + " (second)", b),
                    ("majority of three", maj)):
        d = A.outcome_columns(d).assign(neg=lambda x: x["question_id"].map(neg))
        rows[name] = {k: g["or_rule_num"].sum() / g["or_rule_den"].sum()
                      for k, g in (("negated", d[d["neg"]]), ("affirmative", d[~d["neg"]]))}
    fig_judges(pd.DataFrame(rows).T,
               {"label": (tsum["label_sided_with_a"], tsum["label_sided_with_b"], tsum["label_sided_with_neither"]),
                "OR rule": (tsum["or_rule_sided_with_a"], tsum["or_rule_sided_with_b"], 0)},
               out / "fig4_judge_disagreement.png")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
