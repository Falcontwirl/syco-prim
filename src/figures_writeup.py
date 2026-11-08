"""Figures for WRITEUP.md: headline results under majority-of-three judge labels, plus the judge
comparison that motivated them. Reads stored judge records only (no API calls).

    python -m src.figures_writeup --config config/main_gpt4o.yaml

Writes results/figures/<experiment>/writeup/fig{0..5}_*.png. Colour roles follow analyze.py:
blue/orange always mean primed/unprimed; per-judge series are small multiples or dot plots in ink,
so no extra categorical hues are introduced.
"""

from __future__ import annotations

import argparse
import textwrap

import numpy as np
import pandas as pd

from src import analyze as A
from src import original
from src.config import load_config
from src.generate_filler import select_questions
from src.schemas import JudgeRecord, Trial
from src.utils import JsonlStore

NEGATIONS = {"not", "can't", "cannot", "no", "don't", "doesn't", "isn't", "won't", "wouldn't", "never"}
# q037, primed, L = 0: a short, unambiguous correction ("Actually, elevators can cause...") that the
# original judge labelled sycophantic and both other judges labelled resistant.
EXAMPLE_TRIAL = "9bdaac7fecff4036"
USER_FILL = "#f3f2ef"


def is_negated(premise: str) -> bool:
    return any(w in NEGATIONS for w in premise.lower().split())


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_rates(boot: pd.DataFrame, lengths: list[int], counts: dict[tuple[str, int], tuple[int, int]], path) -> None:
    """Sycophancy rate vs L, primed vs unprimed, majority labels (OR rule), each point labelled k/n."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.arange(len(lengths))
    for cond, dx in (("unprimed", -0.05), ("primed", 0.05)):
        y = boot[f"S_{cond}"].to_numpy()
        lo, hi = boot[f"S_{cond}_lo"].to_numpy(), boot[f"S_{cond}_hi"].to_numpy()
        ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], color=A.COLOR[cond], lw=2, marker="o", ms=8, capsize=3,
                    label=cond, markeredgecolor="white", markeredgewidth=2)
        # Unprimed labels sit left of each pair, primed right; each is nudged away from the line segment
        # on its side so the text never sits on the line.
        left = cond == "unprimed"
        for i, (xi, yi, L) in enumerate(zip(x + dx, y, lengths)):
            k, n = counts[(cond, L)]
            j = i - 1 if left else i + 1
            dy = 0 if not 0 <= j < len(y) else (-7 if y[j] > yi else 7)
            ax.annotate(f"{k}/{n}", (xi, yi), xytext=(-9 if left else 9, dy), textcoords="offset points",
                        ha="right" if left else "left", va="center", color=A.INK2, fontsize=8)
    ax.set_xticks(x, A._xlabels(lengths))
    ax.set_xlabel("Intervening conversation (target tokens)", color=A.INK2)
    ax.set_ylim(0, 0.35)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="upper right")
    A._style(ax, "Sycophancy rate by context length (majority-of-three judges)",
             "Sycophantic trials (95% CI; labels: k/n)")
    _save(fig, path)


def fig_design(path) -> None:
    """Schematic of one trial in each arm. Every block except the primer is identical across arms."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    steps = [("System prompt", 1.6, "user"), ("Primer\n(8 messages)", 1.6, "primer"),
             ("Filler conversation\n0 / 1k / 4k / 16k tokens", 2.9, "user"), ("False claim", 1.4, "user"),
             ("Reply 1", 1.1, "model"), ("Pushback", 1.3, "user"), ("Reply 2", 1.1, "model")]
    gap, h, x0 = 0.15, 0.8, 1.9
    fig, ax = plt.subplots(figsize=(11, 3.3))

    def box(x, y, w, text, face, edge, ls="-", color=A.INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=face, ec=edge,
                                    lw=1.2, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=color)

    for cond, y in (("primed", 2.0), ("unprimed", 0.6)):
        ax.scatter([0.2], [y + h / 2], s=70, color=A.COLOR[cond], edgecolor="white", linewidth=2, zorder=3)
        ax.text(0.38, y + h / 2, cond, ha="left", va="center", fontsize=10, color=A.INK, fontweight="bold")
        x = x0
        for text, w, kind in steps:
            if kind == "primer" and cond == "unprimed":
                box(x, y, w, "(no primer)", "white", A.NEUTRAL[1], ls="--", color=A.INK2)
            elif kind == "primer":
                box(x, y, w, text, "#dde9f8", A.COLOR["primed"])
            elif kind == "model":
                box(x, y, w, text, "white", A.INK2)
            else:
                box(x, y, w, text, USER_FILL, A.GRID)
            if text.startswith("Filler") and cond == "primed":
                ax.annotate("", (x + w / 2, 1.42), (x + w / 2, 1.98),
                            arrowprops=dict(arrowstyle="<->", color=A.INK2, lw=1))
                ax.text(x + w / 2 + 0.08, 1.7, "same filler in both arms", fontsize=8, color=A.INK2, va="center")
            x += w + gap
        ax.annotate("", (x + 0.25, y + h / 2), (x - gap + 0.02, y + h / 2),
                    arrowprops=dict(arrowstyle="->", color=A.INK2, lw=1))
    jx = x + 0.3
    ax.add_patch(FancyBboxPatch((jx, 0.6), 2.3, 2.2, boxstyle="round,pad=0,rounding_size=0.08", fc="white",
                                ec=A.INK2, lw=1.2))
    ax.text(jx + 1.15, 1.7, "Judge\n(GPT-4o-mini,\npaper's prompt)\nreads replies 1 and 2\n\nsycophantic /\nresistant / unclear",
            ha="center", va="center", fontsize=9, color=A.INK)
    ax.text(x0, 0.2, "Grey: fixed text from the paper or our filler.   White: generated by GPT-4o (the model under test).",
            fontsize=8, color=A.INK2, va="center")
    ax.text(x0, -0.15, "Priming benefit at each filler length:  B(L) = S_unprimed(L) − S_primed(L)",
            fontsize=9, color=A.INK, va="center")
    ax.set_xlim(0, jx + 2.4)
    ax.set_ylim(-0.4, 3.2)
    ax.axis("off")
    ax.set_title("One trial, in each arm", loc="left", fontsize=11, color=A.INK)
    _save(fig, path)


def fig_example(trial: Trial, prompt: str, false_premise: str, verdicts: list[tuple[str, str, str]], path) -> None:
    """One trial as a chat transcript, with each judge's label and rationale underneath."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    W, lh, pad = 8.0, 0.2, 0.12  # inches; lh = line height at 10 pt
    turns = [("User", prompt, "user"), ("GPT-4o, reply 1", trial.response_turn_1, "model"),
             ("User (pushback)", original.pushback_prompt(), "user"),
             ("GPT-4o, reply 2", trial.response_turn_2, "model")]
    wrapped = [(who, textwrap.wrap(" ".join(text.split()), 78), kind) for who, text, kind in turns]
    rows = [(j, v, textwrap.wrap(r, 58)) for j, v, r in verdicts]
    H = (0.95 + sum(0.25 + len(ls) * lh + 2 * pad + 0.12 for _, ls, _ in wrapped)
         + 0.55 + sum(max(1, len(r)) * 0.18 + 0.14 for _, _, r in rows) + 0.2)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    y = 0.3
    ax.text(0.4, y, "The paper's judge scores a correction as agreement", fontsize=12, color=A.INK, va="top")
    y += 0.32
    ax.text(0.4, y, f"Item {trial.question_id} · {trial.condition} · no filler · false claim: “{false_premise}”",
            fontsize=8.5, color=A.INK2, va="top")
    y += 0.33
    for who, lines, kind in wrapped:
        x = 0.4 if kind == "user" else 1.4
        ax.text(x, y, who, fontsize=8.5, color=A.INK2, va="top")
        y += 0.25
        bh = len(lines) * lh + 2 * pad
        ax.add_patch(FancyBboxPatch((x, y), W - 1.8, bh, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=USER_FILL if kind == "user" else "white",
                                    ec=A.GRID if kind == "user" else A.INK2, lw=1))
        ax.text(x + 0.15, y + pad, "\n".join(lines), fontsize=10, color=A.INK, va="top", linespacing=1.4)
        y += bh + 0.12
    y += 0.15
    ax.text(0.4, y, "Judge", fontsize=8.5, color=A.INK2, va="top")
    ax.text(2.4, y, "Label", fontsize=8.5, color=A.INK2, va="top")
    ax.text(3.6, y, "Rationale (verbatim)", fontsize=8.5, color=A.INK2, va="top")
    y += 0.25
    ax.plot([0.4, W - 0.4], [y, y], color=A.GRID, lw=1)
    y += 0.1
    for judge, label, r in rows:
        ax.text(0.4, y, judge, fontsize=9, color=A.INK, va="top")
        ax.text(2.4, y, label, fontsize=9, color=A.INK, va="top", fontweight="bold")
        ax.text(3.6, y, "\n".join(r), fontsize=9, color=A.INK2, va="top", linespacing=1.3)
        y += max(1, len(r)) * 0.18 + 0.14
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
    A._style(ax, "The paper's L = 0 effect shrinks once responses are re-judged", "")
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
    tally = A.outcome_columns(core["majority"]).groupby(["condition", "L"])[["or_rule_num", "or_rule_den"]].sum()
    counts = {key: (int(r.or_rule_num), int(r.or_rule_den)) for key, r in tally.iterrows()}
    fig_design(out / "fig0_design.png")
    fig_rates(boot["majority"], lengths, counts, out / "fig1_rates_majority.png")
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

    trial = next(t for t in JsonlStore(cfg.data_path / "trials.jsonl", "trial_id").load(Trial)
                 if t.trial_id == EXAMPLE_TRIAL)
    q = next(q for q in select_questions(cfg) if q.id == trial.question_id)
    judged = {r.judge_model: r.parsed for r in JsonlStore(cfg.data_path / "judge_results.jsonl", "judge_id").load(JudgeRecord)
              if r.trial_id == EXAMPLE_TRIAL and r.parsed}
    names = ((cfg.judge_model.model, "(paper's judge)"), (cfg.second_judge_model, "(second)"),
             (cfg.tiebreak_judge_model.model, "(tie-break)"))
    fig_example(trial, q.prompt, q.false_premise,
                [(f"{m} {role}", judged[m].label, judged[m].rationale) for m, role in names],
                out / "fig5_example_q037.png")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
