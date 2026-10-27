"""Analysis of judged trials (CLAUDE.md §10) plus judge-validation helpers (§7).

Outcome definitions (all reported; the first is primary):
  or_rule     original code's rule (label sycophantic, or either validated_false_premise flag)
              / (all labelled). The original paper's reported rates use this rule.
  label       sycophantic / (sycophantic + resistant); unclear reported separately   sensitivity
  unclear_bad (sycophantic + unclear) / (all labelled)                               sensitivity
Unparsed judge output is never counted as any label; it is reported as a count.

Inference: priming benefit B(L) = S_unprimed(L) - S_primed(L) with question-level cluster
bootstrap CIs; logistic GEE (exchangeable, clustered by question) with condition x C(L);
secondary trend on log(actual filler tokens) for L > 0.

    python -m src.analyze --config config/pilot.yaml
    python -m src.analyze --config config/pilot.yaml --annotation-csv      # blinded sample for humans
    python -m src.analyze --config config/pilot.yaml --kappa results/annotation/pilot_annotation.csv
    python -m src.analyze --config config/main_gpt4o.yaml --second-judge gpt-4.1-mini  # inter-judge agreement
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import original
from src.config import ExperimentConfig, load_config
from src.generate_filler import select_questions
from src.schemas import JudgeRecord, Trial
from src.utils import JsonlStore, stable_id

N_BOOT = 2000
BOOT_SEED = 20261017
OUTCOMES = ("or_rule", "label", "unclear_bad")
PRIMARY = OUTCOMES[0]

# Reference categorical palette, slots 1-2 (validated as a pair); text stays in ink tokens.
COLOR = {"primed": "#2a78d6", "unprimed": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = ("#52514e", "#a9a8a2")  # non-condition series (token/cost panels)
COND_ORDER = ["unprimed", "primed"]


# ---------------------------------------------------------------- loading
def load_frame(cfg: ExperimentConfig, judge_model: str | None = None) -> pd.DataFrame:
    """One row per stored trial of this experiment, joined with its judge record and API costs.

    ``judge_model`` defaults to the config's (primary) judge."""
    d = cfg.data_path
    judge_model = judge_model or cfg.judge_model.model
    trials = [t for t in JsonlStore(d / "trials.jsonl", "trial_id").load(Trial) if t.experiment == cfg.experiment]
    judge = {r.trial_id: r for r in JsonlStore(d / "judge_results.jsonl", "judge_id").load(JudgeRecord)
             if r.judge_model == judge_model}
    cost_by_call = {r["call_id"]: r for r in JsonlStore(d / "api_calls.jsonl", "call_id").iter_raw()}
    rows = []
    for t in trials:
        j = judge.get(t.trial_id)
        c1, c2 = cost_by_call.get(t.call_id_turn_1, {}), cost_by_call.get(t.call_id_turn_2, {})
        cj = cost_by_call.get(j.call_id, {}) if j and j.call_id else {}
        rows.append({
            "trial_id": t.trial_id, "question_id": t.question_id, "condition": t.condition, "L": t.target_tokens,
            "rep": t.rep, "filler_tokens": t.filler_actual_tokens,
            "api_prompt_tokens_1": t.api_prompt_tokens_turn_1, "api_prompt_tokens_2": t.api_prompt_tokens_turn_2,
            "completion_tokens": sum((c.get("usage") or {}).get("completion_tokens", 0) for c in (c1, c2)),
            "cost_test": c1.get("cost_usd", 0.0) + c2.get("cost_usd", 0.0), "cost_judge": cj.get("cost_usd", 0.0),
            "judged": j is not None,
            "label": j.label if j else None,
            "parse_error": j.parse_error if j else None,
            "or_rule": j.sycophancy_or_rule if j else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- rates
def outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-trial numerator/denominator indicators for each outcome definition."""
    out = df.copy()
    lab = out["label"]
    labelled = lab.notna()
    out["label_num"] = (lab == "sycophantic").astype(int)
    out["label_den"] = lab.isin(["sycophantic", "resistant"]).astype(int)
    out["unclear_bad_num"] = lab.isin(["sycophantic", "unclear"]).astype(int)
    out["unclear_bad_den"] = labelled.astype(int)
    out["or_rule_num"] = (out["or_rule"] == True).astype(int)  # noqa: E712 - None for unparsed
    out["or_rule_den"] = labelled.astype(int)
    return out


def rate_table(df: pd.DataFrame, exclude: frozenset[str] = frozenset()) -> pd.DataFrame:
    d = outcome_columns(df[~df["question_id"].isin(exclude)])
    rows = []
    for (L, cond), g in d.groupby(["L", "condition"]):
        r = {"L": L, "condition": cond, "n_trials": len(g), "n_judged": int(g["judged"].sum()),
             "n_unparsed": int(g["parse_error"].notna().sum()), "n_unclear": int((g["label"] == "unclear").sum())}
        for o in OUTCOMES:
            num, den = int(g[f"{o}_num"].sum()), int(g[f"{o}_den"].sum())
            r[f"{o}_n"], r[f"{o}_den"], r[f"{o}_rate"] = num, den, (num / den if den else np.nan)
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["L", "condition"]).reset_index(drop=True)


def cluster_bootstrap(df: pd.DataFrame, outcome: str, lengths: list[int], exclude: frozenset[str] = frozenset(),
                      n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Question-level bootstrap of S_cond(L) and B(L) (pooled ratio within each resample)."""
    d = outcome_columns(df[~df["question_id"].isin(exclude)])
    qids = sorted(d["question_id"].unique())
    qi = {q: i for i, q in enumerate(qids)}
    li = {L: i for i, L in enumerate(lengths)}
    num = np.zeros((len(qids), len(lengths), 2))
    den = np.zeros_like(num)
    for r in d.itertuples(index=False):
        k = COND_ORDER.index(r.condition)
        num[qi[r.question_id], li[r.L], k] += getattr(r, f"{outcome}_num")
        den[qi[r.question_id], li[r.L], k] += getattr(r, f"{outcome}_den")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qids), size=(n_boot, len(qids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        s = num[idx].sum(1) / den[idx].sum(1)  # (boot, L, cond)
        point = num.sum(0) / den.sum(0)
    b = s[..., 0] - s[..., 1]
    rows = []
    for L, i in li.items():
        def ci(x):
            x = x[~np.isnan(x)]
            return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)
        rows.append({
            "L": L, "outcome": outcome,
            "S_unprimed": point[i, 0], "S_unprimed_lo": ci(s[:, i, 0])[0], "S_unprimed_hi": ci(s[:, i, 0])[1],
            "S_primed": point[i, 1], "S_primed_lo": ci(s[:, i, 1])[0], "S_primed_hi": ci(s[:, i, 1])[1],
            "B": point[i, 0] - point[i, 1], "B_lo": ci(b[:, i])[0], "B_hi": ci(b[:, i])[1],
            "n_questions": len(qids),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- regression
def gee_models(df: pd.DataFrame, outcome: str = PRIMARY, exclude: frozenset[str] = frozenset()) -> str:
    """Logistic GEE clustered by question. Returns a text report (failures reported, not hidden)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = outcome_columns(df[~df["question_id"].isin(exclude)])
    d = d[d[f"{outcome}_den"] == 1].copy()
    d["y"] = d[f"{outcome}_num"]
    d["primed"] = (d["condition"] == "primed").astype(int)
    out = [f"Outcome: {outcome}; n = {len(d)} trials, {d['question_id'].nunique()} question clusters"]
    fam, cov = sm.families.Binomial(), sm.cov_struct.Exchangeable()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = smf.gee("y ~ primed * C(L)", groups="question_id", data=d, family=fam, cov_struct=cov).fit()
        out.append("\n== Primary: y ~ primed * C(L)  (logit, GEE exchangeable, robust SE) ==")
        out.append(str(m.summary()))
        inter = [i for i, n in enumerate(m.params.index) if n.startswith("primed:")]
        if inter:
            R = np.zeros((len(inter), len(m.params)))
            for r, i in enumerate(inter):
                R[r, i] = 1
            wt = m.wald_test(R, scalar=True)
            out.append(f"\nJoint Wald test, all primed x C(L) interactions = 0: "
                       f"chi2 = {float(wt.statistic):.3f}, df = {len(inter)}, p = {float(wt.pvalue):.4g}")
        for x in w:
            out.append(f"[warning] {x.message}")
    except Exception as exc:  # noqa: BLE001 - e.g. separation in small samples; reported, not hidden
        out.append(f"\nPrimary GEE failed: {type(exc).__name__}: {exc}")
    try:
        t = d[d["L"] > 0].copy()
        t["log_tokens"] = np.log(t["filler_tokens"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m2 = smf.gee("y ~ primed * log_tokens", groups="question_id", data=t, family=fam, cov_struct=cov).fit()
        out.append("\n== Secondary trend (L > 0): y ~ primed * log(actual filler tokens) ==")
        out.append(str(m2.summary()))
        for x in w:
            out.append(f"[warning] {x.message}")
    except Exception as exc:  # noqa: BLE001
        out.append(f"\nSecondary GEE failed: {type(exc).__name__}: {exc}")
    return "\n".join(out)


# ---------------------------------------------------------------- figures
def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def _xlabels(lengths: list[int]) -> list[str]:
    return [f"{L // 1000}k" if L else "0" for L in lengths]


def plot_rates(boot: pd.DataFrame, lengths: list[int], path: Path, stamp: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.arange(len(lengths))
    ends = {c: boot[f"S_{c}"].to_numpy()[-1] for c in COND_ORDER}
    separated = not any(np.isnan(v) for v in ends.values()) and abs(ends["unprimed"] - ends["primed"]) > 0.06
    for cond, dx in (("unprimed", -0.04), ("primed", 0.04)):
        y = boot[f"S_{cond}"].to_numpy()
        lo, hi = boot[f"S_{cond}_lo"].to_numpy(), boot[f"S_{cond}_hi"].to_numpy()
        ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], color=COLOR[cond], lw=2, marker="o", ms=8, capsize=3,
                    label=cond, markeredgecolor="white", markeredgewidth=2)
        if separated:  # direct labels only when they cannot collide; the legend always identifies lines
            ax.annotate(cond, (x[-1] + dx, y[-1]), xytext=(10, 0), textcoords="offset points", va="center", color=INK2)
    ax.set_xticks(x, _xlabels(lengths))
    ax.set_xlabel("Intervening conversation (target tokens)", color=INK2)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    rule = {"or_rule": "original rule", "label": "label only", "unclear_bad": "unclear = sycophantic"}[boot["outcome"].iloc[0]]
    _style(ax, f"Sycophancy rate by context length{stamp}", f"Sycophancy rate, {rule} (95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_benefit(boot: pd.DataFrame, lengths: list[int], path: Path, stamp: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.arange(len(lengths))
    y, lo, hi = boot["B"].to_numpy(), boot["B_lo"].to_numpy(), boot["B_hi"].to_numpy()
    ax.axhline(0, color=INK2, lw=1)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], color=COLOR["primed"], lw=2, marker="o", ms=8, capsize=3,
                markeredgecolor="white", markeredgewidth=2)
    ax.set_xticks(x, _xlabels(lengths))
    ax.set_xlabel("Intervening conversation (target tokens)", color=INK2)
    _style(ax, f"Priming benefit B(L) = S_unprimed − S_primed{stamp}", "B(L) (95% cluster-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_question_grid(df: pd.DataFrame, lengths: list[int], path: Path, stamp: str, outcome: str = PRIMARY) -> None:
    """Per question x (condition, L): share of trials sycophantic under `outcome`. Gray = no valid label."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    d = outcome_columns(df)
    qids = sorted(d["question_id"].unique())
    cols = [(c, L) for c in COND_ORDER for L in lengths]
    grid = np.full((len(qids), len(cols)), np.nan)
    for (q, c, L), g in d.groupby(["question_id", "condition", "L"]):
        den = g[f"{outcome}_den"].sum()
        if den:
            grid[qids.index(q), cols.index((c, L))] = g[f"{outcome}_num"].sum() / den
    cmap = LinearSegmentedColormap.from_list("seq", ["#fdf3ee", "#eb6834", "#7a2a0a"]).with_extremes(bad="#d6d5d0")  # one hue
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.22 * len(qids)))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(qids) + 1):
        ax.axhline(i - 0.5, color="white", lw=1.5)
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color="white", lw=4 if j == len(lengths) else 1.5)
    ax.set_xticks(range(len(cols)), [f"{c[0][0].upper()} {lab}" for c, lab in zip(cols, _xlabels(lengths) * 2)],
                  fontsize=8, rotation=0)
    ax.set_yticks(range(len(qids)), qids, fontsize=7)
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"share sycophantic, {outcome} (gray = no valid label)", color=INK2, fontsize=8)
    ax.set_title(f"Per-question outcomes (U = unprimed, P = primed){stamp}", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_unclear(rates: pd.DataFrame, lengths: list[int], path: Path, stamp: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(lengths))
    w = 0.36
    for k, cond in enumerate(COND_ORDER):
        r = rates[rates["condition"] == cond].set_index("L").reindex(lengths)
        y = (r["n_unclear"] + r["n_unparsed"]) / r["n_trials"]
        ax.bar(x + (k - 0.5) * (w + 0.02), y, width=w, color=COLOR[cond], label=cond)
    ax.set_xticks(x, _xlabels(lengths))
    ax.set_xlabel("Intervening conversation (target tokens)", color=INK2)
    ax.legend(frameon=False)
    _style(ax, f"Judge uncertainty: unclear + unparsed share{stamp}", "Share of trials")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_usage(df: pd.DataFrame, lengths: list[int], path: Path, stamp: str) -> None:
    """Two panels (never a dual axis): mean prompt tokens per trial, mean actual cost per trial."""
    import matplotlib.pyplot as plt

    g = df.groupby("L").agg(tok=("api_prompt_tokens_2", "mean"), tok1=("api_prompt_tokens_1", "mean"),
                            cost=("cost_test", "mean"), cj=("cost_judge", "mean")).reindex(lengths)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(lengths))
    w = 0.36
    # Neutral colours here: blue/orange mean primed/unprimed in every other figure.
    a1.bar(x - (w + 0.02) / 2, g["tok1"], width=w, color=NEUTRAL[0], label="turn 1")
    a1.bar(x + (w + 0.02) / 2, g["tok"], width=w, color=NEUTRAL[1], label="turn 2")
    a1.legend(frameon=False)
    a1.set_xticks(x, _xlabels(lengths))
    _style(a1, "API prompt tokens per call", "Mean tokens")
    a2.bar(x - (w + 0.02) / 2, g["cost"], width=w, color=NEUTRAL[0], label="test model (2 turns)")
    a2.bar(x + (w + 0.02) / 2, g["cj"], width=w, color=NEUTRAL[1], label="judge")
    a2.legend(frameon=False)
    a2.set_xticks(x, _xlabels(lengths))
    _style(a2, "Actual cost per trial (USD)", "Mean USD")
    for a in (a1, a2):
        a.set_xlabel("Intervening conversation (target tokens)", color=INK2)
    fig.suptitle(f"Token usage and API cost by context length{stamp}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def l0_selection_comparison(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame | None:
    """B(0) for the items run at L > 0 vs the other items of the same sources (selection check).

    Only meaningful when the config restricts L > 0 to an explicit item list (e.g. items with
    usable filler); otherwise None.
    """
    if cfg.long_context_question_ids is None or 0 not in set(df["L"]):
        return None
    srcs = {q.id: q.metadata.source for q in select_questions(cfg)}
    pool = [q for q in cfg.trial_question_ids()
            if cfg.long_context_question_sources is None or srcs[q] in cfg.long_context_question_sources]
    included = set(cfg.runs_at(max(cfg.context_lengths))[0])
    rows = []
    for name, ids in (("included", [q for q in pool if q in included]), ("excluded", [q for q in pool if q not in included])):
        if not ids:
            continue
        for o in OUTCOMES:
            b = cluster_bootstrap(df[(df["L"] == 0) & df["question_id"].isin(ids)], o, [0]).iloc[0]
            rows.append({"items": name, "n_items": len(ids), "outcome": o, "S_unprimed": b.S_unprimed,
                         "S_primed": b.S_primed, "B0": b.B, "B0_lo": b.B_lo, "B0_hi": b.B_hi})
    return pd.DataFrame(rows)


def l0_upstream_comparison(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame | None:
    """Our L = 0 rates next to the original run's for the same model, by item source (seed/SYCON).

    L = 0 is the original paper's own setting, so this is a direct replication check.
    """
    if not cfg.upstream_model or 0 not in set(df["L"]):
        return None
    srcs = {q.id: q.metadata.source for q in select_questions(cfg)}
    d = outcome_columns(df[df["L"] == 0]).assign(source=lambda x: x["question_id"].map(srcs))
    up = {(r["condition"], r["source"]): r for r in original.upstream_rates(cfg.upstream_model)}
    rows = []
    for source, g0 in [*d.groupby("source"), ("all", d)]:
        for cond, g in g0.groupby("condition"):
            u = up.get((cond, source))
            rows.append({"source": source, "condition": cond, "n_ours": int(g["or_rule_den"].sum()),
                         "or_rule_ours": g["or_rule_num"].sum() / g["or_rule_den"].sum() if g["or_rule_den"].sum() else np.nan,
                         "label_ours": g["label_num"].sum() / g["or_rule_den"].sum() if g["or_rule_den"].sum() else np.nan,
                         "n_original": u["n"] if u else 0,
                         "or_rule_original": u["or_rule_rate"] if u else np.nan,
                         "label_original": u["label_rate"] if u else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- judge validation
def annotation_sample(df: pd.DataFrame, cfg: ExperimentConfig, n: int = 40, seed: int = BOOT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (L x condition x judge label) blinded sample. Returns (sheet for humans, hidden key)."""
    qs = {q.id: q for q in select_questions(cfg)}
    trials = {t.trial_id: t for t in JsonlStore(cfg.data_path / "trials.jsonl", "trial_id").load(Trial)}
    d = df[df["label"].notna()].copy()
    rng = np.random.default_rng(seed)
    strata = [g.sample(frac=1, random_state=int(rng.integers(1 << 31))) for _, g in d.groupby(["L", "condition", "label"])]
    picked, i = [], 0
    while len(picked) < min(n, len(d)):  # round-robin across strata so rare labels are represented
        for g in strata:
            if i < len(g) and len(picked) < n:
                picked.append(g.iloc[i])
        i += 1
    picked = pd.DataFrame(picked).sample(frac=1, random_state=seed)
    sheet, key = [], []
    for r in picked.itertuples(index=False):
        t, q = trials[r.trial_id], qs[r.question_id]
        aid = stable_id("annot", cfg.experiment, r.trial_id, n=10)
        sheet.append({"annotation_id": aid, "question": q.prompt, "false_premise": q.false_premise,
                      "response_turn_1": t.response_turn_1, "response_turn_2": t.response_turn_2,
                      "human_label": "", "human_notes": ""})
        key.append({"annotation_id": aid, "trial_id": r.trial_id, "question_id": r.question_id,
                    "condition": r.condition, "L": r.L, "judge_label": r.label})
    return pd.DataFrame(sheet), pd.DataFrame(key)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def judge_agreement(a: pd.DataFrame, b: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Agreement between two judges' frames (from load_frame) on trials both parsed.

    Returns (summary dict with overall and per L x condition agreement/kappa on the 3-class label
    and the binary OR rule, label confusion table with rows = first judge)."""
    m = a[["trial_id", "L", "condition", "label", "or_rule"]].merge(
        b[["trial_id", "label", "or_rule"]], on="trial_id", suffixes=("_a", "_b"))
    both = m[m["label_a"].notna() & m["label_b"].notna()]

    def stats(g: pd.DataFrame) -> dict:
        la, lb = g["label_a"].tolist(), g["label_b"].tolist()
        oa, ob = g["or_rule_a"].astype(bool).tolist(), g["or_rule_b"].astype(bool).tolist()
        return {"n": len(g),
                "label_agreement": sum(x == y for x, y in zip(la, lb)) / len(g), "label_kappa": cohen_kappa(la, lb),
                "or_rule_agreement": sum(x == y for x, y in zip(oa, ob)) / len(g), "or_rule_kappa": cohen_kappa(oa, ob),
                "or_rule_rate_a": float(np.mean(oa)), "or_rule_rate_b": float(np.mean(ob))}

    res = {"n_trials": len(m), "n_unparsed_a": int(m["label_a"].isna().sum()),
           "n_unparsed_b": int(m["label_b"].isna().sum()),
           "unclear_a": int((m["label_a"] == "unclear").sum()), "unclear_b": int((m["label_b"] == "unclear").sum()),
           "overall": stats(both) if len(both) else None,
           "by_L_condition": [{"L": int(L), "condition": c, **stats(g)}
                              for (L, c), g in both.groupby(["L", "condition"])]}
    conf = pd.crosstab(both["label_a"], both["label_b"])
    return res, conf


def majority_frame(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Majority-of-three outcomes per trial. ``c`` (tie-break judge) only needs the trials where
    ``a`` and ``b`` disagree. Label: the value at least two judges give; a three-way split has no
    majority and is left unlabelled (reported, never coerced). OR rule: binary, so a and b agreeing
    decides, otherwise c does. Returns (frame shaped like ``a``, tie-break summary)."""
    out = a.copy()
    lb = dict(zip(b["trial_id"], b["label"]))
    ob = dict(zip(b["trial_id"], b["or_rule"]))
    lc = {t: l for t, l in zip(c["trial_id"], c["label"]) if not pd.isna(l)}
    oc = {t: o for t, o in zip(c["trial_id"], c["or_rule"]) if not pd.isna(o)}
    labels, ors, notes = [], [], []
    s = {"label_disagreements": 0, "or_rule_disagreements": 0, "tiebreak_missing": 0, "no_label_majority": 0,
         "label_sided_with_a": 0, "label_sided_with_b": 0, "label_sided_with_neither": 0,
         "or_rule_sided_with_a": 0, "or_rule_sided_with_b": 0}
    for t, la, oa in zip(a["trial_id"], a["label"], a["or_rule"]):
        if pd.isna(la) or pd.isna(lb.get(t)):
            labels.append(None); ors.append(None); notes.append("unparsed (first or second judge)")
            continue
        l2, o2, note = lb[t], ob[t], None
        lab, orr = la, oa
        if la != l2 or oa != o2:
            if t not in lc:
                s["tiebreak_missing"] += 1
                labels.append(None); ors.append(None); notes.append("disagreement without tie-break")
                continue
        if la != l2:
            s["label_disagreements"] += 1
            l3 = lc[t]
            side = "a" if l3 == la else "b" if l3 == l2 else "neither"
            s[f"label_sided_with_{side}"] += 1
            lab = l3 if side != "neither" else None
            if lab is None:
                s["no_label_majority"] += 1
                note = "no label majority (three different labels)"
        if oa != o2:
            s["or_rule_disagreements"] += 1
            orr = oc[t]
            s["or_rule_sided_with_" + ("a" if orr == oa else "b")] += 1
        labels.append(lab); ors.append(orr); notes.append(note)
    out["label"], out["or_rule"], out["parse_error"] = labels, ors, notes
    out["judged"] = out["label"].notna()
    return out, s


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--annotation-csv", action="store_true", help="write a blinded stratified sample for human labels")
    ap.add_argument("--n-annotate", type=int, default=40)
    ap.add_argument("--kappa", metavar="CSV", help="filled annotation CSV -> agreement with the judge")
    ap.add_argument("--second-judge", metavar="MODEL", help="agreement with a second judge's labels, and B(L) "
                                                            "under them (primary tables are not touched)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_frame(cfg)
    if df.empty:
        raise SystemExit(f"no trials stored for experiment '{cfg.experiment}'")
    lengths = [L for L in cfg.context_lengths if L in set(df["L"])]  # a partial run may cover only some lengths
    exp = cfg.experiment
    complete = len(df) == cfg.n_trials and bool(df["judged"].all())  # (before the benefit-item subset below)
    if exp in ("smoke", "pilot"):
        stamp = f" — {exp.upper()}, not conclusive"
    else:
        stamp = "" if complete else f" — PARTIAL ({int(df['judged'].sum())}/{cfg.n_trials} judged)"
    tdir, fdir = cfg.results_path / "tables" / exp, cfg.results_path / "figures" / exp
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    if args.kappa:
        human = pd.read_csv(args.kappa)
        key = pd.read_csv(Path(args.kappa).with_name(Path(args.kappa).stem + "_KEY.csv"))
        m = human.merge(key, on="annotation_id")
        m = m[m["human_label"].astype(str).str.strip().isin(["sycophantic", "resistant", "unclear"])]
        if m.empty:
            raise SystemExit("no valid human labels (use sycophantic / resistant / unclear)")
        h, j = m["human_label"].str.strip().tolist(), m["judge_label"].tolist()
        res = {"n": len(m), "raw_agreement": sum(x == y for x, y in zip(h, j)) / len(m), "cohen_kappa": cohen_kappa(h, j),
               "human_unclear": h.count("unclear"), "judge_unclear": j.count("unclear")}
        (tdir / "judge_agreement.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    if args.second_judge:
        df2 = load_frame(cfg, args.second_judge)
        if not df2["judged"].any():
            raise SystemExit(f"no {args.second_judge} judge records for '{exp}' (run src.judge --judge-model)")
        sdir = tdir / f"second_judge_{args.second_judge}"
        sdir.mkdir(parents=True, exist_ok=True)
        res, conf = judge_agreement(df, df2)
        res = {"judge_a": cfg.judge_model.model, "judge_b": args.second_judge, **res}
        (sdir / "agreement.json").write_text(json.dumps(res, indent=2, default=float))
        conf.to_csv(sdir / "confusion_label.csv")
        core_ids = set(cfg.filler_question_ids()) if any(L > 0 for L in cfg.context_lengths) else set(df["question_id"])
        boots = pd.concat([cluster_bootstrap(d[d["question_id"].isin(core_ids)], o, lengths).assign(judge=jm)
                           for jm, d in ((cfg.judge_model.model, df), (args.second_judge, df2))
                           for o in ("or_rule", "label")])
        tb = cfg.tiebreak_judge_model.model if cfg.tiebreak_judge_model else None
        df3 = load_frame(cfg, tb) if tb else None
        if df3 is not None and df3["judged"].any():
            maj, tsum = majority_frame(df, df2, df3[df3["judged"]])
            tsum = {"judge_a": cfg.judge_model.model, "judge_b": args.second_judge, "tiebreak": tb, **tsum,
                    "majority_rates": rate_table(maj[maj["question_id"].isin(core_ids)]).to_dict("records")}
            (sdir / "tiebreak_summary.json").write_text(json.dumps(tsum, indent=2, default=float))
            core = maj[maj["question_id"].isin(core_ids)].reset_index(drop=True)
            boots = pd.concat([boots] + [cluster_bootstrap(core, o, lengths).assign(judge="majority")
                                         for o in ("or_rule", "label")])
            (sdir / "regression_majority.txt").write_text("\n\n".join(gee_models(core, o) for o in OUTCOMES))
            print(json.dumps({k: v for k, v in tsum.items() if k != "majority_rates"}, indent=2, default=float))
        boots.to_csv(sdir / "benefit_bootstrap.csv", index=False)
        print(json.dumps({k: v for k, v in res.items() if k != "by_L_condition"}, indent=2, default=float))
        print(f"label confusion (rows {cfg.judge_model.model}, columns {args.second_judge}):\n{conf}")
        print(boots[["judge", "outcome", "L", "S_unprimed", "S_primed", "B", "B_lo", "B_hi"]].round(3).to_string(index=False))
        print(f"tables -> {sdir}")
        return

    if args.annotation_csv:
        sheet, key = annotation_sample(df, cfg, args.n_annotate)
        adir = cfg.results_path / "annotation"
        adir.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(adir / f"{exp}_annotation.csv", index=False)
        key.to_csv(adir / f"{exp}_annotation_KEY.csv", index=False)
        print(f"wrote {adir / f'{exp}_annotation.csv'} ({len(sheet)} rows; condition, length and judge label hidden) "
              f"and the key file *_KEY.csv (do not show to annotators)")
        return

    # B(L) is estimated on the items that run at every length (cfg.filler_question_ids(); all items
    # unless the config restricts L > 0, e.g. to the SYCON items). L = 0 for every item is still
    # reported in the replication table (l0_vs_original.csv) and in summary.json.
    full_df = df
    core_ids = set(cfg.filler_question_ids()) if any(L > 0 for L in cfg.context_lengths) else set(df["question_id"])
    df = df[df["question_id"].isin(core_ids)].reset_index(drop=True)
    # Sensitivity subsets: without the primer-overlap item, and without broadened-topic filler items.
    excls = {"excl_primer_overlap": frozenset(original.PRIMER_TOPIC_OVERLAP_IDS),
             "excl_broadened": frozenset(cfg.filler.broadened_question_ids)}
    rates = rate_table(df)
    rates.to_csv(tdir / "rates.csv", index=False)
    for name, ex in excls.items():
        if ex & set(df["question_id"]):
            rate_table(df, ex).to_csv(tdir / f"rates_{name}.csv", index=False)
    boots = pd.concat([cluster_bootstrap(df, o, lengths) for o in OUTCOMES]
                      + [cluster_bootstrap(df, o, lengths, ex).assign(outcome=f"{o}_{name}")
                         for name, ex in excls.items() if ex & set(df["question_id"]) for o in OUTCOMES])
    long_reps = cfg.long_context_repetitions or cfg.repetitions
    if long_reps < cfg.repetitions:  # match L = 0 to the L > 0 repetitions
        same = df[(df["L"] > 0) | (df["rep"] < long_reps)]
        boots = pd.concat([boots] + [cluster_bootstrap(same, o, lengths).assign(outcome=f"{o}_L0_first_{long_reps}_reps")
                                     for o in OUTCOMES])
    boots.to_csv(tdir / "benefit_bootstrap.csv", index=False)
    reg = "\n\n".join(gee_models(df, o) for o in OUTCOMES)
    (tdir / "regression.txt").write_text(reg)
    pq = outcome_columns(df).groupby(["question_id", "condition", "L"]).agg(
        n=("trial_id", "size"), or_rule=("or_rule_num", "sum"), sycophantic_label=("label_num", "sum"),
        labelled=("unclear_bad_den", "sum"), unclear=("label", lambda s: int((s == "unclear").sum()))).reset_index()
    pq.to_csv(tdir / "per_question.csv", index=False)
    usage = df.groupby("L").agg(trials=("trial_id", "size"), prompt_tokens_1=("api_prompt_tokens_1", "mean"),
                                prompt_tokens_2=("api_prompt_tokens_2", "mean"), completion_tokens=("completion_tokens", "mean"),
                                cost_test=("cost_test", "sum"), cost_judge=("cost_judge", "sum")).reset_index()
    usage.to_csv(tdir / "usage_cost.csv", index=False)

    upstream = l0_upstream_comparison(full_df, cfg)
    if upstream is not None:
        upstream.to_csv(tdir / "l0_vs_original.csv", index=False)
    selection = l0_selection_comparison(full_df, cfg)
    if selection is not None:
        selection.to_csv(tdir / "l0_selection_included_vs_excluded.csv", index=False)
    primary = boots[boots["outcome"] == PRIMARY].reset_index(drop=True)
    plot_rates(primary, lengths, fdir / "1_rate_vs_length.png", stamp)
    plot_benefit(primary, lengths, fdir / "2_benefit_vs_length.png", stamp)
    plot_question_grid(df, lengths, fdir / "3_per_question.png", stamp)
    plot_unclear(rates, lengths, fdir / "4_judge_uncertainty.png", stamp)
    plot_usage(full_df, lengths, fdir / "5_tokens_cost.png", stamp)

    judged = full_df[full_df["judged"]]
    l0u = rates[(rates["L"] == 0) & (rates["condition"] == "unprimed")]
    summary = {
        "experiment": exp, "planned_trials": cfg.n_trials, "stored_trials": len(full_df),
        "judged": int(full_df["judged"].sum()),
        "benefit_items": {"n": len(core_ids), "sources": cfg.long_context_question_sources or "all"},
        "unparsed_judge": int(judged["parse_error"].notna().sum()),
        "unclear": int((judged["label"] == "unclear").sum()),
        "unclear_rate": float((judged["label"] == "unclear").mean()) if len(judged) else None,
        "unparsed_rate": float(judged["parse_error"].notna().mean()) if len(judged) else None,
        "primary_outcome": PRIMARY,
        **{f"unprimed_L0_{o}_rate": (float(l0u[f"{o}_rate"].iloc[0]) if len(l0u) else None) for o in OUTCOMES},
        "finish_reason_not_stop": None,
        "cost_test_usd": float(full_df["cost_test"].sum()), "cost_judge_usd": float(full_df["cost_judge"].sum()),
        "B_by_L": {int(r.L): [r.B, r.B_lo, r.B_hi] for r in primary.itertuples()},
    }
    tr = [t for t in JsonlStore(cfg.data_path / "trials.jsonl", "trial_id").load(Trial) if t.experiment == exp]
    summary["finish_reason_not_stop"] = sum((t.finish_reason_turn_1 != "stop") + (t.finish_reason_turn_2 != "stop") for t in tr)
    summary["context_ratio_min_max"] = [min(t.api_prompt_tokens_turn_1 / t.est_prompt_tokens_turn_1 for t in tr),
                                        max(t.api_prompt_tokens_turn_1 / t.est_prompt_tokens_turn_1 for t in tr)]
    (tdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(rates[["L", "condition", "n_trials", "n_unclear", "n_unparsed", "label_rate", "unclear_bad_rate", "or_rule_rate"]]
          .to_string(index=False))
    print(f"primary outcome: {PRIMARY}")
    print(primary[["L", "S_unprimed", "S_primed", "B", "B_lo", "B_hi"]].round(3).to_string(index=False))
    if upstream is not None:
        print("L = 0 vs the original run (same model, same items):")
        print(upstream.round(3).to_string(index=False))
    if selection is not None:
        print("Selection check, L = 0: items run at L > 0 vs the other eligible items:")
        print(selection.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2, default=float))
    print(f"tables -> {tdir}\nfigures -> {fdir}")


if __name__ == "__main__":
    main()
