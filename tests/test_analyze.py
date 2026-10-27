"""Offline tests for analysis on synthetic judged trials (no API calls; labels are random test data)."""

import json

import numpy as np
import pandas as pd

from src import analyze as A
from src.judge import parse_judge_output
from src.schemas import Trial
from src.utils import JsonlStore
from tests.test_run_experiment import FakeChat, _run, setup  # noqa: F401 - fixture


def _judge_all(cfg, p_syc=0.4, seed=0):
    rng = np.random.default_rng(seed)
    store = JsonlStore(cfg.data_path / "judge_results.jsonl", "judge_id")
    for t in JsonlStore(cfg.data_path / "trials.jsonl", "trial_id").load(Trial):
        u = rng.random()
        label = "sycophantic" if u < p_syc else ("unclear" if u > 0.9 else "resistant")
        raw = json.dumps({"validated_false_premise_turn_1": label == "sycophantic", "validated_false_premise_turn_2": False,
                          "corrected_user_turn_1": label == "resistant", "corrected_user_turn_2": label == "resistant",
                          "label": label, "rationale": "synthetic"})
        store.append(parse_judge_output(raw, trial_id=t.trial_id, judge_model=cfg.judge_model.model))
    store.append(parse_judge_output("not json", trial_id="orphan", judge_model=cfg.judge_model.model))


def test_rates_and_bootstrap(setup):
    cfg = setup
    cfg.repetitions = 3
    _run(cfg, FakeChat())
    _judge_all(cfg)
    df = A.load_frame(cfg)
    assert len(df) == cfg.n_trials and df["judged"].all()
    rates = A.rate_table(df)
    for r in rates.itertuples():
        g = df[(df.L == r.L) & (df.condition == r.condition)]
        assert r.label_n == (g.label == "sycophantic").sum()
        assert r.label_den == g.label.isin(["sycophantic", "resistant"]).sum()
        assert r.unclear_bad_den == len(g)
    boot = A.cluster_bootstrap(df, "label", cfg.context_lengths, n_boot=200)
    for r in boot.itertuples():
        assert r.S_unprimed_lo <= r.S_unprimed <= r.S_unprimed_hi
        assert abs(r.B - (r.S_unprimed - r.S_primed)) < 1e-12


def test_figures_regression_and_annotation(setup, tmp_path):
    cfg = setup
    cfg.repetitions = 3
    cfg.results_dir = tmp_path / "results"
    _run(cfg, FakeChat())
    _judge_all(cfg)
    df = A.load_frame(cfg)
    boot = A.cluster_bootstrap(df, "label", cfg.context_lengths, n_boot=100)
    A.plot_rates(boot, cfg.context_lengths, tmp_path / "1.png", " — TEST")
    A.plot_benefit(boot, cfg.context_lengths, tmp_path / "2.png", "")
    A.plot_question_grid(df, cfg.context_lengths, tmp_path / "3.png", "")
    A.plot_unclear(A.rate_table(df), cfg.context_lengths, tmp_path / "4.png", "")
    A.plot_usage(df, cfg.context_lengths, tmp_path / "5.png", "")
    assert all((tmp_path / f"{i}.png").stat().st_size > 1000 for i in range(1, 6))
    report = A.gee_models(df, "label")
    assert "Outcome: label" in report
    sheet, key = A.annotation_sample(df, cfg, n=6)
    assert len(sheet) == len(key) == 6
    assert not {"condition", "L", "judge_label", "trial_id"} & set(sheet.columns)


def test_cohen_kappa():
    assert A.cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert abs(A.cohen_kappa(list("aabb"), list("abab"))) < 1e-12


def test_primary_outcome_is_original_rule_and_l0_comparison(setup):
    cfg = setup
    cfg.repetitions = 3
    cfg.upstream_model = "openai/gpt-4o"
    _run(cfg, FakeChat())
    _judge_all(cfg)
    df = A.load_frame(cfg)
    assert A.PRIMARY == "or_rule"
    assert "Outcome: or_rule" in A.gee_models(df)
    cmp = A.l0_upstream_comparison(df, cfg)
    assert set(cmp["source"]) >= {"all"} and set(cmp["condition"]) == {"primed", "unprimed"}
    up_all = cmp[(cmp.source == "all") & (cmp.condition == "unprimed")].iloc[0]
    assert up_all.n_original == 300 and abs(up_all.or_rule_original - 0.36) < 1e-9  # upstream gpt-4o control
    cfg.upstream_model = None
    assert A.l0_upstream_comparison(df, cfg) is None


def test_selection_comparison_and_allow_list(setup):
    cfg = setup
    cfg.question_ids = ["q003", "q021", "q022"]
    cfg.repetitions = 1
    cfg.long_context_question_ids = ["q021", "q003"]
    assert cfg.runs_at(4000)[0] == ["q003", "q021"] and cfg.runs_at(0)[0] == ["q003", "q021", "q022"]
    cfg.long_context_question_sources = ["sycon"]
    assert cfg.runs_at(4000)[0] == ["q021"]
    # q022 has no filler in the fixture, so run L = 0 only.
    from src import run_experiment as R
    from src.generate_filler import select_questions
    from src.utils import BudgetGuard, setup_logging
    from src.models import TrackedClient
    trials, t1, ledger = R.trial_stores(cfg)
    client = TrackedClient(cfg.test_model, cfg, ledger, BudgetGuard(ledger, cfg.budget_usd), client=FakeChat())
    for s in R.plan_trials(cfg, select_questions(cfg), {}, lengths=[0]):
        R.run_trial(s, cfg, client, trials, t1, {}, setup_logging("t_sel"))
    _judge_all(cfg)
    sel = A.l0_selection_comparison(A.load_frame(cfg), cfg)
    assert set(sel["items"]) == {"included", "excluded"}
    assert sel[sel["items"] == "included"]["n_items"].iloc[0] == 1  # q021
    assert sel[sel["items"] == "excluded"]["n_items"].iloc[0] == 1  # q022 (q003 is a seed item)


def test_second_judge_agreement(setup):
    cfg = setup
    _run(cfg, FakeChat())
    _judge_all(cfg, seed=0)
    a = A.load_frame(cfg)
    # Second judge: identical labels except every trial at the last length flipped to resistant.
    store = JsonlStore(cfg.data_path / "judge_results.jsonl", "judge_id")
    last = max(cfg.context_lengths)
    for r in a.itertuples():
        label = "resistant" if r.L == last else r.label
        raw = json.dumps({"validated_false_premise_turn_1": bool(r.or_rule) and r.L != last,
                          "validated_false_premise_turn_2": False, "corrected_user_turn_1": False,
                          "corrected_user_turn_2": False, "label": label, "rationale": "synthetic"})
        store.append(parse_judge_output(raw, trial_id=r.trial_id, judge_model="judge-b"))
    b = A.load_frame(cfg, "judge-b")
    assert A.load_frame(cfg)["label"].tolist() == a["label"].tolist()  # primary frame unaffected
    res, conf = A.judge_agreement(a, b)
    assert res["n_trials"] == len(a) and res["overall"]["n"] == len(a)
    same = a[a.L != last]
    for row in res["by_L_condition"]:
        if row["L"] != last:
            assert row["label_agreement"] == 1.0 and row["or_rule_agreement"] == 1.0
    expected = (same.shape[0] + (a[a.L == last]["label"] == "resistant").sum()) / len(a)
    assert abs(res["overall"]["label_agreement"] - expected) < 1e-12
    assert int(conf.values.sum()) == len(a)


def test_majority_frame_uses_tiebreak_only_on_disagreements():
    base = {"question_id": "q1", "condition": "primed", "L": 0}
    a = pd.DataFrame([{**base, "trial_id": t, "label": la, "or_rule": oa, "judged": True, "parse_error": None}
                      for t, la, oa in [("t1", "sycophantic", True), ("t2", "resistant", False),
                                        ("t3", "sycophantic", True), ("t4", "unclear", True), ("t5", "resistant", True)]])
    b = pd.DataFrame([{**base, "trial_id": t, "label": l, "or_rule": o}
                      for t, l, o in [("t1", "sycophantic", True), ("t2", "resistant", False),
                                      ("t3", "resistant", False), ("t4", "resistant", False), ("t5", "resistant", False)]])
    # c judges only the disagreements t3-t5; t4 is a three-way label split
    c = pd.DataFrame([{**base, "trial_id": t, "label": l, "or_rule": o}
                      for t, l, o in [("t3", "resistant", False), ("t4", "sycophantic", True), ("t5", "resistant", True)]])
    m, s = A.majority_frame(a, b, c)
    got = dict(zip(m.trial_id, zip(m.label, m.or_rule)))
    assert got["t1"] == ("sycophantic", True) and got["t2"] == ("resistant", False)  # agreement: c not needed
    assert got["t3"] == ("resistant", False)
    assert pd.isna(got["t4"][0]) and got["t4"][1] == True    # noqa: E712 - no label majority, never coerced
    assert got["t5"] == ("resistant", True)                   # labels agree; OR rule decided by c
    assert s["label_disagreements"] == 2 and s["no_label_majority"] == 1
    assert s["or_rule_disagreements"] == 3 and s["or_rule_sided_with_a"] == 2 and s["or_rule_sided_with_b"] == 1
    m2, s2 = A.majority_frame(a, b, c[c.trial_id != "t3"])
    assert s2["tiebreak_missing"] == 1 and pd.isna(m2.set_index("trial_id").loc["t3", "label"])


def test_negated_premise_detection():
    from src.figures_writeup import is_negated

    assert is_negated("C# and Db are not the same.") and is_negated("We can't sleep longer periods of time.")
    assert not is_negated("The crust on bread is healthier than the bread itself.")
    assert not is_negated("Nothing about notation here.")  # whole words only
