"""Offline tests for the trial runner (fake test model; no API calls)."""

from pathlib import Path

import pytest

from src import original
from src import run_experiment as R
from src.config import load_config
from src.generate_filler import select_questions
from src.models import ChatResponse, OpenAICompatibleClient, TrackedClient
from src.schemas import ChatMessage, FillerConversation, FillerReview, Trial, TrialTurn1, Usage
from src.utils import BudgetGuard, JsonlStore, count_message_tokens, setup_logging, sha256_json
from src.validate_filler import review_store, usable_fillers

ROOT = Path(__file__).resolve().parents[1]


class FakeChat:
    provider = "fake"

    def __init__(self, fail_on_call=None):
        self.requests = []
        self.fail_on_call = fail_on_call

    def chat(self, messages, *, model, temperature, max_tokens, seed=None, json_mode=False, response_schema=None):
        self.requests.append(list(messages))
        if self.fail_on_call == len(self.requests):
            raise RuntimeError("injected failure")
        return ChatResponse(text=f"response {len(self.requests)}", finish_reason="stop",
                            usage=Usage(prompt_tokens=count_message_tokens(messages), completion_tokens=5))


def _filler(qid: str, n_exchanges: int = 150) -> list[dict[str, str]]:
    out = []
    for i in range(n_exchanges):
        out.append({"role": "user", "content": f"{qid} question {i} about habitats?"})
        out.append({"role": "assistant", "content": " ".join(f"{qid}w{i}x{k}" for k in range(30))})
    return out


@pytest.fixture
def setup(tmp_path):
    cfg = load_config(ROOT / "config" / "smoke.yaml")
    cfg.data_dir = tmp_path
    cfg.context_lengths = [0, 4000]
    fillers = JsonlStore(tmp_path / "generated_filler.jsonl", "filler_id")
    for qid in cfg.question_ids:
        msgs = _filler(qid)
        fillers.append(FillerConversation(
            filler_id=f"F{qid}", question_id=qid, messages=[ChatMessage(**m) for m in msgs],
            total_tokens=count_message_tokens(msgs), tokenizer="o200k_base", generator_model="fake",
            generator_params={}, generator_prompt_version="test", generation_log=[], content_sha256=sha256_json(msgs),
        ))
        review_store(cfg).append(FillerReview(review_id=f"r{qid}", filler_id=f"F{qid}", question_id=qid,
                                              decision="approved", note="test"))
    return cfg


def _run(cfg, fake):
    trials, t1_store, ledger = R.trial_stores(cfg)
    client = TrackedClient(cfg.test_model, cfg, ledger, BudgetGuard(ledger, cfg.budget_usd), client=fake)
    specs = R.plan_trials(cfg, select_questions(cfg), usable_fillers(cfg))
    done = {t.trial_id: t for t in t1_store.load(TrialTurn1)}
    log = setup_logging("test_run_experiment")
    for s in specs:
        if not trials.has(s.trial_id):
            try:
                R.run_trial(s, cfg, client, trials, t1_store, done, log)
            except RuntimeError:
                pass
    return trials.load(Trial)


def test_full_context_and_pairing(setup):
    cfg = setup
    fake = FakeChat()
    out = _run(cfg, fake)
    assert len(out) == cfg.n_trials == len(fake.requests) // 2
    n_primer = len(original.load_primer())
    by_key = {(t.question_id, t.target_tokens, t.condition): t for t in out}
    for qid in cfg.question_ids:
        p, u = by_key[(qid, 4000, "primed")], by_key[(qid, 4000, "unprimed")]
        m = [x.model_dump() for x in p.messages_turn_2]
        mu = [x.model_dump() for x in u.messages_turn_2]
        assert m[1 : 1 + n_primer] == list(original.load_primer())
        assert m[1 + n_primer : -3] == mu[1:-3]  # identical filler + question in both conditions
        assert mu[-1] == {"role": "user", "content": original.pushback_prompt()}
        assert mu[-2]["role"] == "assistant" and mu[-2]["content"] == u.response_turn_1
        assert abs(p.filler_actual_tokens - 4000) <= 200 and p.filler_n_messages % 2 == 0
        assert by_key[(qid, 0, "primed")].filler_n_messages == 0
    assert all(t.api_prompt_tokens_turn_1 == t.est_prompt_tokens_turn_1 for t in out)


def test_resume_reuses_turn1_and_is_idempotent(setup):
    cfg = setup
    first = FakeChat(fail_on_call=2)  # turn 2 of the first trial fails
    out1 = _run(cfg, first)
    assert len(out1) == cfg.n_trials - 1
    second = FakeChat()
    out2 = _run(cfg, second)
    assert len(out2) == cfg.n_trials
    assert len(second.requests) == 1  # only the missing turn 2; turn 1 came from the checkpoint
    assert second.requests[0][-1]["content"] == original.pushback_prompt()
    third = FakeChat()
    _run(cfg, third)
    assert third.requests == []


def test_missing_filler_refuses(setup):
    cfg = setup
    with pytest.raises(SystemExit, match="no usable"):
        R.plan_trials(cfg, select_questions(cfg), {})


def test_response_schema_passed_to_sdk():
    class SDK:
        def __init__(self):
            self.kwargs = None
            self.chat = self
            self.completions = self

        def create(self, **kw):
            self.kwargs = kw
            from types import SimpleNamespace as N
            return N(choices=[N(message=N(content="{}"), finish_reason="stop")],
                     usage=N(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None))

    sdk = SDK()
    c = OpenAICompatibleClient("k", sdk_client=sdk)
    c.chat([{"role": "user", "content": "x"}], model="m", temperature=0, max_tokens=5, json_mode=True,
           response_schema={"name": "s", "strict": True, "schema": {}})
    assert sdk.kwargs["response_format"]["type"] == "json_schema"
    c.chat([{"role": "user", "content": "x"}], model="m", temperature=0, max_tokens=5, json_mode=True)
    assert sdk.kwargs["response_format"] == {"type": "json_object"}


def test_parallel_run_completes_each_trial_once(setup):
    cfg = setup
    trials, t1_store, ledger = R.trial_stores(cfg)
    fake = FakeChat()
    client = TrackedClient(cfg.test_model, cfg, ledger, BudgetGuard(ledger, cfg.budget_usd), client=fake)
    specs = R.plan_trials(cfg, select_questions(cfg), usable_fillers(cfg))
    from src.utils import run_parallel
    log = setup_logging("test_parallel")
    run_parallel(lambda s: R.run_trial(s, cfg, client, trials, t1_store, {}, log), specs, 4, log)
    out = trials.load(Trial)
    assert sorted(t.trial_id for t in out) == sorted(s.trial_id for s in specs)
    assert len(fake.requests) == 2 * len(specs)


def test_tolerance_override_applies_only_to_its_length(setup):
    cfg = setup
    cfg.length_tolerance_overrides = {1000: 0.15}
    assert cfg.tolerance_for(1000) == 0.15 and cfg.tolerance_for(4000) == 0.05


def test_validation_id_depends_on_lengths(setup):
    from src.validate_filler import validation_id_for
    cfg = setup
    a = validation_id_for("F", cfg, True)
    cfg.context_lengths = [0, 1000, 4000]
    assert validation_id_for("F", cfg, True) != a


def test_budget_reserves_in_flight_calls(tmp_path):
    from src.utils import BudgetExceeded
    g = BudgetGuard(JsonlStore(tmp_path / "l.jsonl", "call_id"), 1.0)
    g.check(0.6)
    with pytest.raises(BudgetExceeded):
        g.check(0.6)  # first call still in flight
    g.record(0.1, reserved=0.6)
    g.check(0.6)


def test_l0_trials_need_no_filler_and_keep_their_ids(setup):
    """L = 0 trials planned before filler exists are the same trials as in the full plan."""
    cfg = setup
    qs = select_questions(cfg)
    l0_only = R.plan_trials(cfg, qs, {}, lengths=[0])
    full = R.plan_trials(cfg, qs, usable_fillers(cfg))
    assert {s.trial_id for s in l0_only} == {s.trial_id for s in full if s.prefix.target_tokens == 0}
    assert all(s.filler == [] and s.prefix.filler_id == "-" for s in l0_only)
    with pytest.raises(SystemExit, match="no usable"):
        R.plan_trials(cfg, qs, {}, lengths=[4000])


def test_plan_filters_and_cache_friendly_order(setup):
    cfg = setup
    cfg.repetitions = 3
    qs = select_questions(cfg)
    specs = R.plan_trials(cfg, qs, usable_fillers(cfg))
    assert len(specs) == cfg.n_trials
    keys = [(s.question.id, s.condition, s.rep, s.prefix.target_tokens) for s in specs]
    order = {q.id: i for i, q in enumerate(qs)}
    cond = {c: i for i, c in enumerate(cfg.conditions)}
    assert keys == sorted(keys, key=lambda k: (order[k[0]], cond[k[1]], k[2], k[3]))
    sub = R.plan_trials(cfg, qs, usable_fillers(cfg), lengths=[4000], reps=[0])
    assert {(s.prefix.target_tokens, s.rep) for s in sub} == {(4000, 0)}
    assert {s.trial_id for s in sub} <= {s.trial_id for s in specs}
    # seeds depend on (question, L, rep) only, so both conditions of a cell share one
    by = {}
    for s in specs:
        by.setdefault((s.question.id, s.prefix.target_tokens, s.rep), set()).add(s.seed)
    assert all(len(v) == 1 for v in by.values())


def test_long_context_subset_design(setup):
    """L = 0 runs every item with all reps; L > 0 only the chosen sources with fewer reps."""
    cfg = setup
    cfg.question_ids = ["q003", "q021"]  # q003 seed, q021 SYCON
    cfg.repetitions = 3
    cfg.long_context_question_sources = ["sycon"]
    cfg.long_context_repetitions = 1
    assert cfg.runs_at(0) == (["q003", "q021"], 3) and cfg.runs_at(4000) == (["q021"], 1)
    assert cfg.filler_question_ids() == ["q021"]
    assert cfg.n_trials == 2 * 3 * 2 + 1 * 1 * 2
    specs = R.plan_trials(cfg, select_questions(cfg), {k: v for k, v in usable_fillers(cfg).items() if k == "q021"})
    assert len(specs) == cfg.n_trials
    assert {(s.question.id, s.rep) for s in specs if s.prefix.target_tokens == 4000} == {("q021", 0)}
    full = load_config(ROOT / "config" / "smoke.yaml")
    full.data_dir, full.context_lengths, full.repetitions, full.question_ids = cfg.data_dir, cfg.context_lengths, 3, cfg.question_ids
    l0_full = {s.trial_id for s in R.plan_trials(full, select_questions(full), {}, lengths=[0])}
    assert {s.trial_id for s in specs if s.prefix.target_tokens == 0} == l0_full  # subset design keeps L = 0 IDs
