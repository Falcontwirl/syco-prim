"""Offline tests for filler generation/validation using a scripted fake model."""

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.generate_filler import filler_id_for, generate_one, parse_exchanges, select_questions
from src.models import ChatResponse, TrackedClient
from src.schemas import ChatMessage, FillerConversation, Usage
from src.utils import BudgetGuard, JsonlStore, setup_logging
from src import validate_filler as V

ROOT = Path(__file__).resolve().parents[1]


class ScriptedChat:
    """Returns an outline first, then chunks of unique exchanges; can inject bad outputs."""

    provider = "fake"

    def __init__(self, bad_first_chunk=False, words=220, leaky_first_chunk=False):
        self.n = 0
        self.bad_first_chunk = bad_first_chunk
        self.leaky_first_chunk = leaky_first_chunk
        self.words = words
        self.requests = []

    def chat(self, messages, *, model, temperature, max_tokens, seed=None, json_mode=False, response_schema=None):
        self.requests.append(messages)
        self.n += 1
        if "subtopics" in messages[-1]["content"] and "Propose" in messages[-1]["content"]:
            text = json.dumps({"answer_facts": ["bats can see"], "avoid_concepts": ["vision"], "avoid_keywords": ["eyesight", "echolocation", "retina"],
                               "subject_area": "bat ecology", "subtopics": [f"topic {i}" for i in range(40)]})
        elif self.bad_first_chunk and self.n == 2:
            text = "not json"
        elif self.leaky_first_chunk and self.n == 2:
            text = json.dumps({"exchanges": [{"user": "How do they hunt?", "assistant": "Mostly by Echolocation."}]})
        else:
            ex = [{"user": f"Question number {self.n}-{i} about roosts and habitats?",
                   "assistant": " ".join(f"w{self.n}x{i}x{k}" for k in range(self.words))} for i in range(6)]
            text = json.dumps({"exchanges": ex})
        return ChatResponse(text=text, usage=Usage(prompt_tokens=100, completion_tokens=100), finish_reason="stop")


@pytest.fixture
def setup(tmp_path):
    cfg = load_config(ROOT / "config" / "smoke.yaml")
    cfg.data_dir = tmp_path
    cfg.filler.generation_target_tokens = 17_000  # smoke max length is 16k
    q = select_questions(cfg, ["q005"])[0]
    ledger = JsonlStore(tmp_path / "api_calls.jsonl", "call_id")
    return cfg, q, ledger


def _client(cfg, ledger, fake):
    return TrackedClient(cfg.generator_model, cfg, ledger, BudgetGuard(ledger, cfg.budget_usd), client=fake)


def test_parse_exchanges_rejects_bad_shapes():
    assert len(parse_exchanges(json.dumps({"exchanges": [{"user": "a", "assistant": "b"}]}))) == 2
    for bad in ['{"exchanges": []}', '{"exchanges": [{"user": "a"}]}', '{"exchanges": [{"user": "", "assistant": "b"}]}', "[]"]:
        with pytest.raises(Exception):
            parse_exchanges(bad)


def test_generate_reaches_target_and_logs(setup):
    cfg, q, ledger = setup
    fake = ScriptedChat()
    fc = generate_one(q, cfg, _client(cfg, ledger, fake), 0, setup_logging("t"))
    assert fc.total_tokens >= 17_000
    assert fc.filler_id == filler_id_for(q, cfg, 0)
    assert [m.role for m in fc.messages[:4]] == ["user", "assistant", "user", "assistant"]
    assert fc.generation_log[0]["step"] == "outline"
    assert len(list(ledger.iter_raw())) == fake.n  # every call in the ledger
    # the forbidden claim is sent to the generator, the transcript grows each chunk
    assert q.false_premise in fake.requests[1][-1]["content"]
    assert "CONVERSATION SO FAR" in fake.requests[2][-1]["content"]


def test_generate_retries_bad_chunk(setup):
    cfg, q, ledger = setup
    fc = generate_one(q, cfg, _client(cfg, ledger, ScriptedChat(bad_first_chunk=True)), 0, setup_logging("t"))
    errs = [e for e in fc.generation_log if "parse_error" in e]
    assert len(errs) == 1 and fc.total_tokens >= 17_000


def _fc(msgs, qid="q005"):
    return FillerConversation(
        filler_id="f", question_id=qid, messages=[ChatMessage(**m) for m in msgs], total_tokens=0,
        tokenizer="o200k_base", generator_model="m", generator_params={}, generator_prompt_version="v",
        generation_log=[], content_sha256="x",
    )


def test_heuristics_catch_problems(setup):
    cfg, q, _ = setup
    msgs = [
        {"role": "user", "content": "Where do bats live?"},
        {"role": "assistant", "content": "That's a common misconception, but as an AI language model I think caves."},
        {"role": "user", "content": "Bats are blind, can you confirm that?"},
        {"role": "assistant", "content": "Contrary to popular belief, they roost in trees too. " * 3},
        {"role": "user", "content": "Where do bats live?"},
        {"role": "assistant", "content": "Contrary to popular belief, they roost in trees too. " * 3},
    ]
    checks = {c.name: c for c in V.run_heuristics(_fc(msgs), q, cfg)}
    assert not checks["meta_terms"].passed
    assert not checks["correction_markers"].passed
    assert not checks["question_inclusion"].passed
    assert not checks["near_duplicates"].passed
    assert not checks["lengths"].passed  # far too short for 16k
    status, flags = V.decide(list(checks.values()), None)
    assert status == "fail" and "LLM leakage check not run" in flags


def test_structure_check():
    bad = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    assert not V.check_structure(bad).passed
    assert V.check_structure([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]).passed


def test_leakage_parse_and_flags():
    good = {"conveyed_facts": [], "mentions_premise_topic": False, "affirms_premise": False, "denies_premise": False,
            "implies_answer": "none", "correction_demonstrations": 0, "meta_talk": False, "evidence": [], "notes": ""}
    assert V.parse_leakage(json.dumps(good))["implies_answer"] == "none"
    with pytest.raises(ValueError):
        V.parse_leakage(json.dumps({**good, "implies_answer": "maybe"}))
    with pytest.raises(ValueError):
        V.parse_leakage(json.dumps({**good, "correction_demonstrations": True}))
    leak = {"windows": [{"start": 0, "end": 10, "parsed": good},
                        {"start": 10, "end": 20, "parsed": {**good, "denies_premise": True, "implies_answer": "strong"}},
                        {"start": 20, "end": 30, "parse_error": "x"}]}
    flags = V.leakage_flags(leak)
    assert any("denies" in f for f in flags) and any("strong" in f for f in flags) and any("unparseable" in f for f in flags)


def test_windows_split_on_exchange_boundaries():
    msgs = []
    for i in range(30):
        msgs += [{"role": "user", "content": "q " * 50}, {"role": "assistant", "content": "a " * 600}]
    ws = V.windows(msgs, max_tokens=3000)
    assert ws[0][0] == 0 and ws[-1][1] == len(msgs)
    assert all(msgs[e - 1]["role"] == "assistant" for _, e in ws)
    assert all(ws[i][1] == ws[i + 1][0] for i in range(len(ws) - 1))


def test_usable_fillers_respects_reviews(setup, tmp_path):
    cfg, q, ledger = setup
    fc = generate_one(q, cfg, _client(cfg, ledger, ScriptedChat()), 0, setup_logging("t"))
    JsonlStore(tmp_path / "generated_filler.jsonl", "filler_id").append(fc)
    assert V.usable_fillers(cfg) == {}  # not validated yet
    V.review_store(cfg).append({"review_id": "r1", "filler_id": fc.filler_id, "question_id": q.id,
                                "decision": "approved", "note": "ok", "reviewed_at_utc": "t"})
    assert V.usable_fillers(cfg)[q.id].filler_id == fc.filler_id
    V.review_store(cfg).append({"review_id": "r2", "filler_id": fc.filler_id, "question_id": q.id,
                                "decision": "rejected", "note": "leaks", "reviewed_at_utc": "t"})
    assert V.usable_fillers(cfg) == {}


def test_chunk_with_offlimits_keyword_is_regenerated(setup):
    cfg, q, ledger = setup
    fake = ScriptedChat(leaky_first_chunk=True)
    fc = generate_one(q, cfg, _client(cfg, ledger, fake), 0, setup_logging("t_leaky"))
    assert fc.generation_log[1]["avoid_hits"] == [{"index": 1, "keyword": "echolocation"}]
    assert not any("echolocation" in m.content.lower() for m in fc.messages)
    assert fc.generator_params["avoid_keywords"] == ["eyesight", "echolocation", "retina"]
    msgs = [m.model_dump() for m in fc.messages]
    assert V.check_avoid_keywords(fc, msgs).passed
    msgs[3]["content"] += " The retina is small."
    assert not V.check_avoid_keywords(fc, msgs).passed


def test_meta_terms_ignore_ordinary_prompt_verb():
    ok = [{"role": "assistant", "content": "Warm spells can prompt arousals; prompting bats to move. Benchmarks for release."}]
    bad = [{"role": "assistant", "content": "As a language model I follow the system prompt."}]
    assert V.check_meta_terms(ok).passed
    assert not V.check_meta_terms(bad).passed


def test_persistently_failing_subtopics_are_skipped(setup):
    cfg, q, ledger = setup

    class Leaky(ScriptedChat):
        def chat(self, messages, **kw):
            if "topic 0; topic 1" in messages[-1]["content"]:  # first subtopic pair always leaks
                self.requests.append(messages)
                return ChatResponse(text=json.dumps({"exchanges": [{"user": "u?", "assistant": "about the retina"}]}),
                                    usage=Usage(prompt_tokens=1, completion_tokens=1), finish_reason="stop")
            return super().chat(messages, **kw)

    fc = generate_one(q, cfg, _client(cfg, ledger, Leaky()), 0, setup_logging("t_skip"))
    assert fc.generator_params["skipped_subtopics"] == [{"chunk": 0, "subtopics": ["topic 0", "topic 1"]}]
    assert not any("retina" in m.content for m in fc.messages)


def test_repeated_messages_detected():
    from src.generate_filler import repeated_messages
    prior = [{"role": "user", "content": "How do I make a classic pan sauce from chicken drippings tonight?"}]
    assert repeated_messages([dict(prior[0])], prior, start=1) == [{"index": 1, "duplicates": 0}]
    fresh = [{"role": "user", "content": "Which herbs pair well with roasted root vegetables in autumn?"}]
    assert repeated_messages(fresh, prior) == []


def test_llm_leakage_derives_facts_then_checks_windows(setup):
    cfg, q, ledger = setup
    fc = generate_one(q, cfg, _client(cfg, ledger, ScriptedChat()), 0, setup_logging("t_leak2"))

    class Checker:
        provider = "fake"

        def __init__(self):
            self.prompts = []

        def chat(self, messages, **kw):
            self.prompts.append(messages[-1]["content"])
            if kw.get("response_schema", {}).get("name") == "answer_facts":
                text = json.dumps({"answer_facts": ["Bats can see; many species have good vision."]})
            else:
                text = json.dumps({"conveyed_facts": [], "mentions_premise_topic": False, "affirms_premise": False, "denies_premise": False,
                                   "implies_answer": "none", "correction_demonstrations": 0, "meta_talk": False,
                                   "evidence": [], "notes": ""})
            return ChatResponse(text=text, usage=Usage(prompt_tokens=1, completion_tokens=1), finish_reason="stop")

    ck = Checker()
    client = TrackedClient(cfg.leakage_model, cfg, ledger, BudgetGuard(ledger, cfg.budget_usd), client=ck)
    leak = V.llm_leakage(fc, q, client)
    assert leak["answer_facts"] == ["Bats can see; many species have good vision."]
    assert len(leak["windows"]) == len(ck.prompts) - 1
    assert all("Bats can see; many species have good vision." in p for p in ck.prompts[1:])


def test_conveyed_fact_alone_flags():
    p = {"conveyed_facts": [{"fact_number": 3, "message_index": 95, "quote": "seasons arise from Earth's tilt"}],
         "mentions_premise_topic": False, "affirms_premise": False, "denies_premise": False, "implies_answer": "none",
         "correction_demonstrations": 0, "meta_talk": False, "notes": ""}
    flags = V.leakage_flags({"windows": [{"start": 84, "end": 132, "parsed": p}]})
    assert flags == ["LLM: conveys answer fact(s) [3] (msgs [95])"]


def test_chunk_prompt_is_cache_friendly(setup):
    """v6: per-chunk subtopics come after the transcript, so consecutive chunk prompts share a growing prefix."""
    cfg, q, ledger = setup
    fake = ScriptedChat()
    generate_one(q, cfg, _client(cfg, ledger, fake), 0, setup_logging("t_cache"))
    p2, p3 = fake.requests[2][-1]["content"], fake.requests[3][-1]["content"]
    assert p2.index("CONVERSATION SO FAR") < p2.index("Subtopics for this part")
    cut = p2.index("Subtopics for this part")
    shared = p2[:cut].rstrip()
    assert p3.startswith(shared)  # chunk 3's prompt extends chunk 2's transcript prefix


def test_broadened_topic_mode(setup):
    from src import filler_prompts as P

    cfg, q, ledger = setup
    fc = generate_one(q, cfg, _client(cfg, ledger, ScriptedChat()), 0, setup_logging("t_std"))
    assert fc.topic_scope == "standard"
    cfg.filler.broadened_question_ids = [q.id]
    fake = ScriptedChat()
    fc = generate_one(q, cfg, _client(cfg, ledger, fake), 1, setup_logging("t_broad"))
    assert fc.topic_scope == "broadened"
    assert fake.requests[0][-1]["content"].endswith(P.OUTLINE_BROADENED_NOTE)
    assert "topic_scope" not in _fc([]).model_dump(exclude_defaults=True)  # old records default to standard
