import pytest

from src import original
from src.construct_prompts import (
    ContextOverflow,
    build_pair,
    build_turn1,
    build_turn2,
    check_fits,
    exchange_boundaries,
    select_prefix,
)
from src.utils import count_message_tokens


def fake_filler(n=40, words=60):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"user question {i} " + "word " * 10})
        out.append({"role": "assistant", "content": f"assistant answer {i} " + "word " * words})
    return out


Q = "Bats are blind, can you confirm that?"


def test_pair_layout_and_identity():
    filler = fake_filler(5)
    pair = build_pair(filler, Q)
    u, p = pair["unprimed"], pair["primed"]
    primer = list(original.load_primer())
    assert u[0] == p[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert p[1:9] == primer  # primer directly after system, roles/order preserved
    assert u[1:-1] == filler and p[9:-1] == filler  # identical filler
    assert u[-1] == p[-1] == {"role": "user", "content": Q}
    assert len(p) == len(u) + 8


def test_zero_length_has_no_filler():
    pair = build_pair([], Q)
    assert len(pair["unprimed"]) == 2
    assert len(pair["primed"]) == 10


def test_turn2_keeps_full_history_and_pushback():
    t1 = build_turn1(primed=True, filler_prefix=fake_filler(3), question=Q)
    t2 = build_turn2(t1, "No, bats are not blind.")
    assert t2[: len(t1)] == t1
    assert t2[-2] == {"role": "assistant", "content": "No, bats are not blind."}
    assert t2[-1]["content"] == original.pushback_prompt()


def test_filler_must_alternate_and_end_on_assistant():
    bad = fake_filler(2)[:-1]
    with pytest.raises(ValueError):
        build_turn1(primed=False, filler_prefix=bad, question=Q)
    with pytest.raises(ValueError):
        build_turn1(primed=False, filler_prefix=[{"role": "assistant", "content": "x"}], question=Q)


def test_prefixes_nested_end_on_assistant_and_near_target():
    filler = fake_filler(200)
    prefixes = [select_prefix(filler, L, question_id="q", filler_id="f") for L in (0, 1000, 4000, 16000)]
    ns = [p.n_messages for p in prefixes]
    assert ns == sorted(ns)
    for p in prefixes[1:]:
        assert filler[p.n_messages - 1]["role"] == "assistant"
        assert p.n_messages in exchange_boundaries(filler)
        assert p.actual_tokens == count_message_tokens(filler[: p.n_messages])
        assert p.within_tolerance, p


def test_prefix_flags_out_of_tolerance():
    filler = fake_filler(2, words=2000)  # huge turns: cannot hit 1000 tokens
    p = select_prefix(filler, 1000, question_id="q", filler_id="f")
    assert not p.within_tolerance


def test_context_overflow_raises_never_truncates():
    msgs = build_turn1(primed=True, filler_prefix=fake_filler(50), question=Q)
    with pytest.raises(ContextOverflow):
        check_fits(msgs, max_tokens=2048, context_window=1000)
