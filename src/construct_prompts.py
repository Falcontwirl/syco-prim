"""Build trial conversations and select nested filler prefixes.

Turn 1 payload:
    primed:   [system] + primer (8 msgs) + filler_prefix + [user: question]
    unprimed: [system] +                   filler_prefix + [user: question]
Turn 2 payload: turn-1 payload + [assistant: response_1] + [user: pushback]

The filler prefix is the *same list object contents* in both conditions;
``build_pair`` asserts this. Nothing is ever truncated: oversize requests
raise ``ContextOverflow``.

CLI:  python -m src.construct_prompts --config config/smoke.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from src import original
from src.schemas import ChatMessage, FillerPrefix
from src.utils import as_dicts, count_message_tokens


class ContextOverflow(ValueError):
    pass


def _check_alternation(msgs: Sequence[dict[str, str]], first: str, last: str, what: str) -> None:
    if not msgs:
        return
    expected = first
    for i, m in enumerate(msgs):
        if m["role"] != expected:
            raise ValueError(f"{what}: message {i} has role {m['role']!r}, expected {expected!r}")
        if not m["content"].strip():
            raise ValueError(f"{what}: message {i} is empty")
        expected = "assistant" if expected == "user" else "user"
    if msgs[-1]["role"] != last:
        raise ValueError(f"{what}: must end with {last!r}, ends with {msgs[-1]['role']!r}")


def build_turn1(
    *,
    primed: bool,
    filler_prefix: Sequence[dict[str, str] | ChatMessage],
    question: str,
    system_prompt: str | None = None,
    primer: Sequence[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    system_prompt = original.neutral_system_prompt() if system_prompt is None else system_prompt
    primer = list(original.load_primer()) if primer is None else list(primer)
    filler = as_dicts(filler_prefix)
    _check_alternation(primer, "user", "assistant", "primer")
    _check_alternation(filler, "user", "assistant", "filler prefix")
    msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if primed:
        msgs.extend(dict(m) for m in primer)
    msgs.extend(dict(m) for m in filler)
    msgs.append({"role": "user", "content": question})
    return msgs


def build_turn2(turn1: Sequence[dict[str, str]], response_1: str, pushback: str | None = None) -> list[dict[str, str]]:
    pushback = original.pushback_prompt() if pushback is None else pushback
    return [*map(dict, turn1), {"role": "assistant", "content": response_1}, {"role": "user", "content": pushback}]


def build_pair(filler_prefix, question: str) -> dict[str, list[dict[str, str]]]:
    """Turn-1 payloads for both conditions, with the filler-identity invariant checked."""
    unprimed = build_turn1(primed=False, filler_prefix=filler_prefix, question=question)
    primed = build_turn1(primed=True, filler_prefix=filler_prefix, question=question)
    n_primer = len(original.load_primer())
    assert primed[0] == unprimed[0], "system prompt differs between conditions"
    assert primed[1 + n_primer :] == unprimed[1:], "filler/question differ between conditions"
    return {"unprimed": unprimed, "primed": primed}


def check_fits(messages: Sequence[dict[str, str]], max_tokens: int, context_window: int, encoding: str = "o200k_base") -> int:
    n = count_message_tokens(messages, encoding)
    if n + max_tokens > context_window:
        raise ContextOverflow(f"{n} prompt + {max_tokens} output > {context_window} context window")
    return n


def exchange_boundaries(messages: Sequence[dict[str, str] | ChatMessage]) -> list[int]:
    """Prefix lengths (message counts) that end on a complete exchange (after an assistant turn)."""
    msgs = as_dicts(messages)
    return [0] + [i + 1 for i, m in enumerate(msgs) if m["role"] == "assistant"]


def select_prefix(
    messages: Sequence[dict[str, str] | ChatMessage],
    target_tokens: int,
    *,
    question_id: str,
    filler_id: str,
    tolerance: float = 0.05,
    encoding: str = "o200k_base",
    overhead_per_message: int = 4,
) -> FillerPrefix:
    """Choose the complete-exchange prefix whose token count is closest to target.

    All prefixes come from one conversation, so lengths are nested by construction.
    """
    msgs = as_dicts(messages)
    if target_tokens == 0:
        return FillerPrefix(
            question_id=question_id, filler_id=filler_id, target_tokens=0, n_messages=0,
            actual_tokens=0, deviation_pct=None, within_tolerance=True,
        )
    # cumulative token counts at each boundary
    per_msg = [count_message_tokens([m], encoding, overhead_per_message) for m in msgs]
    cum = [0]
    for t in per_msg:
        cum.append(cum[-1] + t)
    best_n, best_tok = 0, 0
    for n in exchange_boundaries(msgs)[1:]:
        if abs(cum[n] - target_tokens) < abs(best_tok - target_tokens):
            best_n, best_tok = n, cum[n]
    dev = (best_tok - target_tokens) / target_tokens
    return FillerPrefix(
        question_id=question_id, filler_id=filler_id, target_tokens=target_tokens, n_messages=best_n,
        actual_tokens=best_tok, deviation_pct=round(100 * dev, 3), within_tolerance=abs(dev) <= tolerance,
    )


# ---------------------------------------------------------------- dry run
def _placeholder_filler(n_exchanges: int = 80) -> list[dict[str, str]]:
    """Synthetic placeholder text for structural checks only. NOT experimental data."""
    out = []
    for i in range(n_exchanges):
        out.append({"role": "user", "content": f"[PLACEHOLDER user turn {i}] " + "lorem ipsum " * 20})
        out.append({"role": "assistant", "content": f"[PLACEHOLDER assistant turn {i}] " + "dolor sit amet " * 140})
    return out


def _dry_run(config_path: str) -> None:
    from src.config import load_config

    cfg = load_config(config_path)
    q = original.build_questions()[0]
    filler = _placeholder_filler()
    print("=" * 78)
    print("DRY RUN — filler below is a synthetic PLACEHOLDER, not experimental data")
    print(f"question {q['id']}: {q['prompt']}")
    print("=" * 78)
    for L in cfg.context_lengths:
        pre = select_prefix(filler, L, question_id=q["id"], filler_id="PLACEHOLDER", tolerance=cfg.tolerance_for(L))
        pair = build_pair(filler[: pre.n_messages], q["prompt"])
        print(f"\n--- target {L} tokens → prefix {pre.n_messages} msgs, {pre.actual_tokens} tokens "
              f"(dev {pre.deviation_pct}%, within tol: {pre.within_tolerance})")
        for cond, msgs in pair.items():
            n = check_fits(msgs, cfg.test_model.max_tokens, cfg.test_model.context_window)
            layout = []
            for m in msgs:
                tag = m["content"][:28].replace("\n", " ")
                layout.append(f"{m['role'][0].upper()}:{tag!r}")
            shown = layout if len(layout) <= 14 else layout[:11] + [f"... {len(layout) - 13} more ..."] + layout[-2:]
            print(f"  [{cond:8s}] {len(msgs):3d} msgs, ~{n} prompt tokens")
            for s in shown:
                print(f"      {s}")
    t2 = build_turn2(pair["primed"], "<response 1>")
    print("\nturn-2 tail:", json.dumps(t2[-3:], ensure_ascii=False)[:300])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print payload layouts using placeholder filler")
    args = ap.parse_args()
    if not args.dry_run:
        ap.error("only --dry-run is supported here; trials are built by src.run_experiment")
    _dry_run(args.config)


if __name__ == "__main__":
    main()
