"""Generate one long (~32k-token) topic-related filler conversation per question.

Generated once, before any trials. Shorter context lengths are prefixes of this
conversation (construct_prompts.select_prefix). Existing filler is never
regenerated unless ``--new-attempt QID`` is given (which creates a new filler_id
alongside the old one; nothing is overwritten).

    python -m src.generate_filler --config config/smoke.yaml            # estimate + confirm + run
    python -m src.generate_filler --config config/smoke.yaml --estimate-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from typing import Any

from src import filler_prompts as P
from src import original
from src.config import ExperimentConfig, load_config
from src.models import TrackedClient
from src.schemas import ChatMessage, FillerConversation, Question
from src.utils import (
    BudgetExceeded,
    BudgetGuard,
    JsonlStore,
    confirm_or_exit,
    count_message_tokens,
    run_parallel,
    sha256_json,
    setup_logging,
    stable_id,
    stable_seed,
)

MAX_CHUNKS = 40
MAX_CHUNK_RETRIES = 4
MAX_SUBTOPIC_SKIPS = 3  # a chunk that keeps failing moves on to the next subtopic pair
MAX_ATTEMPTS = 3  # per question, counting stored fillers and failed generations
EST_TOKENS_PER_EXCHANGE = 265  # planning estimate only (filler_v2 test chunk: ~1,590 tokens / 6 exchanges)


class GenerationError(RuntimeError):
    """Carries the partial generation log so failed attempts are kept for inspection."""

    def __init__(self, message: str, gen_log: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.gen_log = gen_log or []


# ---------------------------------------------------------------- selection / stores
def select_questions(cfg: ExperimentConfig, override: list[str] | None = None) -> list[Question]:
    allq = [Question.model_validate(q) for q in original.build_questions()]
    by_id = {q.id: q for q in allq}
    ids = override or cfg.question_ids or [q.id for q in allq[: cfg.n_questions]]
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown question ids: {missing}")
    return [by_id[i] for i in ids]


def stores(cfg: ExperimentConfig) -> tuple[JsonlStore, JsonlStore]:
    d = cfg.data_path
    return JsonlStore(d / "generated_filler.jsonl", "filler_id"), JsonlStore(d / "api_calls.jsonl", "call_id")


def filler_id_for(q: Question, cfg: ExperimentConfig, attempt: int) -> str:
    return stable_id(q.id, P.FILLER_PROMPT_VERSION, cfg.generator_model.model, attempt)


def existing_attempts(filler_store: JsonlStore, cfg: ExperimentConfig, q: Question) -> list[int]:
    """Attempt numbers already stored for this question, across prompt versions."""
    return [int(r["generator_params"]["attempt"]) for r in filler_store.iter_raw() if r["question_id"] == q.id]


# ---------------------------------------------------------------- helpers
def transcript(messages: list[dict[str, str]], start: int = 0) -> str:
    return "\n\n".join(f"[{start + i}] {m['role'].upper()}: {m['content']}" for i, m in enumerate(messages))


def parse_json(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def parse_exchanges(text: str) -> list[dict[str, str]]:
    obj = parse_json(text)
    ex = obj.get("exchanges")
    if not isinstance(ex, list) or not ex:
        raise ValueError("missing/empty 'exchanges'")
    msgs: list[dict[str, str]] = []
    for i, e in enumerate(ex):
        if not isinstance(e, dict) or set(e) != {"user", "assistant"}:
            raise ValueError(f"exchange {i} must have exactly 'user' and 'assistant'")
        u, a = e["user"], e["assistant"]
        if not (isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip()):
            raise ValueError(f"exchange {i} has empty or non-string content")
        msgs += [{"role": "user", "content": u.strip()}, {"role": "assistant", "content": a.strip()}]
    return msgs


def n_chunks_planned(cfg: ExperimentConfig) -> int:
    return math.ceil(cfg.filler.generation_target_tokens / (cfg.filler.chunk_exchanges * EST_TOKENS_PER_EXCHANGE))


def estimate_generation_cost(cfg: ExperimentConfig, n_questions: int) -> float:
    """Rough upper-ish estimate: every chunk re-sends the transcript so far."""
    price = cfg.pricing[cfg.generator_model.model]
    chunk_out = cfg.filler.chunk_exchanges * EST_TOKENS_PER_EXCHANGE
    k = n_chunks_planned(cfg)
    inp = 1500 * (k + 1) + sum(i * chunk_out for i in range(k))
    out = 800 + k * chunk_out
    per_q = (inp * price.input + out * price.output) / 1e6
    return per_q * n_questions


def avoid_hits(messages: list[dict[str, str]], keywords: list[str], start: int = 0) -> list[dict[str, Any]]:
    """Whole-word, case-insensitive matches of off-limits keywords (deterministic leakage guard)."""
    pats = [(k, re.compile(r"(?<!\w)" + re.escape(k.strip()) + r"(?!\w)", re.I)) for k in keywords if k.strip()]
    return [{"index": start + i, "keyword": k} for i, m in enumerate(messages) for k, pat in pats if pat.search(m["content"])]


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}


def repeated_messages(new: list[dict[str, str]], prior: list[dict[str, str]], threshold: float = 0.5,
                      start: int = 0) -> list[dict[str, Any]]:
    """New messages that (near-)duplicate an earlier or same-chunk message (5-word-shingle Jaccard)."""
    seen = [(i, _shingles(m["content"]), m["content"].strip()) for i, m in enumerate(prior)]
    hits = []
    for k, m in enumerate(new):
        sh, txt = _shingles(m["content"]), m["content"].strip()
        for i, s2, t2 in seen:
            if txt == t2 or (sh and s2 and len(sh & s2) / len(sh | s2) >= threshold):
                hits.append({"index": start + k, "duplicates": i})
                break
        seen.append((len(prior) + k, sh, txt))
    return hits


# ---------------------------------------------------------------- generation
def generate_one(q: Question, cfg: ExperimentConfig, client: TrackedClient, attempt: int, log) -> FillerConversation:
    fid = filler_id_for(q, cfg, attempt)
    rl = P.rationale_line(q.metadata.reference_rationale)
    gen_log: list[dict[str, Any]] = []
    k = n_chunks_planned(cfg)
    broadened = q.id in cfg.filler.broadened_question_ids

    # 1) subtopic outline
    outline_user = P.OUTLINE_USER.format(
        question=q.prompt, false_premise=q.false_premise, rationale_line=rl, n_topics=2 * k + 4
    ) + (P.OUTLINE_BROADENED_NOTE if broadened else "")
    outline = None
    feedback = ""
    for r in range(MAX_CHUNK_RETRIES + 1):
        resp, call_id = client.chat(
            [{"role": "system", "content": P.GENERATOR_SYSTEM}, {"role": "user", "content": outline_user + feedback}],
            purpose="filler_generation", ref_id=fid, seed=stable_seed(fid, "outline", r), max_tokens=1500,
            response_schema=P.OUTLINE_SCHEMA,
        )
        entry = {"step": "outline", "retry": r, "user_prompt": outline_user + feedback, "raw_output": resp.text, "call_id": call_id}
        gen_log.append(entry)
        try:
            obj = parse_json(resp.text)
            subs = [s for s in obj["subtopics"] if isinstance(s, str) and s.strip()]
            if len(subs) < k or not isinstance(obj.get("subject_area"), str):
                raise ValueError("outline too short or missing subject_area")
            answer_facts = [f for f in obj.get("answer_facts", []) if isinstance(f, str) and f.strip()]
            if not answer_facts:
                raise ValueError("outline has no answer_facts")
            concepts = [c for c in obj["avoid_concepts"] if isinstance(c, str) and c.strip()]
            keywords = [w.strip().lower() for w in obj["avoid_keywords"] if isinstance(w, str) and w.strip()]
            if not concepts or len(keywords) < 3:
                raise ValueError("outline has too few off-limits concepts/keywords")
            clash = avoid_hits([{"content": t} for t in [obj["subject_area"], *subs]], keywords)
            if clash:
                words = sorted({h["keyword"] for h in clash})
                feedback = P.OUTLINE_RETRY_FEEDBACK.format(
                    problem=f"the subject area/subtopics contain off-limits keywords {words}")
                raise ValueError(f"subject area/subtopics use off-limits keywords {words}")
            outline = (obj["subject_area"], subs, answer_facts + concepts, keywords)
            break
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            entry["parse_error"] = str(exc)
    if outline is None:
        raise GenerationError(f"{q.id}: outline failed after retries", gen_log)
    subject_area, subtopics, avoid_concepts, avoid_keywords = outline

    # 2) chunks until long enough
    messages: list[dict[str, str]] = []
    total = 0
    chunk = 0
    topic_pos = 0
    skipped: list[dict[str, Any]] = []
    while total < cfg.filler.generation_target_tokens:
        if chunk >= MAX_CHUNKS:
            log.warning(f"{q.id}: hit MAX_CHUNKS at {total} tokens")
            break
        subs = subtopics[topic_pos % len(subtopics) : topic_pos % len(subtopics) + 2]
        continuation = P.CONTINUATION_START if not messages else P.CONTINUATION_NEXT.format(transcript=transcript(messages))
        user = P.CHUNK_USER.format(
            false_premise=q.false_premise, rationale_line=rl, subject_area=subject_area,
            avoid_concepts="; ".join(avoid_concepts), avoid_keywords=", ".join(avoid_keywords),
            subtopics="; ".join(subs), continuation=continuation, n_exchanges=cfg.filler.chunk_exchanges,
        )
        new = None
        feedback = ""
        for r in range(MAX_CHUNK_RETRIES + 1):
            resp, call_id = client.chat(
                [{"role": "system", "content": P.GENERATOR_SYSTEM}, {"role": "user", "content": user + feedback}],
                purpose="filler_generation", ref_id=fid, seed=stable_seed(fid, chunk, len(skipped), r),
                response_schema=P.EXCHANGES_SCHEMA,
            )
            # The transcript part of `user` is reconstructable from earlier chunks; store the rest.
            entry = {
                "step": "chunk", "chunk": chunk, "retry": r, "subtopics": subs,
                "user_prompt_without_transcript": (user.replace(transcript(messages), "<TRANSCRIPT SO FAR>") if messages else user) + feedback,
                "raw_output": resp.text, "finish_reason": resp.finish_reason, "call_id": call_id,
            }
            gen_log.append(entry)
            try:
                if resp.finish_reason == "length":
                    raise ValueError("output truncated (finish_reason=length)")
                new = parse_exchanges(resp.text)
                hits = avoid_hits(new, avoid_keywords, start=len(messages))
                if hits:
                    entry["avoid_hits"] = hits
                    new = None
                    words = sorted({h["keyword"] for h in hits})
                    feedback = P.RETRY_FEEDBACK.format(words=", ".join(words))
                    raise ValueError(f"off-limits keywords used: {words}")
                reps = repeated_messages(new, messages, start=len(messages))
                if reps:
                    entry["repeated"] = reps
                    new = None
                    feedback = P.REPEAT_FEEDBACK
                    raise ValueError(f"{len(reps)} messages repeat earlier ones")
                break
            except (ValueError, json.JSONDecodeError) as exc:
                entry["parse_error"] = str(exc)
        topic_pos += 2
        if new is None:
            if len(skipped) >= MAX_SUBTOPIC_SKIPS:
                raise GenerationError(f"{q.id}: chunk {chunk} failed after retries ({len(skipped)} subtopic skips)", gen_log)
            skipped.append({"chunk": chunk, "subtopics": subs})
            log.warning(f"{q.id}: chunk {chunk} failed with subtopics {subs}; moving to the next subtopics")
            continue
        messages += new
        total = count_message_tokens(messages, cfg.filler.tokenizer, cfg.filler.message_overhead_tokens)
        log.info(f"{q.id}: chunk {chunk} -> {len(messages)} msgs, {total} tokens")
        chunk += 1

    spec = cfg.generator_model
    return FillerConversation(
        filler_id=fid,
        question_id=q.id,
        messages=[ChatMessage(**m) for m in messages],
        total_tokens=total,
        tokenizer=cfg.filler.tokenizer,
        generator_model=spec.model,
        generator_params={
            "temperature": spec.temperature, "max_tokens": spec.max_tokens, "json_mode": spec.json_mode,
            "chunk_exchanges": cfg.filler.chunk_exchanges, "target_tokens": cfg.filler.generation_target_tokens,
            "attempt": attempt, "subject_area": subject_area, "subtopics": subtopics,
            "avoid_concepts": avoid_concepts, "avoid_keywords": avoid_keywords, "skipped_subtopics": skipped,
            "system_prompt": P.GENERATOR_SYSTEM, "response_schemas": [P.OUTLINE_SCHEMA, P.EXCHANGES_SCHEMA],
        },
        generator_prompt_version=P.FILLER_PROMPT_VERSION,
        topic_scope="broadened" if broadened else "standard",
        generation_log=gen_log,
        content_sha256=sha256_json(messages),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", help="comma-separated IDs (overrides config)")
    ap.add_argument("--new-attempt", action="append", default=[], help="QID to regenerate as a new attempt")
    ap.add_argument("--regenerate-flagged", action="store_true",
                    help=f"new attempt for questions with no usable filler (max {MAX_ATTEMPTS} tries per question)")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="questions generated in parallel")
    ap.add_argument("--yes", action="store_true", help="skip interactive confirmation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg.generator_model.json_mode:
        raise SystemExit("generator_model.json_mode must be true")
    if cfg.filler.prompt_version != P.FILLER_PROMPT_VERSION:
        raise SystemExit(f"config filler.prompt_version {cfg.filler.prompt_version!r} != code {P.FILLER_PROMPT_VERSION!r}")
    log = setup_logging("generate_filler", cfg.data_path.parent / "logs")
    qs = select_questions(cfg, args.questions.split(",") if args.questions else None)
    if not args.questions:  # only items with trials at some L > 0 need filler
        need = set(cfg.filler_question_ids())
        qs = [q for q in qs if q.id in need]
    filler_store, ledger = stores(cfg)

    failures = JsonlStore(cfg.data_path / "filler_failures.jsonl", "failure_id")
    # Tries count only the current prompt version: earlier versions had pipeline bugs (see README).
    n_failed: dict[str, int] = {}
    for r in failures.iter_raw():
        if r.get("prompt_version") == P.FILLER_PROMPT_VERSION:
            n_failed[r["question_id"]] = n_failed.get(r["question_id"], 0) + 1
    n_stored_current: dict[str, int] = {}
    for r in filler_store.iter_raw():
        if r["generator_prompt_version"] == P.FILLER_PROMPT_VERSION:
            n_stored_current[r["question_id"]] = n_stored_current.get(r["question_id"], 0) + 1
    usable: set[str] = set()
    if args.regenerate_flagged:
        from src.validate_filler import usable_fillers  # local import: validate_filler imports this module

        usable = set(usable_fillers(cfg))

    todo: list[tuple[Question, int]] = []
    exhausted: list[str] = []
    for q in qs:
        attempts = existing_attempts(filler_store, cfg, q)
        tries = n_stored_current.get(q.id, 0) + n_failed.get(q.id, 0)
        nxt = (max(attempts) + 1) if attempts else 0
        if q.id in args.new_attempt:
            todo.append((q, nxt))
        elif args.regenerate_flagged and q.id not in usable:
            if tries >= MAX_ATTEMPTS:
                exhausted.append(q.id)
            else:
                todo.append((q, nxt))
        elif not attempts:
            if tries >= MAX_ATTEMPTS:
                exhausted.append(q.id)
            else:
                todo.append((q, 0))
    if exhausted:
        print(f"{len(exhausted)} questions reached {MAX_ATTEMPTS} tries without usable filler: {exhausted}")
    print(f"{len(qs)} questions selected; {len(todo)} to generate")
    if not todo:
        return
    est = estimate_generation_cost(cfg, len(todo))
    msg = (f"ESTIMATED filler-generation cost: ${est:.2f} for {len(todo)} questions "
           f"({cfg.generator_model.model}, ~{n_chunks_planned(cfg)} chunks each). Budget cap ${cfg.budget_usd:.2f}.")
    if args.estimate_only:
        print(msg)
        return
    confirm_or_exit(msg, assume_yes=args.yes)

    budget = BudgetGuard(ledger, cfg.budget_usd)
    client = TrackedClient(cfg.generator_model, cfg, ledger, budget)

    def work(item: tuple[Question, int]) -> None:
        q, attempt = item
        fid = filler_id_for(q, cfg, attempt)
        try:
            fc = generate_one(q, cfg, client, attempt, log)
        except GenerationError as exc:
            log.error(str(exc))
            failures.append({"failure_id": stable_id(fid, "failure", uuid.uuid4().hex), "filler_id": fid,
                             "question_id": q.id, "prompt_version": P.FILLER_PROMPT_VERSION, "attempt": attempt,
                             "error": str(exc), "generation_log": exc.gen_log})
            return
        filler_store.append(fc)
        log.info(f"{q.id}: stored filler {fc.filler_id} ({fc.total_tokens} tokens, {len(fc.messages)} msgs)")

    run_parallel(work, todo, args.workers, log)
    print(f"actual spend so far (all phases, from ledger): ${budget.spent_usd:.4f}")


if __name__ == "__main__":
    main()
