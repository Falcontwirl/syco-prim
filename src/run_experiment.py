"""Run trials: (question, context length, condition, repetition) -> two test-model responses.

Per trial (CLAUDE.md §6):
    turn 1: [system] + (primer if primed) + filler prefix + [user: question]
    turn 2: turn 1 + [assistant: response 1] + [user: pushback]

Only fillers that are usable (validate_filler.usable_fillers) are used; a
question without one is reported and the run refuses to start. Prefixes that
miss the length tolerance also stop the run. Nothing is ever truncated.

Resumable and idempotent: completed trials (data/trials.jsonl) are skipped, and
completed first turns are checkpointed (data/trial_turn1.jsonl) so a crash
between turns never repeats a successful call.

    python -m src.run_experiment --config config/smoke.yaml --estimate-only
    python -m src.run_experiment --config config/smoke.yaml            # estimate + confirm + run
    python -m src.run_experiment --config config/smoke.yaml --report   # context-passing table, no calls
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from statistics import mean

from src import original
from src.config import ExperimentConfig, load_config
from src.construct_prompts import build_pair, build_turn1, build_turn2, check_fits, select_prefix
from src.cost import ExperimentEstimate, estimate_trials
from src.generate_filler import select_questions
from src.judge import build_judge_messages
from src.models import TrackedClient
from src.schemas import ChatMessage, FillerConversation, FillerPrefix, Question, Trial, TrialTurn1
from src.utils import (
    BudgetExceeded,
    BudgetGuard,
    JsonlStore,
    StopRun,
    confirm_or_exit,
    count_message_tokens,
    run_parallel,
    setup_logging,
    sha256_json,
    stable_id,
    stable_seed,
)
from src.validate_filler import usable_fillers


FAILURE_WINDOW = 200
MAX_FAILURES_PER_WINDOW = 10  # >5% failures in the last 200 trials stops the run


@dataclass
class TrialSpec:
    trial_id: str
    question: Question
    condition: str
    prefix: FillerPrefix
    filler: list[dict[str, str]]  # the selected prefix messages (identical across conditions)
    rep: int
    seed: int | None


def trial_id_for(cfg: ExperimentConfig, qid: str, filler_id: str | None, target: int, condition: str, rep: int) -> str:
    """Stable trial ID. At L = 0 the payload has no filler, so the ID ignores filler_id; L = 0 trials
    run before filler exists are then the same trials as in the full plan. (Trials stored before
    this rule, i.e. smoke/pilot L = 0, keep the IDs they were stored under.)"""
    m = cfg.test_model
    fid = (filler_id or "-") if target > 0 else "-"
    return stable_id(cfg.experiment, qid, fid, target, condition, rep, m.provider, m.model, m.temperature, m.max_tokens)


def trial_stores(cfg: ExperimentConfig) -> tuple[JsonlStore, JsonlStore, JsonlStore]:
    d = cfg.data_path
    return (
        JsonlStore(d / "trials.jsonl", "trial_id"),
        JsonlStore(d / "trial_turn1.jsonl", "trial_id"),
        JsonlStore(d / "api_calls.jsonl", "call_id"),
    )


# ---------------------------------------------------------------- planning
def plan_trials(
    cfg: ExperimentConfig, questions: list[Question], fillers: dict[str, FillerConversation],
    lengths: list[int] | None = None, reps: list[int] | None = None,
) -> list[TrialSpec]:
    """Trials in the config (optionally only some lengths / repetition indices).

    Raises if a needed filler is missing or any prefix misses tolerance. Order is
    (question, condition, rep, L ascending) so nested filler prefixes are sent close together,
    which lets the provider's automatic prompt cache reuse them; order never affects payloads.
    """
    lengths = cfg.context_lengths if lengths is None else [L for L in cfg.context_lengths if L in lengths]
    runs = {L: cfg.runs_at(L) for L in lengths}  # (question ids, n reps) per length
    planned = {L: (set(ids), [r for r in range(n) if reps is None or r in reps]) for L, (ids, n) in runs.items()}
    missing = [q.id for q in questions if any(L > 0 and q.id in planned[L][0] for L in lengths) and q.id not in fillers]
    if missing:
        raise SystemExit(f"no usable (validated/approved) filler for: {missing}. Run generate_filler/validate_filler first.")
    specs: list[TrialSpec] = []
    off_tolerance: list[str] = []
    for q in questions:
        q_lengths = [L for L in lengths if q.id in planned[L][0]]
        fc = fillers.get(q.id) if any(L > 0 for L in q_lengths) else None
        msgs = [m.model_dump() for m in fc.messages] if fc else []
        prefixes = {}
        for L in q_lengths:
            # L = 0 has no filler content, so it is planned (and identified) without a filler.
            fid = fc.filler_id if (fc and L > 0) else "-"
            pre = select_prefix(
                msgs if L > 0 else [], L, question_id=q.id, filler_id=fid, tolerance=cfg.tolerance_for(L),
                encoding=cfg.filler.tokenizer, overhead_per_message=cfg.filler.message_overhead_tokens,
            )
            if not pre.within_tolerance:
                off_tolerance.append(f"{q.id}@{L}: {pre.actual_tokens} tokens ({pre.deviation_pct:+.1f}%, "
                                     f"tolerance ±{cfg.tolerance_for(L):.0%})")
            prefixes[L] = pre
        for cond in cfg.conditions:
            for rep in sorted({r for _, rs in planned.values() for r in rs}):
                for L in q_lengths:
                    if rep not in planned[L][1]:
                        continue
                    pre = prefixes[L]
                    seed = stable_seed(cfg.seed_base, q.id, L, rep) if cfg.test_model.use_seed else None
                    specs.append(TrialSpec(
                        trial_id=trial_id_for(cfg, q.id, pre.filler_id, L, cond, rep), question=q, condition=cond,
                        prefix=pre, filler=msgs[: pre.n_messages], rep=rep, seed=seed,
                    ))
    if off_tolerance:
        raise SystemExit("filler prefixes outside tolerance:\n  " + "\n  ".join(off_tolerance))
    return specs


def estimate(cfg: ExperimentConfig, specs: list[TrialSpec]) -> ExperimentEstimate:
    """Cost estimate for the trial + judge phases using the actual filler prefix lengths."""
    q0 = specs[0].question.prompt
    base = {
        "unprimed": count_message_tokens(build_turn1(primed=False, filler_prefix=[], question=q0)),
        "primed": count_message_tokens(build_turn1(primed=True, filler_prefix=[], question=q0)),
    }
    # Lengths not in `specs` (a filtered plan) fall back to the target length.
    filler_tokens = {L: mean([s.prefix.actual_tokens for s in specs if s.prefix.target_tokens == L] or [L])
                     for L in cfg.context_lengths}
    judge_base = count_message_tokens(build_judge_messages(q0, specs[0].question.false_premise, "", ""))
    return estimate_trials(cfg, base, filler_tokens, judge_base)


# ---------------------------------------------------------------- execution
def run_trial(
    spec: TrialSpec,
    cfg: ExperimentConfig,
    client: TrackedClient,
    trials: JsonlStore,
    turn1_store: JsonlStore,
    turn1_done: dict[str, TrialTurn1],
    log,
) -> Trial:
    m = cfg.test_model
    pair = build_pair(spec.filler, spec.question.prompt)  # asserts identical filler across conditions
    turn1 = pair[spec.condition]
    sha1 = sha256_json(turn1)
    est1 = check_fits(turn1, m.max_tokens, m.context_window, cfg.filler.tokenizer)

    t1 = turn1_done.get(spec.trial_id)
    if t1 is not None:
        if t1.payload_sha256_turn_1 != sha1:
            raise RuntimeError(f"{spec.trial_id}: turn-1 payload changed since checkpoint; refusing to mix")
        log.info(f"{spec.trial_id}: reusing checkpointed turn 1")
    else:
        resp, call_id = client.chat(turn1, purpose="trial_turn_1", ref_id=spec.trial_id, seed=spec.seed)
        t1 = TrialTurn1(
            trial_id=spec.trial_id, payload_sha256_turn_1=sha1, response_turn_1=resp.text, call_id_turn_1=call_id,
            est_prompt_tokens=est1, api_prompt_tokens=resp.usage.prompt_tokens, finish_reason=resp.finish_reason,
        )
        turn1_store.append(t1)
        turn1_done[spec.trial_id] = t1

    turn2 = build_turn2(turn1, t1.response_turn_1)
    est2 = check_fits(turn2, m.max_tokens, m.context_window, cfg.filler.tokenizer)
    resp2, call_id2 = client.chat(turn2, purpose="trial_turn_2", ref_id=spec.trial_id, seed=spec.seed)
    for turn, fr in ((1, t1.finish_reason), (2, resp2.finish_reason)):
        if fr != "stop":
            log.warning(f"{spec.trial_id}: turn {turn} finish_reason={fr!r} (response kept as returned)")

    trial = Trial(
        trial_id=spec.trial_id, experiment=cfg.experiment, question_id=spec.question.id, condition=spec.condition,  # type: ignore[arg-type]
        target_tokens=spec.prefix.target_tokens, filler_id=None if spec.prefix.filler_id == "-" else spec.prefix.filler_id,
        filler_n_messages=spec.prefix.n_messages, filler_actual_tokens=spec.prefix.actual_tokens, rep=spec.rep,
        model=m.model, temperature=m.temperature, max_tokens=m.max_tokens, seed=spec.seed,
        payload_sha256_turn_1=sha1, payload_sha256_turn_2=sha256_json(turn2),
        messages_turn_2=[ChatMessage(**x) for x in turn2] if cfg.store_full_messages else None,
        response_turn_1=t1.response_turn_1, response_turn_2=resp2.text,
        call_id_turn_1=t1.call_id_turn_1, call_id_turn_2=call_id2,
        est_prompt_tokens_turn_1=est1, est_prompt_tokens_turn_2=est2,
        api_prompt_tokens_turn_1=t1.api_prompt_tokens, api_prompt_tokens_turn_2=resp2.usage.prompt_tokens,
        finish_reason_turn_1=t1.finish_reason, finish_reason_turn_2=resp2.finish_reason,
    )
    trials.append(trial)
    return trial


def report(cfg: ExperimentConfig, trials: JsonlStore) -> None:
    """Context-passing check: API-reported prompt tokens vs our local count, per trial."""
    rows = [t for t in trials.load(Trial) if t.experiment == cfg.experiment]
    n_primer = len(original.load_primer())
    print(f"{'trial_id':16} {'q':5} {'target':>6} {'cond':8} {'filler':>7} {'msgs1':>5} "
          f"{'est1':>7} {'api1':>7} {'api1/est1':>9} {'est2':>7} {'api2':>7} {'fr1':>6} {'fr2':>6}")
    for t in sorted(rows, key=lambda t: (t.question_id, t.target_tokens, t.rep, t.condition)):
        n1 = 1 + (n_primer if t.condition == "primed" else 0) + t.filler_n_messages + 1
        ratio = (t.api_prompt_tokens_turn_1 or 0) / (t.est_prompt_tokens_turn_1 or 1)
        print(f"{t.trial_id:16} {t.question_id:5} {t.target_tokens:>6} {t.condition:8} {t.filler_actual_tokens:>7} {n1:>5} "
              f"{t.est_prompt_tokens_turn_1:>7} {t.api_prompt_tokens_turn_1:>7} {ratio:>9.3f} "
              f"{t.est_prompt_tokens_turn_2:>7} {t.api_prompt_tokens_turn_2:>7} {str(t.finish_reason_turn_1):>6} {str(t.finish_reason_turn_2):>6}")
    print(f"{len(rows)} trials for experiment '{cfg.experiment}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--report", action="store_true", help="print the context-passing table and exit (no API calls)")
    ap.add_argument("--limit", type=int, help="run at most N new trials this invocation")
    ap.add_argument("--lengths", type=int, nargs="+", help="only these context lengths (subset of the config)")
    ap.add_argument("--question-ids", nargs="+", help="only these questions (subset of the config)")
    ap.add_argument("--reps", type=int, nargs="+", help="only these repetition indices (0-based)")
    ap.add_argument("--workers", type=int, default=1, help="trials run in parallel (turns within a trial stay ordered)")
    ap.add_argument("--yes", action="store_true", help="skip interactive confirmation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("run_experiment", cfg.data_path.parent / "logs")
    trials, turn1_store, ledger = trial_stores(cfg)
    if args.report:
        report(cfg, trials)
        return

    questions = [q for q in select_questions(cfg) if q.id not in cfg.excluded_question_ids]
    if cfg.excluded_question_ids:
        print(f"EXCLUDED (no usable filler; see data/filler_reviews.jsonl): {cfg.excluded_question_ids}")
    if args.question_ids:
        unknown = set(args.question_ids) - {q.id for q in questions}
        if unknown:
            raise SystemExit(f"--question-ids not in this config (or excluded): {sorted(unknown)}")
        questions = [q for q in questions if q.id in args.question_ids]
    subset = bool(args.lengths or args.question_ids or args.reps is not None)
    specs = plan_trials(cfg, questions, usable_fillers(cfg), lengths=args.lengths, reps=args.reps)
    if not specs:
        raise SystemExit("no trials match the given filters")
    remaining = [s for s in specs if not trials.has(s.trial_id)]
    todo = remaining if args.limit is None else remaining[: args.limit]
    est = estimate(cfg, specs)
    print(est.table())
    if subset:
        print("(estimate table: config totals; the trial count below is the filtered subset)")
    print(f"{len(specs)} trials selected; {len(specs) - len(remaining)} already done; {len(todo)} to run now. "
          f"Budget cap ${cfg.budget_usd:.2f} (shared ledger).")
    if args.estimate_only or not todo:
        return
    # The table covers every trial in the config; sum the per-trial estimate of what runs now.
    per_trial = {x.target_tokens: x.cost_usd_no_cache / x.n_trials for x in est.per_length if x.n_trials}
    est_now = sum(per_trial[s.prefix.target_tokens] for s in todo)
    msg = f"ESTIMATED cost of {len(todo)} trials (+ their judge calls, run separately): ~${est_now:.2f} without caching."
    if est_now > cfg.confirm_above_usd:
        confirm_or_exit(msg, assume_yes=args.yes)
    else:
        print(msg)

    budget = BudgetGuard(ledger, cfg.budget_usd)
    client = TrackedClient(cfg.test_model, cfg, ledger, budget)
    turn1_done = {t.trial_id: t for t in turn1_store.load(TrialTurn1)}
    lock = threading.Lock()
    outcomes: list[bool] = []  # completion order; drives the failure-rate stop

    def work(spec: TrialSpec) -> None:
        try:
            t = run_trial(spec, cfg, client, trials, turn1_store, turn1_done, log)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded in the ledger; trial stays missing and is retried on resume
            log.error(f"{spec.trial_id} ({spec.question.id}, {spec.prefix.target_tokens}, {spec.condition}) failed: "
                      f"{type(exc).__name__}: {exc}")
            ok = False
        else:
            ok = True
            log.info(f"{t.trial_id} {t.question_id} L={t.target_tokens} {t.condition} rep={t.rep}: "
                     f"api prompt tokens {t.api_prompt_tokens_turn_1}/{t.api_prompt_tokens_turn_2}")
        with lock:
            outcomes.append(ok)
            window = outcomes[-FAILURE_WINDOW:]
            n_fail = window.count(False)
            if len(outcomes) % 25 == 0:
                log.info(f"progress: {len(outcomes)}/{len(todo)} done, {outcomes.count(False)} failed")
        if n_fail > MAX_FAILURES_PER_WINDOW:
            raise StopRun(f"{n_fail} of the last {len(window)} trials failed (> {MAX_FAILURES_PER_WINDOW}); stopping")

    run_parallel(work, todo, args.workers, log)
    print(f"{outcomes.count(True)} trials completed, {outcomes.count(False)} failed this run; "
          f"actual spend so far (all phases, from ledger): ${budget.spent_usd:.4f}")


if __name__ == "__main__":
    main()
