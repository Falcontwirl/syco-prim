"""Structured records persisted to data/*.jsonl.

Every record carries a stable ID so interrupted runs can resume without
repeating successful API calls (see utils.JsonlStore).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]
Condition = Literal["primed", "unprimed"]
JudgeLabel = Literal["sycophantic", "resistant", "unclear"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatMessage(BaseModel):
    role: Role
    content: str


class QuestionMetadata(BaseModel):
    source: Literal["seed", "sycon"]
    reference_rationale: str | None
    primer_topic_overlap: bool
    tier: int
    upstream_commit: str


class Question(BaseModel):
    """One original item. ``id``/``prompt``/``false_premise`` are verbatim upstream."""

    id: str
    prompt: str
    false_premise: str
    metadata: QuestionMetadata


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0


class APICallRecord(BaseModel):
    """One row of the cost/usage ledger (data/api_calls.jsonl)."""

    call_id: str
    purpose: Literal["filler_generation", "leakage_check", "trial_turn_1", "trial_turn_2", "judge", "other"]
    ref_id: str | None = None
    provider: str
    model: str
    params: dict[str, Any]
    usage: Usage | None = None
    cost_usd: float = 0.0
    latency_s: float | None = None
    attempts: int = 1
    response_id: str | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    timestamp_utc: str = Field(default_factory=utc_now)


class FillerConversation(BaseModel):
    """The single ~32k-token conversation generated once per question.

    Shorter conditions are prefixes of ``messages`` (see construct_prompts.select_prefix).
    """

    filler_id: str
    question_id: str
    messages: list[ChatMessage]
    total_tokens: int
    tokenizer: str
    generator_model: str
    generator_params: dict[str, Any]
    generator_prompt_version: str
    # "broadened": subject area chosen one level above the question's own topic (see
    # filler_prompts.OUTLINE_BROADENED_NOTE); records written before this field default to "standard".
    topic_scope: Literal["standard", "broadened"] = "standard"
    generation_log: list[dict[str, Any]] = Field(
        description="Per-chunk prompts, raw outputs and api call_ids, for reproducibility"
    )
    content_sha256: str
    created_at_utc: str = Field(default_factory=utc_now)


class FillerPrefix(BaseModel):
    question_id: str
    filler_id: str
    target_tokens: int
    n_messages: int
    actual_tokens: int
    deviation_pct: float | None
    within_tolerance: bool


class CheckResult(BaseModel):
    name: str
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class FillerValidation(BaseModel):
    validation_id: str  # stable_id(filler_id, validator version, llm on/off)
    validator_version: str
    filler_id: str
    question_id: str
    checks: list[CheckResult]
    leakage_llm: dict[str, Any] | None = None
    status: Literal["pass", "flag", "fail"]
    flags: list[str] = Field(default_factory=list)
    validated_at_utc: str = Field(default_factory=utc_now)


class FillerReview(BaseModel):
    """Manual decision on a filler (append-only; latest decision per filler wins)."""

    review_id: str
    filler_id: str
    question_id: str
    decision: Literal["approved", "rejected"]
    note: str
    reviewed_at_utc: str = Field(default_factory=utc_now)


class Trial(BaseModel):
    """One (question, length, condition, rep) trial: two test-model responses.

    The full payload is reconstructable from immutable filler + config; its
    SHA-256 is stored to verify reconstruction. ``messages_turn_2`` is stored
    only when the config sets ``store_full_messages``.
    """

    trial_id: str
    experiment: str
    question_id: str
    condition: Condition
    target_tokens: int
    filler_id: str | None
    filler_n_messages: int
    filler_actual_tokens: int
    rep: int
    model: str
    temperature: float
    max_tokens: int
    seed: int | None
    payload_sha256_turn_1: str
    payload_sha256_turn_2: str
    messages_turn_2: list[ChatMessage] | None = None
    response_turn_1: str
    response_turn_2: str
    call_id_turn_1: str
    call_id_turn_2: str
    # Context-passing check: our tiktoken count vs the API's reported prompt_tokens.
    est_prompt_tokens_turn_1: int | None = None
    est_prompt_tokens_turn_2: int | None = None
    api_prompt_tokens_turn_1: int | None = None
    api_prompt_tokens_turn_2: int | None = None
    finish_reason_turn_1: str | None = None
    finish_reason_turn_2: str | None = None
    timestamp_utc: str = Field(default_factory=utc_now)


class TrialTurn1(BaseModel):
    """Checkpoint of a completed turn 1, so a crash before turn 2 never repeats the call."""

    trial_id: str
    payload_sha256_turn_1: str
    response_turn_1: str
    call_id_turn_1: str
    est_prompt_tokens: int
    api_prompt_tokens: int
    finish_reason: str | None
    timestamp_utc: str = Field(default_factory=utc_now)


class JudgeResult(BaseModel):
    """Identical schema to the original runner's JudgeResult."""

    validated_false_premise_turn_1: bool
    validated_false_premise_turn_2: bool
    corrected_user_turn_1: bool
    corrected_user_turn_2: bool
    label: JudgeLabel
    rationale: str


class JudgeRecord(BaseModel):
    """Raw judge output plus parse result. ``label`` is None iff parsing failed."""

    judge_id: str  # stable_id(trial_id, judge_model, judge_prompt_sha256)
    trial_id: str
    judge_model: str
    judge_prompt_version: str
    judge_prompt_sha256: str
    raw_response: str | None
    parsed: JudgeResult | None
    label: JudgeLabel | None
    parse_error: str | None
    # Derived outcomes (None when unparsed)
    sycophantic_label: bool | None = None  # primary: label == "sycophantic"
    sycophancy_or_rule: bool | None = None  # original code's definition (secondary)
    call_id: str | None = None
    timestamp_utc: str = Field(default_factory=utc_now)
