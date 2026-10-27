"""YAML experiment configuration (config/*.yaml)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.original import REPO_ROOT

Provider = Literal["openai", "openrouter", "openai_compatible", "anthropic", "gemini"]

DEFAULT_BASE_URLS = {
    "openai": None,  # SDK default
    "openrouter": "https://openrouter.ai/api/v1",
}
DEFAULT_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class ModelSpec(BaseModel):
    provider: Provider = "openai"
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(ge=1)
    context_window: int = Field(ge=1)
    timeout_s: float = 120.0
    max_retries: int = 6
    use_seed: bool = True
    json_mode: bool = False
    # Reasoning models (gpt-5.x, o-series): sent as reasoning_effort; temperature is not sent and
    # max_tokens becomes max_completion_tokens (it caps hidden reasoning plus visible output).
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None

    def resolved_base_url(self) -> str | None:
        return self.base_url or DEFAULT_BASE_URLS.get(self.provider)

    def resolved_key_env(self) -> str:
        env = self.api_key_env or DEFAULT_KEY_ENVS.get(self.provider)
        if env is None:
            raise ValueError(f"provider {self.provider!r} needs api_key_env")
        return env


class Price(BaseModel):
    """USD per 1M tokens. Verify against the provider's price page before paid runs."""

    input: float
    output: float
    cached_input: float | None = None


class FillerSpec(BaseModel):
    tokenizer: str = "o200k_base"
    message_overhead_tokens: int = 4
    # Generate past the largest target so the longest prefix can land within tolerance.
    generation_target_tokens: int = 33_500
    chunk_exchanges: int = 6
    prompt_version: str = "filler_v6"
    # Questions whose narrow topic is itself the answer (every standard attempt leaked):
    # generated with filler_prompts.OUTLINE_BROADENED_NOTE, stored with topic_scope="broadened".
    broadened_question_ids: list[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    experiment: str
    question_ids: list[str] | None = None
    n_questions: int | None = None  # first N items in original dataset order
    # Questions with no usable filler after the allowed attempts; excluded from trials and
    # reported (never silently dropped). Reasons are in data/filler_reviews.jsonl.
    excluded_question_ids: list[str] = Field(default_factory=list)
    context_lengths: list[int]
    length_tolerance: float = 0.05
    # Per-length overrides. Complete exchanges average ~265 tokens, so at 1,000 tokens the nearest
    # exchange boundary can be ~13% away; spec §3 prefers natural turn boundaries over exact counts.
    length_tolerance_overrides: dict[int, float] = Field(default_factory=dict)
    conditions: list[Literal["unprimed", "primed"]] = ["unprimed", "primed"]
    repetitions: int = Field(ge=1)
    # Budget designs: L > 0 trials only for items from these sources ("seed"/"sycon"; None = all
    # items) and with this many repetitions (None = `repetitions`). L = 0 always uses every item
    # and `repetitions`, so the original paper's own setting is always complete.
    long_context_question_sources: list[str] | None = None
    # Optional explicit allow-list for L > 0 (e.g. the items with usable filler); applied after
    # the source filter. Items left out still have their L = 0 trials.
    long_context_question_ids: list[str] | None = None
    long_context_repetitions: int | None = Field(default=None, ge=1)
    store_full_messages: bool = False
    seed_base: int = 20261015

    test_model: ModelSpec
    # The same model's id in the original run (data/original/results.jsonl), for the L = 0
    # replication table; None when the original did not test this model.
    upstream_model: str | None = None
    judge_model: ModelSpec
    # Judge-agreement check (src.judge --judge-model / --tiebreak; src.analyze --second-judge):
    # the second judge's model id, and a stronger judge run only where the first two disagree.
    second_judge_model: str | None = None
    tiebreak_judge_model: ModelSpec | None = None
    generator_model: ModelSpec
    leakage_model: ModelSpec
    filler: FillerSpec = FillerSpec()

    pricing: dict[str, Price]
    budget_usd: float = Field(gt=0)
    confirm_above_usd: float = 1.0
    expected_output_tokens_per_turn: int = 350
    expected_judge_output_tokens: int = 120

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    config_path: Path | None = None

    @model_validator(mode="after")
    def _check(self) -> "ExperimentConfig":
        if (self.question_ids is None) == (self.n_questions is None):
            raise ValueError("set exactly one of question_ids / n_questions")
        if sorted(self.context_lengths) != self.context_lengths or self.context_lengths[0] < 0:
            raise ValueError("context_lengths must be ascending and non-negative")
        specs = [self.test_model, self.judge_model, self.generator_model, self.leakage_model]
        for spec in specs + ([self.tiebreak_judge_model] if self.tiebreak_judge_model else []):
            if spec.model not in self.pricing:
                raise ValueError(f"no pricing entry for model {spec.model!r}")
        return self

    def tolerance_for(self, target_tokens: int) -> float:
        return self.length_tolerance_overrides.get(target_tokens, self.length_tolerance)

    @property
    def data_path(self) -> Path:
        return self.data_dir if self.data_dir.is_absolute() else REPO_ROOT / self.data_dir

    @property
    def results_path(self) -> Path:
        return self.results_dir if self.results_dir.is_absolute() else REPO_ROOT / self.results_dir

    def trial_question_ids(self) -> list[str]:
        """Items with trials at L = 0: the configured items minus exclusions, in dataset order."""
        from src.original import load_dataset_raw

        ids = self.question_ids or [q["id"] for q in load_dataset_raw()[: self.n_questions]]
        return [i for i in ids if i not in self.excluded_question_ids]

    def runs_at(self, target_tokens: int) -> tuple[list[str], int]:
        """(question ids, repetitions) planned at this context length."""
        ids = self.trial_question_ids()
        if target_tokens == 0:
            return ids, self.repetitions
        if self.long_context_question_sources is not None:
            from src.original import build_questions

            src = {q["id"]: q["metadata"]["source"] for q in build_questions()}
            ids = [i for i in ids if src[i] in self.long_context_question_sources]
        if self.long_context_question_ids is not None:
            ids = [i for i in ids if i in self.long_context_question_ids]
        return ids, self.long_context_repetitions or self.repetitions

    def filler_question_ids(self) -> list[str]:
        """Items that need filler: those with trials at some L > 0."""
        need = {i for L in self.context_lengths if L > 0 for i in self.runs_at(L)[0]}
        return [i for i in self.trial_question_ids() if i in need]

    @property
    def n_trials(self) -> int:
        return sum(len(self.runs_at(L)[0]) * self.runs_at(L)[1] * len(self.conditions) for L in self.context_lengths)


def load_config(path: str | Path) -> ExperimentConfig:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:  # pragma: no cover
        pass
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["config_path"] = str(path)
    return ExperimentConfig.model_validate(raw)


def api_key_for(spec: ModelSpec) -> str:
    env = spec.resolved_key_env()
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"environment variable {env} is not set (see .env.example)")
    return key
