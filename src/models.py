"""Provider-agnostic chat clients.

``ChatClient`` is the protocol every provider adapter implements. Only an
OpenAI-compatible adapter exists now (OpenAI direct, OpenRouter, or any
compatible base_url). Anthropic/Gemini adapters can be added by implementing
``chat`` and registering them in ``make_client``; experiment code depends only
on the protocol and on ``TrackedClient``.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from src.config import ExperimentConfig, ModelSpec, api_key_for
from src.cost import compute_cost, max_call_cost
from src.schemas import APICallRecord, Usage
from src.utils import BudgetGuard, JsonlStore, count_message_tokens


@dataclass
class ChatResponse:
    text: str
    usage: Usage
    response_id: str | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    attempts: int = 1
    latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class ChatClient(Protocol):
    provider: str

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        json_mode: bool = False,
        response_schema: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResponse: ...


class EmptyResponseError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """OpenAI SDK client with our own retry/backoff (SDK retries disabled)."""

    def __init__(
        self,
        api_key: str,
        *,
        provider: str = "openai",
        base_url: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 6,
        sdk_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.provider = provider
        self.max_retries = max_retries
        self._sleep = sleep
        if sdk_client is None:
            from openai import OpenAI

            sdk_client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        self._sdk = sdk_client

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        import openai

        if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code >= 500
        return False

    def chat(
        self, messages, *, model, temperature, max_tokens, seed=None, json_mode=False, response_schema=None,
        reasoning_effort=None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {"model": model, "messages": list(messages)}
        if reasoning_effort is None:
            kwargs.update(temperature=temperature, max_tokens=max_tokens)
        else:  # reasoning models reject temperature and max_tokens
            kwargs.update(max_completion_tokens=max_tokens, reasoning_effort=reasoning_effort)
        if seed is not None:
            kwargs["seed"] = seed
        if response_schema is not None:  # strict Structured Outputs take precedence over plain JSON mode
            kwargs["response_format"] = {"type": "json_schema", "json_schema": response_schema}
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        attempt = 0
        t0 = time.monotonic()
        while True:
            attempt += 1
            try:
                resp = self._sdk.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt > self.max_retries or not self._retryable(exc):
                    raise
                delay = min(60.0, 2.0 ** (attempt - 1)) * (1 + random.random() * 0.25)
                self._sleep(delay)

        choice = resp.choices[0]
        text = choice.message.content
        if text is None:
            raise EmptyResponseError(f"empty content (finish_reason={choice.finish_reason})")
        u = resp.usage
        cached = 0
        details = getattr(u, "prompt_tokens_details", None) if u else None
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        return ChatResponse(
            text=text,
            usage=Usage(
                prompt_tokens=u.prompt_tokens if u else 0,
                completion_tokens=u.completion_tokens if u else 0,
                cached_prompt_tokens=cached,
            ),
            response_id=getattr(resp, "id", None),
            response_model=getattr(resp, "model", None),
            system_fingerprint=getattr(resp, "system_fingerprint", None),
            finish_reason=choice.finish_reason,
            attempts=attempt,
            latency_s=time.monotonic() - t0,
        )


def make_client(spec: ModelSpec) -> ChatClient:
    if spec.provider in ("openai", "openrouter", "openai_compatible"):
        return OpenAICompatibleClient(
            api_key_for(spec),
            provider=spec.provider,
            base_url=spec.resolved_base_url(),
            timeout_s=spec.timeout_s,
            max_retries=spec.max_retries,
        )
    raise NotImplementedError(
        f"provider {spec.provider!r} not implemented yet; add an adapter implementing ChatClient"
    )


class TrackedClient:
    """Wraps a ChatClient for one ModelSpec: budget check, cost, and ledger row per call."""

    def __init__(
        self,
        spec: ModelSpec,
        cfg: ExperimentConfig,
        ledger: JsonlStore,
        budget: BudgetGuard,
        client: ChatClient | None = None,
    ):
        self.spec = spec
        self.price = cfg.pricing[spec.model]
        self.ledger = ledger
        self.budget = budget
        self.client = client or make_client(spec)
        self.tokenizer = cfg.filler.tokenizer

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        purpose: str,
        ref_id: str | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> tuple[ChatResponse, str]:
        max_tokens = max_tokens or self.spec.max_tokens
        est_prompt = count_message_tokens(messages, self.tokenizer)
        if est_prompt + max_tokens > self.spec.context_window:
            raise ValueError(
                f"request needs ~{est_prompt}+{max_tokens} tokens > context window "
                f"{self.spec.context_window} for {self.spec.model}; refusing to truncate"
            )
        reserved = max_call_cost(self.price, est_prompt, max_tokens)
        self.budget.check(reserved)

        call_id = uuid.uuid4().hex
        params = {
            "temperature": self.spec.temperature,
            "max_tokens": max_tokens,
            "seed": seed if self.spec.use_seed else None,
            "json_mode": self.spec.json_mode,
            "response_schema": response_schema["name"] if response_schema else None,
            "base_url": self.spec.resolved_base_url(),
            "reasoning_effort": self.spec.reasoning_effort,
        }
        record = APICallRecord(
            call_id=call_id,
            purpose=purpose,  # type: ignore[arg-type]
            ref_id=ref_id,
            provider=self.spec.provider,
            model=self.spec.model,
            params=params,
        )
        try:
            resp = self.client.chat(
                messages,
                model=self.spec.model,
                temperature=self.spec.temperature,
                max_tokens=max_tokens,
                seed=params["seed"],
                json_mode=self.spec.json_mode,
                **({"response_schema": response_schema} if response_schema is not None else {}),
                **({"reasoning_effort": self.spec.reasoning_effort} if self.spec.reasoning_effort else {}),
            )
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"[:2000]
            self.ledger.append(record)
            self.budget.release(reserved)
            raise
        cost = compute_cost(self.price, resp.usage)
        record.usage = resp.usage
        record.cost_usd = cost
        record.latency_s = resp.latency_s
        record.attempts = resp.attempts
        record.response_id = resp.response_id
        record.response_model = resp.response_model
        record.system_fingerprint = resp.system_fingerprint
        record.finish_reason = resp.finish_reason
        self.ledger.append(record)
        self.budget.record(cost, reserved)
        return resp, call_id
