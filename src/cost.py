"""Cost computation and pre-run estimation.

Estimates are always printed and stored separately from actual costs (the
actual-cost ledger is data/api_calls.jsonl, written by models.TrackedClient).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import ExperimentConfig, Price
from src.schemas import Usage


def compute_cost(price: Price, usage: Usage) -> float:
    cached = usage.cached_prompt_tokens if price.cached_input is not None else 0
    uncached = usage.prompt_tokens - cached
    cost = uncached * price.input + usage.completion_tokens * price.output
    if cached:
        cost += cached * price.cached_input  # type: ignore[operator]
    return cost / 1_000_000


def max_call_cost(price: Price, prompt_tokens: int, max_tokens: int) -> float:
    """Upper bound for a single call (no caching, full max_tokens output)."""
    return (prompt_tokens * price.input + max_tokens * price.output) / 1_000_000


@dataclass
class LengthEstimate:
    target_tokens: int
    n_trials: int
    test_input_tokens: int = 0
    test_output_tokens: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    cost_usd_no_cache: float = 0.0
    cost_usd_with_cache: float = 0.0


@dataclass
class ExperimentEstimate:
    experiment: str
    per_length: list[LengthEstimate] = field(default_factory=list)

    @property
    def total_no_cache(self) -> float:
        return sum(x.cost_usd_no_cache for x in self.per_length)

    @property
    def total_with_cache(self) -> float:
        return sum(x.cost_usd_with_cache for x in self.per_length)

    def table(self) -> str:
        rows = [
            f"Cost estimate for '{self.experiment}' (ESTIMATE, not actual spend)",
            f"{'length':>8} {'trials':>7} {'test_in':>12} {'test_out':>10} {'judge_in':>10} {'$ no-cache':>11} {'$ cached':>10}",
        ]
        for x in self.per_length:
            rows.append(
                f"{x.target_tokens:>8} {x.n_trials:>7} {x.test_input_tokens:>12,} {x.test_output_tokens:>10,} "
                f"{x.judge_input_tokens:>10,} {x.cost_usd_no_cache:>11.2f} {x.cost_usd_with_cache:>10.2f}"
            )
        rows.append(f"{'TOTAL':>8} {'':>7} {'':>12} {'':>10} {'':>10} {self.total_no_cache:>11.2f} {self.total_with_cache:>10.2f}")
        return "\n".join(rows)


def estimate_trials(
    cfg: ExperimentConfig,
    base_prompt_tokens: dict[str, int],
    filler_tokens: dict[int, float],
    judge_base_tokens: int,
) -> ExperimentEstimate:
    """Estimate test-model + judge cost for the trial phase.

    base_prompt_tokens: {"unprimed": sys+question tokens, "primed": sys+primer+question}
    filler_tokens: {target_length: mean actual filler tokens} (use targets if no filler yet)
    judge_base_tokens: judge system prompt + template + question/premise (excluding responses)

    Cache model: turn 2 re-sends turn 1's prompt (cacheable prefix); repetitions
    re-send the same turn-1 prompt. Only prefixes >= 1024 tokens are counted as cacheable.
    """
    test_p = cfg.pricing[cfg.test_model.model]
    judge_p = cfg.pricing[cfg.judge_model.model]
    out_t = cfg.expected_output_tokens_per_turn
    est = ExperimentEstimate(cfg.experiment)
    for L in cfg.context_lengths:
        le = LengthEstimate(target_tokens=L, n_trials=0)
        ids, reps = cfg.runs_at(L)
        n_q = len(ids)
        for cond in cfg.conditions:
            n = n_q * reps
            p1 = base_prompt_tokens[cond] + filler_tokens[L]
            p2 = p1 + out_t + base_pushback_tokens()
            inp = n * (p1 + p2)
            out = n * 2 * out_t
            j_in = n * (judge_base_tokens + 2 * out_t)
            j_out = n * cfg.expected_judge_output_tokens
            le.n_trials += n
            le.test_input_tokens += int(inp)
            le.test_output_tokens += out
            le.judge_input_tokens += int(j_in)
            le.judge_output_tokens += j_out
            no_cache = compute_cost(test_p, Usage(prompt_tokens=int(inp), completion_tokens=out))
            # cached: turn-2 prefix (p1) always cacheable if >=1024; turn-1 cacheable on reps after the first.
            cached = 0.0
            if p1 >= 1024 and test_p.cached_input is not None:
                cached += n * p1  # turn 2's prefix
                cached += n_q * (reps - 1) * p1
            with_cache = compute_cost(
                test_p, Usage(prompt_tokens=int(inp), completion_tokens=out, cached_prompt_tokens=int(cached))
            )
            judge_cost = compute_cost(judge_p, Usage(prompt_tokens=int(j_in), completion_tokens=j_out))
            le.cost_usd_no_cache += no_cache + judge_cost
            le.cost_usd_with_cache += with_cache + judge_cost
        est.per_length.append(le)
    return est


def base_pushback_tokens() -> int:
    from src.original import pushback_prompt
    from src.utils import count_tokens

    return count_tokens(pushback_prompt()) + 4
