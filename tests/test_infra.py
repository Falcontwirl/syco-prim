from pathlib import Path
from types import SimpleNamespace

try:
    import httpx2 as httpx  # openai>=3 uses httpx2
except ImportError:  # pragma: no cover
    import httpx
import openai
import pytest

from src.config import load_config
from src.cost import compute_cost, estimate_trials
from src.models import OpenAICompatibleClient, TrackedClient
from src.schemas import Usage
from src.utils import BudgetExceeded, BudgetGuard, JsonlStore, stable_id, stable_seed

ROOT = Path(__file__).resolve().parents[1]


# pilot: 20 questions minus 4 excluded for unusable filler (config excluded_question_ids)
@pytest.mark.parametrize("name,n", [("smoke", 8), ("pilot", 160), ("full", 3000)])
def test_configs_load_with_expected_trial_counts(name, n):
    cfg = load_config(ROOT / "config" / f"{name}.yaml")
    assert cfg.n_trials == n
    assert cfg.test_model.model == "gpt-4.1-mini"
    assert cfg.judge_model.model == "gpt-4o-mini" and cfg.judge_model.temperature == 0.0
    assert cfg.test_model.temperature == 0.2 and cfg.test_model.max_tokens == 2048


def test_stable_ids_deterministic():
    assert stable_id("a", 1) == stable_id("a", 1) != stable_id("a", 2)
    assert 0 <= stable_seed("q", 1000, 0) < 2**31


def test_jsonl_store_resume_and_immutability(tmp_path):
    s = JsonlStore(tmp_path / "x.jsonl", key="id")
    s.append({"id": "a", "v": 1})
    with pytest.raises(KeyError):
        s.append({"id": "a", "v": 2})
    s2 = JsonlStore(tmp_path / "x.jsonl", key="id")  # fresh process view
    assert s2.has("a") and not s2.has("b")
    assert list(s2.iter_raw()) == [{"id": "a", "v": 1}]


def test_cost_and_budget(tmp_path):
    cfg = load_config(ROOT / "config" / "smoke.yaml")
    p = cfg.pricing["gpt-4.1-mini"]
    c = compute_cost(p, Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, cached_prompt_tokens=500_000))
    assert c == pytest.approx(0.5 * 0.40 + 0.5 * 0.10 + 1.60)
    ledger = JsonlStore(tmp_path / "calls.jsonl", key="call_id")
    ledger.append({"call_id": "c1", "cost_usd": 1.5})
    g = BudgetGuard(ledger, budget_usd=2.0)
    g.check(0.4)
    with pytest.raises(BudgetExceeded):
        g.check(0.6)


def test_estimate_runs():
    cfg = load_config(ROOT / "config" / "pilot.yaml")
    est = estimate_trials(cfg, {"unprimed": 30, "primed": 250}, {L: L for L in cfg.context_lengths}, 900)
    assert sum(x.n_trials for x in est.per_length) == cfg.n_trials == 160
    assert est.total_with_cache <= est.total_no_cache
    assert "ESTIMATE" in est.table()


class _FakeCompletions:
    def __init__(self, fail_times, exc_factory):
        self.calls, self.fail_times, self.exc_factory = 0, fail_times, exc_factory

    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        return SimpleNamespace(
            id="r1", model=kw["model"], system_fingerprint="fp",
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
        )


def _rate_limit():
    req = httpx.Request("POST", "https://x")
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)


def _bad_request():
    req = httpx.Request("POST", "https://x")
    return openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)


def _client(fail_times, exc, max_retries=6):
    comp = _FakeCompletions(fail_times, exc)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    sleeps = []
    c = OpenAICompatibleClient("k", sdk_client=sdk, max_retries=max_retries, sleep=sleeps.append)
    return c, comp, sleeps


def test_backoff_retries_rate_limits():
    c, comp, sleeps = _client(3, _rate_limit)
    r = c.chat([{"role": "user", "content": "x"}], model="m", temperature=0, max_tokens=5)
    assert r.text == "hi" and r.attempts == 4 and comp.calls == 4
    assert len(sleeps) == 3 and sleeps[0] < sleeps[1] < sleeps[2]


def test_no_retry_on_client_errors():
    c, comp, _ = _client(1, _bad_request)
    with pytest.raises(openai.BadRequestError):
        c.chat([{"role": "user", "content": "x"}], model="m", temperature=0, max_tokens=5)
    assert comp.calls == 1


def test_gives_up_after_max_retries():
    c, comp, _ = _client(10, _rate_limit, max_retries=2)
    with pytest.raises(openai.RateLimitError):
        c.chat([{"role": "user", "content": "x"}], model="m", temperature=0, max_tokens=5)
    assert comp.calls == 3


def test_tracked_client_ledger_and_budget(tmp_path):
    cfg = load_config(ROOT / "config" / "smoke.yaml")
    ledger = JsonlStore(tmp_path / "calls.jsonl", key="call_id")
    budget = BudgetGuard(ledger, cfg.budget_usd)
    inner, _, _ = _client(0, _rate_limit)
    tc = TrackedClient(cfg.test_model, cfg, ledger, budget, client=inner)
    resp, call_id = tc.chat([{"role": "user", "content": "x"}], purpose="other", seed=1)
    rows = list(ledger.iter_raw())
    assert rows[0]["call_id"] == call_id and rows[0]["usage"]["prompt_tokens"] == 10
    assert rows[0]["params"]["seed"] == 1
    assert budget.spent_usd == pytest.approx(rows[0]["cost_usd"])
    with pytest.raises(ValueError, match="context window"):
        tc.chat([{"role": "user", "content": "x"}], purpose="other", max_tokens=cfg.test_model.context_window)
