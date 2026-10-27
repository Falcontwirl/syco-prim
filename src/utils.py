"""Shared helpers: IDs, hashing, token counting, append-only storage, budget, logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------- IDs / hashing
def stable_id(*parts: Any, n: int = 16) -> str:
    """Deterministic short ID from the given parts (order matters)."""
    blob = json.dumps([str(p) for p in parts], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


def sha256_json(obj: Any) -> str:
    """SHA-256 of a canonical JSON encoding (used for exact payload verification)."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    """31-bit seed derived from parts. Recorded per call; never assumed deterministic."""
    return int(stable_id(*parts, n=8), 16) & 0x7FFFFFFF


# ---------------------------------------------------------------- tokens
@lru_cache(maxsize=4)
def _encoding(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str = "o200k_base") -> int:
    return len(_encoding(encoding).encode(text))


def count_message_tokens(
    messages: Sequence[dict[str, str]] | Sequence[BaseModel],
    encoding: str = "o200k_base",
    overhead_per_message: int = 4,
) -> int:
    """Content tokens + fixed per-message overhead (approximates chat formatting).

    This is the project's definition of "conversation length". The API's
    reported prompt_tokens is recorded alongside it as an empirical check.
    """
    total = 0
    for m in messages:
        content = m["content"] if isinstance(m, dict) else m.content  # type: ignore[union-attr]
        total += count_tokens(content, encoding) + overhead_per_message
    return total


def as_dicts(messages: Iterable[dict[str, str] | BaseModel]) -> list[dict[str, str]]:
    return [m if isinstance(m, dict) else m.model_dump() for m in messages]  # type: ignore[misc]


# ---------------------------------------------------------------- storage
class JsonlStore:
    """Append-only JSONL file keyed by an ID field.

    - ``append`` refuses duplicate keys (raw data is immutable).
    - ``has`` enables resuming without repeating successful calls.
    - Each append is flushed and fsynced so a crash loses at most the
      in-flight record.
    """

    def __init__(self, path: Path, key: str):
        self.path = Path(path)
        self.key = key
        self._keys: set[str] | None = None
        self._lock = threading.RLock()  # appends may come from worker threads

    def _load_keys(self) -> set[str]:
        if self._keys is None:
            self._keys = {row[self.key] for row in self.iter_raw()}
        return self._keys

    def iter_raw(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    # A torn final line from a crash: surface it, never silently drop.
                    raise RuntimeError(f"{self.path}:{lineno} is not valid JSON") from exc

    def load(self, model: type[T]) -> list[T]:
        return [model.model_validate(r) for r in self.iter_raw()]

    def has(self, key_value: str) -> bool:
        return key_value in self._load_keys()

    def keys(self) -> set[str]:
        return set(self._load_keys())

    def append(self, record: BaseModel | dict[str, Any]) -> None:
        row = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        k = row[self.key]
        with self._lock:
            if self.has(k):
                raise KeyError(f"{self.key}={k!r} already stored in {self.path.name}; records are immutable")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._load_keys().add(k)


# ---------------------------------------------------------------- budget
class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    """Hard spending cap computed from the actual-cost ledger (api_calls.jsonl).

    ``check(next_cost)`` raises before a call that could push actual spend
    past the cap, so runs stop gracefully between calls.
    """

    def __init__(self, ledger: JsonlStore, budget_usd: float):
        self.ledger = ledger
        self.budget_usd = budget_usd
        self._spent = sum(float(r.get("cost_usd", 0.0)) for r in ledger.iter_raw())
        self._lock = threading.Lock()
        self._reserved = 0.0  # max cost of calls in flight (parallel workers)

    @property
    def spent_usd(self) -> float:
        return self._spent

    def check(self, next_call_max_cost: float = 0.0) -> None:
        """Raise if the call could overshoot; otherwise reserve its max cost until ``record``/``release``."""
        with self._lock:
            if self._spent + self._reserved + next_call_max_cost > self.budget_usd:
                raise BudgetExceeded(
                    f"budget ${self.budget_usd:.2f} would be exceeded (spent ${self._spent:.4f} + in flight "
                    f"${self._reserved:.4f} + next call up to ${next_call_max_cost:.4f})"
                )
            self._reserved += next_call_max_cost

    def release(self, reserved: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - reserved)

    def record(self, cost_usd: float, reserved: float = 0.0) -> None:
        with self._lock:
            self._spent += cost_usd
            self._reserved = max(0.0, self._reserved - reserved)


# ---------------------------------------------------------------- logging / CLI
def setup_logging(name: str, log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Console + JSON-lines file logging."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(console)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

        class _Json(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "ts": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                payload.update(getattr(record, "extra_fields", {}))
                return json.dumps(payload, ensure_ascii=False)

        fh = logging.FileHandler(log_dir / f"{name}.jsonl", encoding="utf-8")
        fh.setFormatter(_Json())
        logger.addHandler(fh)
    return logger


class StopRun(RuntimeError):
    """Raised by a worker to stop the whole parallel run (e.g. budget exhausted)."""


def run_parallel(fn: Callable[[Any], Any], items: Sequence[Any], workers: int, log: logging.Logger) -> list[Any]:
    """Apply ``fn`` to items with a thread pool. BudgetExceeded/StopRun stop new work gracefully."""
    results: list[Any] = []
    stop = threading.Event()

    def guarded(item: Any) -> Any:
        if stop.is_set():
            return None
        try:
            return fn(item)
        except (BudgetExceeded, StopRun) as exc:
            if not stop.is_set():
                log.error(f"stopping: {exc}")
            stop.set()
            return None

    if workers <= 1:
        for it in items:
            results.append(guarded(it))
            if stop.is_set():
                break
        return results
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(guarded, it) for it in items]
        for f in as_completed(futs):
            results.append(f.result())
    return results


def confirm_or_exit(message: str, assume_yes: bool = False) -> None:
    """Require explicit typed confirmation before spending money."""
    print(message, file=sys.stderr)
    if assume_yes:
        print("(--yes given; proceeding)", file=sys.stderr)
        return
    if not sys.stdin.isatty():
        sys.exit("refusing to proceed without confirmation (non-interactive; pass --yes to override)")
    if input("Type 'yes' to proceed: ").strip().lower() != "yes":
        sys.exit("aborted by user")
