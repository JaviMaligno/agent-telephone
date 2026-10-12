"""BudgetGuard: hard spending cap that survives restarts and concurrent callers, and fails closed.

Each paid call is wrapped by `guard.call(...)`:
1. worst case = input_tokens_estimate * in_price + max_tokens * out_price;
2. under a file lock, refuse if the ledger is halted or if spent + outstanding reservations +
   worst case > limit; otherwise persist a reservation for the worst case;
3. after the call, replace the reservation with the actual cost reported by the provider.

Failure handling (the provider may have billed a request that then failed on our side):
- the backend raises `RequestNotSent` only when the request provably never reached billing
  (connection refused, auth/validation/rate-limit rejection): the reservation is released at 0;
- any other exception (timeouts, Ctrl-C, malformed 200 responses...) and a response without
  usage are settled at the **reserved worst case**, never at 0;
- an actual cost above the reservation is recorded and **halts** the ledger: every later
  `reserve` is refused until a human calls `clear_halt()`.

Reservations are persisted, so a crash mid-call keeps the money counted. The ledger lives at
one absolute path outside the repo (DEFAULT_LEDGER); a missing ledger is refused unless the
caller explicitly initialises a new one (`init_ledger=True` / `--init-ledger`), so deleting it
never silently resets spend. The limit can never exceed HARD_CAP_USD.
Prices are USD per million tokens; a null input/output price makes the guard refuse.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import TypeVar

import yaml
from filelock import FileLock

from .schema import Usage

DEFAULT_LIMIT_USD = Decimal("400")
HARD_CAP_USD = Decimal("450")  # the account cap is USD 500 with no overspend; keep a margin
DEFAULT_PRICES = Path(__file__).resolve().parent.parent / "config" / "prices.yaml"
DEFAULT_LEDGER = (Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
                  / "agent-telephone" / "budget.ledger.json")
PRICE_KEYS = ("input", "output", "cache_write", "cache_read")
MILLION = Decimal(1_000_000)
T = TypeVar("T")


class BudgetExceeded(RuntimeError):
    """The call was refused: over the limit, unknown price, halted or missing ledger."""


class RequestNotSent(RuntimeError):
    """Raised by a backend when the request provably was not billed (safe to release at 0)."""

    def __init__(self, msg: str, retryable: bool = False):
        super().__init__(msg)
        self.retryable = retryable


def load_prices(path: str | Path = DEFAULT_PRICES) -> dict[str, dict[str, Decimal | None]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {
        model: {k: (None if p.get(k) is None else Decimal(str(p[k]))) for k in PRICE_KEYS}
        for model, p in (raw.get("models") or {}).items()
    }


def estimate_input_tokens(system: str, messages: list[dict]) -> int:
    """Conservative token estimate for worst-case pricing: 2 ASCII chars/token, 1 token per
    non-ASCII char (accented names tokenise badly), plus per-message overhead."""
    text = system + "".join(str(m.get("content", "")) for m in messages)
    non_ascii = sum(not c.isascii() for c in text)
    return math.ceil((len(text) - non_ascii) / 2) + non_ascii + 16 * (len(messages) + 1)


class BudgetGuard:
    def __init__(self, limit_usd: float | Decimal | str = DEFAULT_LIMIT_USD, ledger_path: str | Path = DEFAULT_LEDGER,
                 prices: dict | str | Path = DEFAULT_PRICES, init_ledger: bool = False):
        self.limit = Decimal(str(limit_usd))
        if not 0 <= self.limit <= HARD_CAP_USD:
            raise ValueError(f"budget limit ${self.limit} outside [0, {HARD_CAP_USD}]")
        self.ledger_path = Path(ledger_path).expanduser().resolve()
        self.prices = prices if isinstance(prices, dict) else load_prices(prices)
        self._lock_path = str(self.ledger_path) + ".lock"
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_path):
            if not self.ledger_path.exists():
                if not init_ledger:
                    raise BudgetExceeded(f"no ledger at {self.ledger_path}; refusing to start from $0. "
                                         "Pass init_ledger=True (--init-ledger) only for a genuinely new budget.")
                self._write({"spent": "0", "reservations": {}, "entries": [], "halted": None})

    # --- pricing ---------------------------------------------------------------
    def _price(self, model: str) -> dict[str, Decimal | None]:
        p = self.prices.get(model)
        if not p or p.get("input") is None or p.get("output") is None:
            raise BudgetExceeded(f"no price configured for model {model!r}; refusing to run")
        return p

    def cost(self, model: str, usage: Usage) -> Decimal:
        """Actual cost. Unpriced cache tokens are charged conservatively (writes 2x, reads 1x input)."""
        p = self._price(model)
        pin, pout = Decimal(str(p["input"])), Decimal(str(p["output"]))
        cw = p.get("cache_write") if p.get("cache_write") is not None else 2 * pin
        cr = p.get("cache_read") if p.get("cache_read") is not None else pin
        return (usage.input_tokens * pin + usage.output_tokens * pout
                + usage.cache_write_tokens * Decimal(str(cw)) + usage.cache_read_tokens * Decimal(str(cr))) / MILLION

    def worst_case(self, model: str, input_tokens_estimate: int, max_tokens: int, cache: bool = False) -> Decimal:
        p = self._price(model)
        pin = Decimal(str(p["input"]))
        if cache:
            if p.get("cache_write") is None or p.get("cache_read") is None:
                raise BudgetExceeded(f"cached call to {model!r} refused: cache prices are not configured")
            pin = max(pin, Decimal(str(p["cache_write"])))
        return (input_tokens_estimate * pin + max_tokens * Decimal(str(p["output"]))) / MILLION

    # --- ledger ----------------------------------------------------------------
    def _read(self) -> dict:
        if not self.ledger_path.exists():  # deleted while running: fail closed
            raise BudgetExceeded(f"ledger {self.ledger_path} disappeared; refusing")
        return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def _write(self, ledger: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.ledger_path.parent, prefix=".ledger-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.ledger_path)

    @property
    def spent(self) -> Decimal:
        with FileLock(self._lock_path):
            return Decimal(self._read()["spent"])

    @property
    def committed(self) -> Decimal:
        """Spent plus outstanding reservations."""
        with FileLock(self._lock_path):
            led = self._read()
            return Decimal(led["spent"]) + sum((Decimal(v) for v in led["reservations"].values()), Decimal(0))

    @property
    def halted(self) -> str | None:
        with FileLock(self._lock_path):
            return self._read().get("halted")

    def clear_halt(self) -> None:
        """Human action after investigating an overshoot."""
        with FileLock(self._lock_path):
            led = self._read()
            led["halted"] = None
            self._write(led)

    def reserve(self, model: str, input_tokens_estimate: int, max_tokens: int, cache: bool = False) -> str:
        worst = self.worst_case(model, input_tokens_estimate, max_tokens, cache)
        with FileLock(self._lock_path):
            led = self._read()
            if led.get("halted"):
                raise BudgetExceeded(f"ledger halted: {led['halted']}")
            outstanding = sum((Decimal(v) for v in led["reservations"].values()), Decimal(0))
            total = Decimal(led["spent"]) + outstanding + worst
            if total > self.limit:
                raise BudgetExceeded(f"worst case ${worst} would bring committed spend to ${total} > limit ${self.limit}")
            rid = uuid.uuid4().hex
            led["reservations"][rid] = str(worst)
            self._write(led)
            return rid

    def _close(self, rid: str, amount: Decimal | None, entry: dict) -> Decimal:
        """Remove a reservation and add `amount` (None = the reserved worst case) to spent."""
        with FileLock(self._lock_path):
            led = self._read()
            reserved = Decimal(led["reservations"].pop(rid, "0"))
            amount = reserved if amount is None else amount
            led["spent"] = str(Decimal(led["spent"]) + amount)
            led["entries"].append({**entry, "reserved": str(reserved), "cost": str(amount)})
            if amount > reserved:
                led["entries"].append({"status": "overshoot", "excess": str(amount - reserved)})
                led["halted"] = f"call cost ${amount} > reserved worst case ${reserved} ({entry.get('model')})"
            self._write(led)
        return amount

    def settle(self, rid: str, model: str, usage: Usage) -> Decimal:
        """Replace a reservation with the actual cost; an overshoot halts the ledger."""
        return self._close(rid, self.cost(model, usage), {"status": "ok", "model": model, **usage.model_dump()})

    def settle_worst(self, rid: str, model: str, reason: str) -> Decimal:
        """Outcome unknown (failed after sending, no usage reported): keep the worst case as spent."""
        return self._close(rid, None, {"status": "unknown", "model": model, "reason": reason[:200]})

    def release(self, rid: str, reason: str) -> None:
        """The request was provably not billed."""
        self._close(rid, Decimal(0), {"status": "not_sent", "reason": reason[:200]})

    def call(self, model: str, input_tokens_estimate: int, max_tokens: int, fn: Callable[[], T],
             usage_of: Callable[[T], Usage | None] = lambda r: r.usage, retries: int = 0, cache: bool = False) -> T:
        """Run `fn` under a reservation. `usage_of` returning None settles at the worst case.
        Retries (only for retryable `RequestNotSent`) each take their own reservation."""
        for attempt in range(retries + 1):
            rid = self.reserve(model, input_tokens_estimate, max_tokens, cache)
            try:
                result = fn()
            except RequestNotSent as e:
                self.release(rid, str(e))
                if e.retryable and attempt < retries:
                    continue
                raise
            except BaseException as e:  # may have been billed: count the worst case
                self.settle_worst(rid, model, f"{type(e).__name__}: {e}")
                raise
            usage = usage_of(result)
            if usage is None:
                self.settle_worst(rid, model, "provider reported no usage")
            else:
                self.settle(rid, model, usage)
            return result
        raise AssertionError("unreachable")
