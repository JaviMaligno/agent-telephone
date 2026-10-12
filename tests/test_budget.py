import threading
from pathlib import Path
from decimal import Decimal

import pytest

from agent_telephone.backends.base import Completion
from agent_telephone.budget import (DEFAULT_LEDGER, HARD_CAP_USD, BudgetExceeded, BudgetGuard, RequestNotSent,
                                    estimate_input_tokens, load_prices)
from agent_telephone.schema import Usage

PRICES = {"m": {"input": Decimal("1"), "output": Decimal("10")}}  # USD per million tokens


def _ok(inp, out):
    return lambda: Completion(text="x", usage=Usage(input_tokens=inp, output_tokens=out))


def test_repo_prices_claude_filled_others_refused(tmp_path):
    prices = load_prices()
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5"):
        assert all(v is not None and v > 0 for v in prices[model].values())
    guard = BudgetGuard(400, tmp_path / "l.ledger.json", init_ledger=True)
    for model in [m for m, p in prices.items() if any(v is None for v in p.values())]:
        with pytest.raises(BudgetExceeded):
            guard.call(model, 10, 10, _ok(1, 1))
    with pytest.raises(BudgetExceeded):
        guard.call("unknown-model", 10, 10, _ok(1, 1))
    assert guard.spent == 0


def test_refuses_null_price(tmp_path):
    guard = BudgetGuard(400, tmp_path / "l.ledger.json", {"m": {"input": Decimal("1"), "output": None}}, init_ledger=True)
    called = []
    with pytest.raises(BudgetExceeded):
        guard.call("m", 1, 1, lambda: called.append(1))
    assert not called


def test_refuses_worst_case_over_limit_by_one_cent(tmp_path):
    # worst case = 1M in * $1 + 100k out * $10 = $2.00
    guard = BudgetGuard("2.00", tmp_path / "l.ledger.json", PRICES, init_ledger=True)
    guard.call("m", 1_000_000, 100_000, _ok(0, 0))  # exactly at the limit: allowed, costs 0
    guard2 = BudgetGuard("1.99", tmp_path / "l2.ledger.json", PRICES, init_ledger=True)
    called = []
    with pytest.raises(BudgetExceeded):
        guard2.call("m", 1_000_000, 100_000, lambda: called.append(1))
    assert not called and guard2.committed == 0


def test_actual_cost_recorded_and_persists(tmp_path):
    path = tmp_path / "l.ledger.json"
    BudgetGuard("10", path, PRICES, init_ledger=True).call("m", 2_000_000, 100_000, _ok(1_000_000, 50_000))  # $1 + $0.5
    g2 = BudgetGuard("10", path, PRICES)
    assert g2.spent == Decimal("1.5") and g2.committed == Decimal("1.5")
    g2.call("m", 1_000_000, 0, _ok(1_000_000, 0))
    assert BudgetGuard("10", path, PRICES).spent == Decimal("2.5")
    # remaining 7.5: a 7.51 worst case is refused by a fresh instance
    with pytest.raises(BudgetExceeded):
        BudgetGuard("10", path, PRICES).call("m", 7_510_000, 0, _ok(0, 0))


def _raise(exc):
    def fn():
        raise exc
    return fn


@pytest.mark.parametrize("exc", [RuntimeError("timeout after send"), KeyError("choices"), KeyboardInterrupt()])
def test_failure_after_send_keeps_worst_case(tmp_path, exc):
    """A failure that may have been billed is settled at the reserved worst case, never at 0."""
    guard = BudgetGuard("10", tmp_path / "l.ledger.json", PRICES, init_ledger=True)
    with pytest.raises(type(exc)):
        guard.call("m", 1_000_000, 0, _raise(exc))  # worst case $1
    assert guard.spent == Decimal("1") and guard.committed == Decimal("1")


def test_request_not_sent_releases_and_retries_reserve_each(tmp_path):
    guard = BudgetGuard("10", tmp_path / "l.ledger.json", PRICES, init_ledger=True)
    with pytest.raises(RequestNotSent):
        guard.call("m", 1_000_000, 0, _raise(RequestNotSent("401")))
    assert guard.committed == 0
    attempts = []

    def flaky():
        attempts.append(guard.committed)  # each attempt runs under its own fresh reservation
        if len(attempts) < 3:
            raise RequestNotSent("429", retryable=True)
        return Completion(text="x", usage=Usage(input_tokens=500_000, output_tokens=0))

    guard.call("m", 1_000_000, 0, flaky, retries=2)
    assert attempts == [Decimal("1")] * 3 and guard.spent == Decimal("0.5")
    with pytest.raises(RequestNotSent):  # retries exhausted
        guard.call("m", 1_000_000, 0, _raise(RequestNotSent("429", retryable=True)), retries=1)
    assert guard.spent == Decimal("0.5")


def test_missing_usage_settles_worst_case(tmp_path):
    guard = BudgetGuard("10", tmp_path / "l.ledger.json", PRICES, init_ledger=True)
    guard.call("m", 1_000_000, 100_000, lambda: "text", usage_of=lambda r: None)
    assert guard.spent == Decimal("2")


def test_overshoot_halts_until_cleared(tmp_path):
    path = tmp_path / "l.ledger.json"
    guard = BudgetGuard("100", path, PRICES, init_ledger=True)
    guard.call("m", 1_000, 1_000, _ok(1_000, 50_000))  # reasoning model ignored max_tokens
    assert guard.spent == Decimal("0.501") and "reserved worst case" in guard.halted
    with pytest.raises(BudgetExceeded, match="halted"):
        BudgetGuard("100", path, PRICES).call("m", 1, 1, _ok(0, 0))  # persisted across instances
    guard.clear_halt()
    guard.call("m", 1, 1, _ok(0, 0))


def test_ledger_fails_closed(tmp_path):
    path = tmp_path / "l.ledger.json"
    with pytest.raises(BudgetExceeded, match="no ledger"):
        BudgetGuard("10", path, PRICES)
    guard = BudgetGuard("10", path, PRICES, init_ledger=True)
    guard.call("m", 1_000_000, 0, _ok(1_000_000, 0))
    BudgetGuard("10", path, PRICES, init_ledger=True)  # init never resets an existing ledger
    assert guard.spent == Decimal("1")
    path.unlink()  # `rm -rf runs`, `git clean -fdX`...
    with pytest.raises(BudgetExceeded):
        guard.call("m", 1, 1, _ok(0, 0))
    with pytest.raises(BudgetExceeded):
        BudgetGuard("10", path, PRICES)


def test_limit_is_clamped_and_ledger_is_absolute_outside_repo():
    with pytest.raises(ValueError):
        BudgetGuard("4000", "/nonexistent/l.json", PRICES)
    assert HARD_CAP_USD <= 450
    repo = Path(__file__).resolve().parent.parent
    assert DEFAULT_LEDGER.is_absolute() and repo not in DEFAULT_LEDGER.parents


def test_cache_tokens_are_charged_and_cached_calls_need_prices(tmp_path):
    guard = BudgetGuard("10", tmp_path / "l.ledger.json", PRICES, init_ledger=True)
    with pytest.raises(BudgetExceeded, match="cache prices"):
        guard.reserve("m", 10, 10, cache=True)
    # unpriced cache tokens: writes at 2x input, reads at 1x input (conservative)
    assert guard.cost("m", Usage(cache_write_tokens=1_000_000, cache_read_tokens=1_000_000)) == Decimal("3")


def test_input_estimate_is_conservative_for_non_ascii():
    assert estimate_input_tokens("", [{"content": "ąęłńśźż" * 100}]) >= 700


def test_crash_mid_call_keeps_reservation(tmp_path):
    path = tmp_path / "l.ledger.json"
    BudgetGuard("3", path, PRICES, init_ledger=True).reserve("m", 2_000_000, 0)  # process dies before settle
    with pytest.raises(BudgetExceeded):
        BudgetGuard("3", path, PRICES).call("m", 1_500_000, 0, _ok(0, 0))


def test_concurrent_threads_never_exceed_limit(tmp_path):
    path = tmp_path / "l.ledger.json"
    limit = Decimal("1.00")
    BudgetGuard(limit, path, PRICES, init_ledger=True)
    refused, done = [], []
    barrier = threading.Barrier(16)

    def worker():
        g = BudgetGuard(limit, path, PRICES)  # one instance per thread, same ledger
        barrier.wait()
        for _ in range(10):
            try:
                # worst case $0.03; actual cost equals the worst case
                g.call("m", 10_000, 2_000, _ok(10_000, 2_000))
                done.append(1)
            except BudgetExceeded:
                refused.append(1)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    spent = BudgetGuard(limit, path, PRICES).spent
    assert spent <= limit
    assert len(done) == 33 and spent == Decimal("0.99")
    assert len(refused) == 160 - 33


# --- backends: what reaches the ledger ----------------------------------------------------

class _Resp:
    def __init__(self, status, data=None, text=""):
        self.status_code, self._data, self.text = status, data, text

    def json(self):
        return self._data


def _gateway(tmp_path, resp, **kw):
    from agent_telephone.backends.gateway import GatewayBackend

    key = tmp_path / "key"
    key.write_text("sk-secret")
    guard = BudgetGuard("10", tmp_path / "g.ledger.json", {"gpt": PRICES["m"]}, init_ledger=True)
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs["json"])
        return resp() if callable(resp) else resp

    return GatewayBackend("http://gw.invalid", key, guard, post=post, **kw), guard, sent


def _choice(usage):
    return {"choices": [{"message": {"content": "hi"}}], **({"usage": usage} if usage is not None else {})}


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": None, "completion_tokens": None}])
def test_gateway_without_usage_is_charged_worst_case(tmp_path, usage):
    gw, guard, sent = _gateway(tmp_path, _Resp(200, _choice(usage)), reasoning_effort="low")
    assert gw.complete("s", [{"role": "user", "content": "x" * 2000}], "gpt", 1000).text == "hi"
    worst = guard.worst_case("gpt", estimate_input_tokens("s", [{"role": "user", "content": "x" * 2000}]), 1000)
    assert guard.spent == worst > 0
    assert sent[0]["max_completion_tokens"] == 1000 and sent[0]["reasoning_effort"] == "low"


def test_gateway_output_over_cap_fails_and_halts(tmp_path):
    gw, guard, _ = _gateway(tmp_path, _Resp(200, _choice({"prompt_tokens": 10, "completion_tokens": 5000})))
    with pytest.raises(RuntimeError, match="ignored the output cap"):
        gw.complete("s", [{"role": "user", "content": "x"}], "gpt", 100)
    assert guard.halted and guard.spent == Decimal("0.05001")


def test_gateway_error_classes(tmp_path):
    gw, guard, sent = _gateway(tmp_path, _Resp(401, text="bad key"))
    with pytest.raises(RequestNotSent):
        gw.complete("s", [{"role": "user", "content": "x"}], "gpt", 100)
    assert guard.spent == 0 and len(sent) == 1  # not retryable
    (tmp_path / "b").mkdir()
    gw, guard, _ = _gateway(tmp_path / "b", _Resp(502, text="proxy"))
    with pytest.raises(RuntimeError):
        gw.complete("s", [{"role": "user", "content": "x"}], "gpt", 100)
    assert guard.spent > 0  # may have been billed upstream


def test_anthropic_backend_counts_cache_and_disables_sdk_retries(tmp_path):
    from types import SimpleNamespace

    from agent_telephone.backends.anthropic_api import AnthropicBackend

    key = tmp_path / "key"
    key.write_text("sk-ant-test")
    guard = BudgetGuard("10", tmp_path / "a.ledger.json", {"claude-x": PRICES["m"]}, init_ledger=True)
    assert AnthropicBackend(guard, key).client.max_retries == 0
    usage = SimpleNamespace(input_tokens=1_000, output_tokens=100, cache_creation_input_tokens=1_000_000,
                            cache_read_input_tokens=None)
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        usage=usage, content=[SimpleNamespace(type="text", text="ok")])))
    c = AnthropicBackend(guard, client=client).complete("s", [{"role": "user", "content": "x"}], "claude-x", 100)
    assert c.text == "ok" and c.usage.cache_write_tokens == 1_000_000
    assert guard.spent == Decimal("2.002")  # 1k in + 100 out + 1M cache writes at 2x input
