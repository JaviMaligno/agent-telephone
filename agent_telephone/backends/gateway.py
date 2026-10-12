"""OpenAI-compatible backend for the LiteLLM gateway. The key is read from a file and never printed.

Budget safety: `max_completion_tokens` is sent explicitly (reasoning models ignore or drop
`max_tokens`; reasoning tokens count towards it); a response without usage is settled at the
reserved worst case; a completion longer than `max_tokens` is recorded (and halts the ledger as
an overshoot) and then fails the call. Before any batch, verify on ONE cheap call that the
gateway honours the cap for the model (completion_tokens <= max_tokens).
"""
from __future__ import annotations

from pathlib import Path

import httpx

from ..budget import BudgetGuard, RequestNotSent, estimate_input_tokens
from ..schema import Usage
from .base import Completion

NOT_BILLED = {400, 401, 403, 404, 422}  # rejected before generation
RETRYABLE = {429}


class GatewayBackend:
    def __init__(self, base_url: str, key_path: str | Path, guard: BudgetGuard, timeout: float = 300.0,
                 reasoning_effort: str | None = None, retries: int = 2, post=httpx.post):
        if not base_url:
            raise ValueError("gateway base URL is required (--gateway-url or AGENT_TELEPHONE_GATEWAY_URL)")
        self.base_url = base_url.rstrip("/")
        self._key_path = Path(key_path).expanduser()
        self.guard, self.timeout, self.retries = guard, timeout, retries
        self.reasoning_effort = reasoning_effort
        self._post = post

    def __repr__(self) -> str:
        return f"GatewayBackend(base_url={self.base_url!r}, key=<redacted>)"

    def _key(self) -> str:
        return self._key_path.read_text(encoding="utf-8").strip()

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> Completion:
        body = {"model": model, "max_completion_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, *messages]}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        def do() -> Completion:
            try:
                r = self._post(f"{self.base_url}/v1/chat/completions", json=body, timeout=self.timeout,
                               headers={"Authorization": f"Bearer {self._key()}"})
            except httpx.ConnectError as e:
                raise RequestNotSent(f"gateway unreachable: {type(e).__name__}", retryable=True) from None
            if r.status_code in NOT_BILLED | RETRYABLE:  # keep the request (and key) out of the error
                raise RequestNotSent(f"gateway HTTP {r.status_code}: {r.text[:500]}", retryable=r.status_code in RETRYABLE)
            if r.status_code >= 400:
                raise RuntimeError(f"gateway HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            u = data.get("usage") or {}
            known = isinstance(u.get("prompt_tokens"), int) and isinstance(u.get("completion_tokens"), int)
            usage = Usage(input_tokens=u["prompt_tokens"], output_tokens=u["completion_tokens"]) if known else None
            return Completion(text=data["choices"][0]["message"]["content"] or "", usage=usage or Usage()), usage

        c, usage = self.guard.call(model, estimate_input_tokens(system, messages), max_tokens, do,
                                   usage_of=lambda r: r[1], retries=self.retries)
        if usage is not None and usage.output_tokens > max_tokens:
            raise RuntimeError(f"gateway ignored the output cap: {usage.output_tokens} > {max_tokens} completion tokens")
        return c
