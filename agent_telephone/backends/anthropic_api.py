"""Anthropic API backend (Phase 3, paid). Every attempt goes through BudgetGuard.

The SDK's own retries are disabled (max_retries=0): a retry inside one reservation would hide
up to three billed attempts. Retries go through `guard.call`, each with its own reservation,
and only for errors that were provably not billed.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx

from ..budget import BudgetGuard, RequestNotSent, estimate_input_tokens
from ..schema import Usage
from .base import Completion


def _not_sent(e: Exception) -> RequestNotSent | None:
    import anthropic

    if isinstance(e, (anthropic.RateLimitError, anthropic.OverloadedError)) or getattr(e, "status_code", None) == 529:
        return RequestNotSent(f"{type(e).__name__}", retryable=True)
    if isinstance(e, (anthropic.BadRequestError, anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                      anthropic.NotFoundError, anthropic.UnprocessableEntityError)):
        return RequestNotSent(f"{type(e).__name__}")
    if isinstance(e, anthropic.APIConnectionError) and not isinstance(e, anthropic.APITimeoutError) \
            and isinstance(e.__cause__, httpx.ConnectError):
        return RequestNotSent("connection refused", retryable=True)
    return None


def read_env_file(path: str | Path) -> dict[str, str]:
    """Parse KEY=VALUE lines of a .env file (no shell expansion)."""
    out = {}
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class AnthropicBackend:
    """`key_path` is either a file holding only the key or a .env with ANTHROPIC_API_KEY and
    ANTHROPIC_WORKSPACE_ID. Personal multi-workspace keys need the `anthropic-workspace-id` header."""

    def __init__(self, guard: BudgetGuard, key_path: str | Path | None = None, client=None, retries: int = 2,
                 workspace_id: str | None = None):
        self.guard, self.retries = guard, retries
        if client is None:
            import anthropic

            key = None
            if key_path:
                text = Path(key_path).expanduser().read_text(encoding="utf-8").strip()
                if "ANTHROPIC_API_KEY=" in text:
                    env = read_env_file(key_path)
                    key, workspace_id = env.get("ANTHROPIC_API_KEY") or None, workspace_id or env.get("ANTHROPIC_WORKSPACE_ID")
                else:
                    key = text
            workspace_id = workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID")
            headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
            client = anthropic.Anthropic(api_key=key, max_retries=0, default_headers=headers) if key \
                else anthropic.Anthropic(max_retries=0, default_headers=headers)
        self.client = client

    def __repr__(self) -> str:
        return "AnthropicBackend(key=<redacted>)"

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> Completion:
        def do() -> Completion:
            try:
                r = self.client.messages.create(model=model, system=system, messages=messages, max_tokens=max_tokens)
            except Exception as e:
                if (ns := _not_sent(e)) is not None:
                    raise ns from None
                raise
            u = r.usage
            usage = Usage(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                          cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
                          cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return Completion(text=text, usage=usage)

        return self.guard.call(model, estimate_input_tokens(system, messages), max_tokens, do, retries=self.retries)
