"""Codex CLI backend: GPT models on the author's ChatGPT subscription (no API spend).

Isolation: a dedicated CODEX_HOME holding only a symlink to the user's auth.json, so the user's
global AGENTS.md, config.toml, skills and memories are never loaded (verified 2026-10-12: with the
default home the model reports the global AGENTS.md; with this home it reports none). Each call runs
`codex exec` in a fresh temp dir, read-only sandbox, ephemeral session, stdin closed (otherwise
`codex exec` waits on stdin forever).

Used both as an agent backend (gpt-6-luna) and as the cross-family judge for Claude runs
(gpt-6.1-sol). Conversation turns are flattened into one prompt: each call is stateless anyway.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ..schema import Usage
from .base import Completion

RATE_LIMIT_WAITS = (60, 300, 900, 1800, 3600, 3600, 3600, 3600)  # ~4 h in total, then give up


class RateLimited(RuntimeError):
    pass


DEFAULT_HOME = Path.home() / ".local/state/agent-telephone/codex-home"
ENV_ALLOW = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "SHELL")


def find_codex() -> str:
    """Newest codex binary: $AGENT_TELEPHONE_CODEX, PATH, then the ChatGPT app's daemon releases."""
    if p := os.environ.get("AGENT_TELEPHONE_CODEX"):
        return p
    from shutil import which

    if p := which("codex"):
        return p
    rel = sorted((Path.home() / ".codex/packages/app-server-daemon/releases").glob("*/bin/codex"),
                 key=lambda p: [int(x) if x.isdigit() else x for x in p.parts[-3].split("-")[0].split(".")])
    if rel:
        return str(rel[-1])
    raise FileNotFoundError("codex CLI not found; set AGENT_TELEPHONE_CODEX")


def ensure_home(home: Path = DEFAULT_HOME) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    link = home / "auth.json"
    if not link.exists():
        link.symlink_to(Path.home() / ".codex/auth.json")
    for stray in ("AGENTS.md", "config.toml"):
        if (home / stray).exists():
            raise RuntimeError(f"{home / stray} must not exist: it would leak user instructions/config")
    return home


def render(system: str, messages: list[dict]) -> str:
    parts = [f"<system>\n{system}\n</system>"] if system else []
    for m in messages:
        c = m["content"] if isinstance(m["content"], str) else "".join(b.get("text", "") for b in m["content"])
        parts.append(f"<{m['role']}>\n{c}\n</{m['role']}>")
    return "\n\n".join(parts)


def parse_events(lines: list[str]) -> tuple[str, Usage]:
    text, usage = "", Usage()
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "item.completed" and ev.get("item", {}).get("type") == "agent_message":
            text = ev["item"].get("text", "")
        elif ev.get("type") == "turn.completed":
            u = ev.get("usage") or {}
            cached = u.get("cached_input_tokens", 0) or 0
            usage = usage + Usage(input_tokens=(u.get("input_tokens", 0) or 0) - cached,
                                  output_tokens=(u.get("output_tokens", 0) or 0) + (u.get("reasoning_output_tokens", 0) or 0),
                                  cache_read_tokens=cached, cache_write_tokens=u.get("cache_write_input_tokens", 0) or 0)
    return text, usage


class CodexBackend:
    """`complete()` matches the Backend protocol; `max_tokens` is not enforceable through the CLI
    and is ignored (subscription, no spend). Pass `schema` to force a JSON answer."""

    def __init__(self, reasoning_effort: str | None = "medium", home: Path = DEFAULT_HOME, timeout: float = 600,
                 codex_bin: str | None = None, runner=subprocess.run, rate_limit_waits: tuple[float, ...] = RATE_LIMIT_WAITS,
                 sleep=time.sleep):
        self.effort, self.timeout, self.runner = reasoning_effort, timeout, runner
        self.waits, self.sleep = rate_limit_waits, sleep
        self.home = home if runner is not subprocess.run else ensure_home(home)
        self.bin = codex_bin or (find_codex() if runner is subprocess.run else "codex")

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int,
                 schema: dict | None = None) -> Completion:
        """A usage-limit 429 (subscription quota) is waited out with backoff, not raised."""
        for wait in (*self.waits, None):
            try:
                return self._complete_once(system, messages, model, max_tokens, schema)
            except RateLimited:
                if wait is None:
                    raise
                self.sleep(wait)

    def _complete_once(self, system: str, messages: list[dict], model: str, max_tokens: int,
                       schema: dict | None) -> Completion:
        with tempfile.TemporaryDirectory(prefix="agent-telephone-codex-") as wd:
            cmd = [self.bin, "exec", "--json", "--ignore-rules", "--ephemeral", "--skip-git-repo-check",
                   "-s", "read-only", "-m", model, "-o", "last.txt"]
            if self.effort:
                cmd += ["-c", f"model_reasoning_effort={self.effort}"]
            if schema is not None:
                Path(wd, "schema.json").write_text(json.dumps(schema), encoding="utf-8")
                cmd += ["--output-schema", "schema.json"]
            cmd.append(render(system, messages))
            env = {k: v for k, v in os.environ.items() if k in ENV_ALLOW} | {"CODEX_HOME": str(self.home)}
            proc = self.runner(cmd, cwd=wd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=self.timeout)
            if proc.returncode and "status: 429" in (proc.stdout or "") + (proc.stderr or ""):
                raise RateLimited("codex usage limit (429)")
            if proc.returncode:
                raise RuntimeError(f"codex exited {proc.returncode}: {(proc.stderr or '')[-400:]}")
            text, usage = parse_events(proc.stdout.splitlines())
            last = Path(wd, "last.txt")
            if last.exists():
                text = last.read_text(encoding="utf-8") or text
        return Completion(text=text, usage=usage)
