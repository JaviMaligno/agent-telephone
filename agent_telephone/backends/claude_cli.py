"""Single-shot Claude backend on the subscription: `claude -p` with the isolation of runner_cc
(no user CLAUDE.md, hooks, plugins, MCP, memory; fresh temp dir; no tools) and `--json-schema`
for structured answers. Used for the model annotator / adjudicator, not for agent runs.
"""
from __future__ import annotations

import json
import subprocess
import tempfile

from ..runner_cc import _assert_outside_repos, isolated_env
from ..schema import Usage
from .base import Completion
from .codex_cli import render


class ClaudeCliBackend:
    def __init__(self, effort: str | None = None, timeout: float = 600, claude_bin: str = "claude",
                 runner=subprocess.run):
        self.effort, self.timeout, self.bin, self.runner = effort, timeout, claude_bin, runner

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int,
                 schema: dict | None = None) -> Completion:
        with tempfile.TemporaryDirectory(prefix="agent-telephone-claude-") as wd:
            _assert_outside_repos(__import__("pathlib").Path(wd).resolve())
            cmd = [self.bin, "-p", render("", messages), "--model", model, "--system-prompt", system,
                   "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                   "--disable-slash-commands", "--tools", "", "--permission-mode", "dontAsk",
                   "--no-session-persistence", "--output-format", "json"]
            if self.effort:
                cmd += ["--effort", self.effort]
            if schema is not None:
                cmd += ["--json-schema", json.dumps(schema)]
            proc = self.runner(cmd, cwd=wd, env=isolated_env(), stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=self.timeout)
        if proc.returncode:
            raise RuntimeError(f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout or '')[-400:]}")
        out = json.loads(proc.stdout)
        if out.get("is_error"):
            raise RuntimeError(f"claude error: {str(out.get('result'))[:400]}")
        if out.get("apiKeySource", "none") not in ("none", None):
            raise RuntimeError("claude did not run on the subscription")
        text = json.dumps(out["structured_output"]) if out.get("structured_output") is not None else out.get("result", "")
        u = out.get("usage") or {}
        return Completion(text=text, usage=Usage(input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
                                                 cache_read_tokens=u.get("cache_read_input_tokens", 0),
                                                 cache_write_tokens=u.get("cache_creation_input_tokens", 0)))
