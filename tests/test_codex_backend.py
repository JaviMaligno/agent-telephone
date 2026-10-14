import json
import subprocess

import pytest

from agent_telephone.backends.codex_cli import CodexBackend, ensure_home, parse_events, render

EVENTS = [
    json.dumps({"type": "thread.started", "thread_id": "x"}),
    json.dumps({"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": "ok"}}),
    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 15412, "cached_input_tokens": 12288,
                                                    "cache_write_input_tokens": 0, "output_tokens": 5,
                                                    "reasoning_output_tokens": 7}}),
]


def test_parse_events_text_and_usage():
    text, u = parse_events(EVENTS)
    assert text == "ok"
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens) == (15412 - 12288, 12288, 12)


def test_render_keeps_roles():
    p = render("sys", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    assert p.index("<system>") < p.index("<user>") < p.index("<assistant>")


def test_command_is_isolated_and_stdin_closed(tmp_path):
    seen = {}

    def runner(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, 0, "\n".join(EVENTS), "")

    b = CodexBackend(home=tmp_path, runner=runner)
    out = b.complete("sys", [{"role": "user", "content": "hi"}], "gpt-6.1-sol", 100, schema={"type": "object"})
    assert out.text == "ok"
    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["env"]["CODEX_HOME"] == str(tmp_path)
    assert not any(k.startswith(("OPENAI", "ANTHROPIC")) for k in seen["env"])
    for flag in ("--ephemeral", "--ignore-rules", "--output-schema", "read-only"):
        assert flag in seen["cmd"]


def test_home_refuses_leaked_instructions(tmp_path):
    (tmp_path / "auth.json").write_text("{}")
    (tmp_path / "AGENTS.md").write_text("user instructions")
    with pytest.raises(RuntimeError):
        ensure_home(tmp_path)


def test_rate_limit_is_waited_out(tmp_path):
    fail = json.dumps({"type": "turn.failed", "error": {"message": "exceeded retry limit, last status: 429 Too Many Requests"}})
    outs = iter([subprocess.CompletedProcess([], 1, fail, ""), subprocess.CompletedProcess([], 0, "\n".join(EVENTS), "")])
    slept = []
    b = CodexBackend(home=tmp_path, runner=lambda cmd, **kw: next(outs), rate_limit_waits=(5,), sleep=slept.append)
    assert b.complete("s", [{"role": "user", "content": "x"}], "m", 10).text == "ok" and slept == [5]
