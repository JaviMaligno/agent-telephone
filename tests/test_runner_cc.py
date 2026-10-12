import json
import subprocess
from pathlib import Path

import pytest

from agent_telephone import runner_cc, topologies
from agent_telephone.schema import Condition, MessageLog, Scenario

FIX = Path(__file__).resolve().parent.parent / "fixtures"
NESTED, HIER, HUB = FIX / "cc_nested_stream.jsonl", FIX / "cc_isolated_hier_stream.jsonl", FIX / "cc_isolated_stream.jsonl"

# the scenario used for the isolated hub_relay capture (fixtures/cc_isolated_stream.jsonl)
SMOKE = Scenario.model_validate({
    "id": "cc_smoke", "domain": "software",
    "brief": ("Please write a Python function greet(name) that returns 'Hello, <name>!'. The first user is "
              "Siobhán Ní Dhúill, so keep accents intact. Do NOT log or print anything."),
    "facts": [{"id": "f_name", "type": "name", "text": "Siobhán Ní Dhúill",
               "check": {"kind": "exact", "pattern": "Siobhán Ní Dhúill"}},
              {"id": "f_nolog", "type": "negation", "text": "Do NOT log or print anything.", "check": {"kind": "judge"}}],
    "roles": [{"id": "lead", "title": "team lead", "instructions": "Coordinate."},
              {"id": "architect", "title": "software architect", "instructions": "Design."},
              {"id": "implementer", "title": "implementer", "instructions": "Write the final code."}],
    "endpoint": {"kind": "hidden_tests", "path": "tests.py"},
})


def _cond(**kw):
    return Condition(**{"topology": "hier_chain", "model": "haiku", "handoff_format": "free", **kw})


def _edges(hops):
    return [(h.kind, h.sender, h.receiver, h.distance) for h in hops]


def test_nested_fixture_edges_from_parent_ids_not_line_order():
    events = runner_cc.read_stream(NESTED)
    hops = runner_cc.parse_stream(events, "brief")
    assert _edges(hops) == [
        ("delegate", "user", "lead", 1), ("delegate", "lead", "pm", 2), ("delegate", "pm", "general-purpose", 3),
        ("return", "pm", "lead", 3),  # background: pm handed back before its child finished
        ("return", "general-purpose", "pm", 4), ("final", "lead", "user", 4)]
    assert hops[1].text.startswith("Client request (verbatim)") and "Siobhán" in hops[1].text
    assert hops[3].text.startswith("I launched an implementer agent")
    assert hops[4].text.startswith("I added a greeting function")
    assert hops[5].text.startswith("La función de saludo")  # contaminated: replied in Spanish
    assert hops[0].model == "human" and all(h.model == "claude-haiku-5-5" for h in hops[1:])


def test_nested_fixture_is_flagged_invalid():
    events = runner_cc.read_stream(NESTED)
    problems = runner_cc.check_run(events, runner_cc.parse_stream(events, "b"), {"lead": ["pm"], "pm": ["implementer"]})
    text = "\n".join(problems)
    for needle in ["hook ran", "MCP servers loaded", "user plugins loaded", "auto-memory", "lead has extra tools",
                   "background subagent", "non-delegation tool used: Write", "unexpected delegation pm -> general-purpose",
                   "delegation pm -> implementer happened 0 times"]:
        assert needle in text
    assert len(problems) == len(set(problems))


def test_isolated_hier_capture():
    events = runner_cc.read_stream(HIER)
    hops = runner_cc.parse_stream(events, "brief")
    assert _edges(hops) == [
        ("delegate", "user", "lead", 1), ("delegate", "lead", "pm", 2), ("delegate", "pm", "implementer", 3),
        ("return", "implementer", "pm", 4), ("return", "pm", "lead", 5), ("final", "lead", "user", 6)]
    assert "def add(a, b)" in hops[3].text and "[Subagent hand-back]" not in hops[3].text
    assert runner_cc.check_run(events, hops, {"lead": ["pm"], "pm": ["implementer"]}) == []
    assert all(h.usage.input_tokens > 0 for h in hops[1:])


def test_isolated_hub_relay_capture_record(tmp_path):
    cond = _cond(topology="hub_relay")
    log = tmp_path / "log.jsonl"
    res = runner_cc.record(runner_cc.read_stream(HUB), SMOKE, cond, log, "r1", HUB)
    assert res.valid, res.problems
    got = [(m.sender, m.receiver, m.distance) for m in res.messages]
    assert got[:4] == [tuple(e[:3]) for e in topologies.hub_relay(["lead", "architect", "implementer"])]
    # the lead still holds the brief when it briefs the implementer: information distance 2, not 4
    assert [(m.direction, m.info_distance) for m in res.messages[:4]] == [
        (e.direction, e.info_distance) for e in topologies.hub_relay(["lead", "architect", "implementer"])]
    assert res.messages[3].info_distance == 2 and res.messages[2].direction == "up"
    assert got[4:] == [("implementer", "lead", 5), ("lead", "user", 6), ("implementer", "endpoint", 5)]
    assert res.messages[0].text == SMOKE.brief and res.messages[0].model == "human"
    assert res.artifact_path == tmp_path / "r1.artifact.py"
    assert res.artifact_path.read_text() == res.messages[-1].text and "def greet" in res.messages[-1].text
    assert MessageLog(log).read() == res.messages


def test_check_run_flags_sibling_skipped_or_repeated():
    events = runner_cc.read_stream(HUB)
    hops = runner_cc.parse_stream(events, "b")
    allowed = {"lead": ["architect", "implementer"]}
    assert runner_cc.check_run(events, hops, allowed) == []
    no_r1 = [h for h in hops if h.receiver != "architect" and h.sender != "architect"]
    assert "delegation lead -> architect happened 0 times (expected 1)" in runner_cc.check_run(events, no_r1, allowed)
    assert any("happened 2 times" in p for p in runner_cc.check_run(events, hops + [hops[1]], allowed))


def test_build_command_and_agents():
    s = SMOKE.model_copy(update={"roles": [*SMOKE.roles, SMOKE.roles[2].model_copy(update={"id": "qa", "title": "QA"})]})
    cmd = runner_cc.build_command(s, _cond(), models={"qa": "sonnet"})
    for flag in ["-p", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
                 "--forward-subagent-text", "--verbose"]:
        assert flag in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--tools") + 1] == "Agent"
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("-p") + 1] == s.brief
    assert "Agent(general-purpose)" in cmd and "Agent(lead)" in cmd and "Agent(architect)" not in cmd
    assert "--bare" not in cmd and "--safe-mode" not in cmd
    agents = json.loads(cmd[cmd.index("--agents") + 1])
    assert list(agents) == ["architect", "implementer", "qa"]
    assert agents["architect"]["tools"] == ["Agent(implementer)"] and agents["implementer"]["tools"] == ["Agent(qa)"]
    assert agents["qa"]["tools"] == [] and agents["qa"]["model"] == "sonnet" and agents["architect"]["model"] == "haiku"
    assert 'subagent_type "architect"' in cmd[cmd.index("--system-prompt") + 1]
    assert "single Python module" in agents["qa"]["prompt"]


def test_hub_prompts_and_forward():
    agents = runner_cc.agents_json(SMOKE, _cond(topology="hub_relay", handoff_format="forward"))
    assert agents["architect"]["tools"] == [] and agents["implementer"]["tools"] == []
    lead = runner_cc.prompts(SMOKE, _cond(topology="hub_relay"))["lead"]
    assert 'subagent_type "architect"' in lead and 'subagent_type "implementer"' in lead
    assert runner_cc.formats.ATTACH_MARKER in agents["architect"]["prompt"]
    with pytest.raises(ValueError):
        runner_cc.routes("peer_route", ["a", "b", "c"])


def test_isolated_env_drops_api_keys(tmp_path):
    leaky = {"ANTHROPIC_API_KEY": "sk-x", "CLAUDE_CONFIG_DIR": "/x", "PATH": "/bin", "HOME": "/h",
             "CLAUDE_CODE_USE_FOUNDRY": "1", "ANTHROPIC_FOUNDRY_RESOURCE": "r", "CLAUDE_CODE_USE_VERTEX": "1",
             "ANTHROPIC_VERTEX_PROJECT_ID": "p", "CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": "x",
             "ANTHROPIC_CUSTOM_HEADERS": "h", "HTTPS_PROXY": "http://p", "GOOGLE_APPLICATION_CREDENTIALS": "/c"}
    env = runner_cc.isolated_env(leaky)
    assert set(env) == {"PATH", "HOME", *runner_cc.ISOLATION_ENV} and env["PATH"] == "/bin"
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1" and env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    tok = tmp_path / "tok"
    tok.write_text("oauth-token\n")
    env = runner_cc.isolated_env({}, oauth_token_file=tok)
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token" and Path(env["CLAUDE_CONFIG_DIR"]).is_dir()


def test_unwrap_and_extract():
    wrapped = ("[Subagent hand-back] blah. The report follows:\n  ```python\n  def f():\n      return 1\n  ```\n"
               "agentId: abc\n<usage>subagent_tokens: 1</usage>")
    assert runner_cc.unwrap_handback(wrapped) == "```python\ndef f():\n    return 1\n```"
    assert runner_cc.extract_artifact("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"
    assert runner_cc.extract_artifact("Here:\n```json\n{\"a\": 1}\n```\nDone.") == '{"a": 1}'
    assert runner_cc.extract_artifact("def f(): pass") == "def f(): pass"


def test_run_with_fake_claude(tmp_path):
    calls = []

    def fake(cmd, cwd, env, stdout, **kw):
        calls.append((cmd, cwd, env))
        stdout.write(HUB.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    (tmp_path / "wd").mkdir()
    res = runner_cc.run(SMOKE, _cond(topology="hub_relay"), tmp_path / "runs" / "log.jsonl", run_id="x",
                        workdir=tmp_path / "wd", runner=fake)
    assert res.valid and res.stream_path.read_text(encoding="utf-8") == HUB.read_text(encoding="utf-8")
    (cmd, cwd, env), = calls
    assert cwd == (tmp_path / "wd").resolve() and "ANTHROPIC_API_KEY" not in env


def test_run_refuses_inside_a_repo(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "wd").mkdir()
    with pytest.raises(RuntimeError, match="refusing"):
        runner_cc.run(SMOKE, _cond(topology="hub_relay"), tmp_path / "log.jsonl", workdir=tmp_path / "wd",
                      runner=lambda *a, **k: pytest.fail("claude must not be started"))


@pytest.mark.parametrize("init, assistant_model, needle", [
    ({"apiKeySource": "ANTHROPIC_API_KEY"}, None, "not billed to the subscription"),
    ({"model": "us.anthropic.claude-haiku-5-5-v1:0"}, None, "unexpected model"),
    ({}, "claude-haiku-5-5@vertex", "unexpected model"),
])
def test_check_run_flags_non_subscription_auth(init, assistant_model, needle):
    events = runner_cc.read_stream(HUB)
    assert runner_cc.check_run(events, runner_cc.parse_stream(events, "b"), {"lead": ["architect", "implementer"]}) == []
    for ev in events:
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            ev.update(init)
        if assistant_model and ev.get("type") == "assistant":
            ev["message"]["model"] = assistant_model
    problems = runner_cc.check_run(events, runner_cc.parse_stream(events, "b"), {"lead": ["architect", "implementer"]})
    assert any(needle in p for p in problems)


def test_published_files_carry_no_work_identity():
    """Fixtures, code and docs are meant to be public: no work email, home path, company or tooling."""
    root = Path(__file__).resolve().parent.parent
    forbidden = ["javier", "aguilar", "simplekyc", "skyc", "/users/", "/private/var/folders", "messaging_socket_path",
                 "argocd", "aks-tst", "litellm.infra"]
    files = [p for d in ("fixtures", "agent_telephone", "docs", "config", "scenarios") for p in (root / d).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts] + [root / "README.md", root / "DESIGN.md"]
    hits = [(p.name, w) for p in files for w in forbidden if w in p.read_text(encoding="utf-8", errors="ignore").lower()]
    assert hits == []


def test_hub_relay_with_four_roles_targets_the_last_role():
    from agent_telephone.schema import Condition, Scenario
    sc = Scenario.load("scenarios/software_pilot")
    cond = Condition(topology="hub_relay", model="haiku", handoff_format="free", noise="clean")
    agents = runner_cc.agents_json(sc, cond)
    assert set(agents) == {"pm", "implementer"}
    assert runner_cc.routes("hub_relay", [r.id for r in sc.roles]) == {"lead": ["pm", "implementer"]}
