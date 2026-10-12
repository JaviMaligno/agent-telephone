"""Claude Code runner (Phase 1): one isolated `claude -p` session per run.

The first scenario role is the lead (the main session, system prompt replaced); the
other roles are subagents defined with `--agents`, each with its own model. The log is
rebuilt from the stream-json output via `parent_tool_use_id` and tool_use/tool_result
pairs, never from line order (see `parse_stream`).

Isolation (verified 2026-10-12 on Claude Code 2.1.296, evidence in
docs/claude_code_isolation.md). The working invocation is what `build_command` +
`isolated_env` produce:

    env -i PATH=... HOME=... USER=... LANG=... (allowlist, see isolated_env) \
      CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 \
      CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=1 \
      CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    claude -p "<brief>" --model <lead model> \
      --setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' \
      --disable-slash-commands --system-prompt "<lead role>" --agents '<json>' \
      --tools Agent --disallowedTools "Agent(general-purpose)" ... \
      --permission-mode dontAsk --no-session-persistence \
      --output-format stream-json --verbose --forward-subagent-text

run in a fresh temp dir outside any repo. Auth is the normal subscription login (macOS
keychain); a fresh CLAUDE_CONFIG_DIR loses it, and --bare / --safe-mode are unusable
(--bare ignores OAuth, --safe-mode drops --agents).

Routing: a subagent's `tools: ["Agent(x)"]` grants the Agent tool but does NOT restrict
the subagent type, and `--allowedTools "Agent(x)"` does not either; only deny rules
(`--disallowedTools "Agent(x)"`, session-wide) are enforced. So we deny every agent type
that no role may call, give each role the Agent tool only if it must delegate, steer the
route by prompt, and `check_run` flags any delegation outside the topology as invalid.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import formats, topologies
from .runner_api import ENDPOINT, ENDPOINT_INSTRUCTIONS
from .schema import Condition, Message, MessageLog, Scenario, Usage

AGENT_TOOLS = {"Agent", "Task"}  # the init event lists the tool as "Task", tool_use says "Agent"
BUILTIN_AGENTS = ["general-purpose", "Explore", "Plan", "statusline-setup", "output-style-setup", "claude"]
ISOLATION_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",  # subagents run in the foreground
    "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",  # no general-purpose / Explore / Plan
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
# Allowlist, not a denylist: provider switches (CLAUDE_CODE_USE_BEDROCK/VERTEX/FOUNDRY), ANTHROPIC_*,
# AWS_*/GOOGLE_* credentials, custom headers and proxies would silently move billing off the subscription.
ENV_ALLOW = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "SHELL")
FIRST_PARTY_MODEL = re.compile(r"^claude-[a-z0-9-]+$")  # Bedrock/Vertex/Foundry ids look different
_ASYNC_PREFIX = "Async agent launched"
_FENCE = re.compile(r"^```[\w-]*\n(.*?)\n```\s*$", re.S)


# --------------------------------------------------------------------------- routing

def routes(topology: str, roles: list[str]) -> dict[str, list[str]]:
    """Allowed delegations sender -> [receivers] for the Claude Code runner."""
    if topology == "hier_chain":
        topologies.hier_chain(roles)
        return {a: [b] for a, b in zip(roles, roles[1:])}
    if topology == "hub_relay":
        topologies.hub_relay(roles)
        return {roles[0]: [roles[1], roles[-1]]}
    raise ValueError(f"topology {topology!r} is not run with Claude Code (use the API harness)")


def _role_text(scenario: Scenario, role_id: str) -> str:
    role = scenario.role(role_id)
    return f"You are the {role.title} in a team working for a client.\n\n{role.instructions}"


def _handoff(fmt: str) -> str:
    text = formats.instruction(fmt)
    if fmt == "forward":  # no harness in the loop: the agent copies the brief itself
        text = (f"{text.split(' Do not')[0]} Then add a line `{formats.ATTACH_MARKER}` followed by the "
                "original client brief exactly as you received it, character for character.")
    return text


def _delegate(target: str, fmt: str) -> str:
    return (f"Delegate exactly once to {target}: Agent tool, subagent_type \"{target}\". Do not use any other agent "
            f"for this step and do not do the work yourself. The Agent prompt is your hand-off message. {_handoff(fmt)}")


def prompts(scenario: Scenario, condition: Condition) -> dict[str, str]:
    """System prompt per role (the lead's goes to --system-prompt)."""
    roles = [r.id for r in scenario.roles]
    lead, terminal = roles[0], roles[-1]
    endpoint, _ = ENDPOINT_INSTRUCTIONS[scenario.domain]
    out = {}
    if condition.topology == "hier_chain":
        for a, b in zip(roles, roles[1:]):
            out[a] = (f"{_role_text(scenario, a)}\n\n{_delegate(b, condition.handoff_format)} "
                      "When it answers, reply with its answer verbatim.")
    else:
        r1, r2 = roles[1], roles[-1]
        out[lead] = (f"{_role_text(scenario, lead)}\n\nStep 1: {_delegate(r1, condition.handoff_format)}\n"
                     f"Step 2: when {r1} answers, write a new hand-off for {r2} based on what {r1} sent you and "
                     f"delegate exactly once with subagent_type \"{r2}\". {_handoff(condition.handoff_format)} "
                     f"Never skip {r1}. When {r2} answers, reply with its answer verbatim.")
        out[r1] = (f"{_role_text(scenario, r1)}\n\nYou have no tools. Reply with your hand-off for the next team "
                   f"member. {_handoff(condition.handoff_format)}")
    out[terminal] = f"{_role_text(scenario, terminal)}\n\nYou have no tools. {endpoint}"
    return out


def agents_json(scenario: Scenario, condition: Condition, models: dict[str, str] | None = None) -> dict:
    roles = [r.id for r in scenario.roles]
    allowed, sys = routes(condition.topology, roles), prompts(scenario, condition)
    return {r: {"description": scenario.role(r).title, "prompt": sys[r],
                "tools": [f"Agent({allowed[r][0]})"] if r in allowed else [],
                "model": (models or {}).get(r, condition.model)}
            for r in _used_roles(condition.topology, roles)[1:]}


def _used_roles(topology: str, roles: list[str]) -> list[str]:
    return roles if topology == "hier_chain" else [roles[0], roles[1], roles[-1]]


def build_command(scenario: Scenario, condition: Condition, models: dict[str, str] | None = None,
                  claude_bin: str = "claude") -> list[str]:
    roles = [r.id for r in scenario.roles]
    called = {b for bs in routes(condition.topology, roles).values() for b in bs}
    denied = [f"Agent({a})" for a in [*BUILTIN_AGENTS, *(r for r in roles if r not in called)]]
    return [claude_bin, "-p", scenario.source_brief(condition.noise),
            "--model", (models or {}).get(roles[0], condition.model),
            "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--disable-slash-commands", "--system-prompt", prompts(scenario, condition)[roles[0]],
            "--agents", json.dumps(agents_json(scenario, condition, models)),
            "--tools", "Agent", "--disallowedTools", *denied,
            "--permission-mode", "dontAsk", "--no-session-persistence",
            "--output-format", "stream-json", "--verbose", "--forward-subagent-text"]


def isolated_env(base: dict[str, str] | None = None, oauth_token_file: str | Path | None = None) -> dict[str, str]:
    """Environment for an isolated subscription run, built from an allowlist. With `oauth_token_file`
    (output of `claude setup-token`, never printed) the session also gets a fresh CLAUDE_CONFIG_DIR."""
    base = os.environ if base is None else base
    env = {k: base[k] for k in ENV_ALLOW if k in base} | ISOLATION_ENV
    if oauth_token_file:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = Path(oauth_token_file).read_text(encoding="utf-8").strip()
        env["CLAUDE_CONFIG_DIR"] = tempfile.mkdtemp(prefix="agent-telephone-cfg-")
    return env


# --------------------------------------------------------------------------- parsing

@dataclass
class Hop:
    """One reconstructed message: `delegate` (Agent prompt), `return` (hand-back) or `final`."""
    kind: str
    sender: str
    receiver: str
    distance: int
    text: str
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    tool_use_id: str | None = None
    direction: str = "down"  # delegate = down; return / final = up
    info_distance: int = 0   # 1 + the closest distance to the source the writer had received


@dataclass
class _Call:
    id: str
    owner: str | None  # context (parent tool_use id) that made the call; None = lead
    target: str
    prompt: str
    line: int
    model: str
    msg_id: str
    recv_line: int | None = None
    result_text: str = ""
    background: bool = False


def _usage(u: dict | None) -> Usage:
    u = u or {}
    return Usage(input_tokens=u.get("input_tokens") or 0, output_tokens=u.get("output_tokens") or 0,
                 cache_write_tokens=u.get("cache_creation_input_tokens") or 0,
                 cache_read_tokens=u.get("cache_read_input_tokens") or 0)


def _blocks(ev: dict) -> list[dict]:
    content = (ev.get("message") or {}).get("content")
    return content if isinstance(content, list) else []


def _split(total: int, floors: list[int], weights: list[int]) -> list[int]:
    """Split `total` into parts >= floors, the excess proportional to weights (largest remainder)."""
    extra = max(0, total - sum(floors))
    w = [max(1, x) for x in weights]
    raw = [extra * x / sum(w) for x in w]
    parts = [int(r) for r in raw]
    for i in sorted(range(len(w)), key=lambda i: parts[i] - raw[i])[: extra - sum(parts)]:
        parts[i] += 1
    return [f + p for f, p in zip(floors, parts)]


def turn_usage(events: list[dict]) -> tuple[dict[str, Usage], dict[str | None, list[str]]]:
    """Usage per API message (turn) id, and the turn ids of each context in stream order.

    Assistant events are streamed snapshots carrying the message_start usage: input and cache are
    right, output_tokens is a stub (2-16). The real output of a turn comes from:
    1. a subagent's final turn: `tool_use_result.usage` on the caller's tool_result (exact API usage;
       only for subagents called by the lead), else its `task_notification` `usage.total_tokens`
       (= that turn's input + output; agreed with the API usage in 71/75 pilot cross-checks, within
       ~15% otherwise);
    2. the lead's turns: `result.usage` is their sum;
    3. any other turn (a subagent's delegating turn): the per-model remainder of
       `result.modelUsage` once 1. and 2. are subtracted (`task_progress` fires when the Agent call
       starts, so it only gives a lower bound). Totals are used only when their input tokens match
       the stream's turns exactly (every API call is in the stream).
    A total shared by one open turn is exact; several split it above their floors by visible content
    size (`estimated=True`), as do turns no total covers.
    """
    snap: dict[str, Usage] = {}
    turns: dict[str | None, list[str]] = {}
    seen_blocks: dict[str, set[str]] = {}
    weight: dict[str, int] = {}
    by_desc: dict[tuple[str | None, str], str] = {}
    progress: list[tuple[str, str, int]] = []
    notified: dict[str, int] = {}
    reported: dict[str, Usage] = {}
    model: dict[str, str] = {}
    result = None
    for ev in events:
        t, st = ev.get("type"), ev.get("subtype")
        if t == "assistant":
            msg, ctx = ev["message"], ev.get("parent_tool_use_id")
            mid, u = msg.get("id"), _usage(msg.get("usage"))
            if mid not in snap:
                turns.setdefault(ctx, []).append(mid)
                snap[mid], seen_blocks[mid], weight[mid], model[mid] = u, set(), 0, msg.get("model", "")
            snap[mid] = Usage(**{k: max(getattr(u, k), getattr(snap[mid], k)) for k in
                                 ("input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens")})
            for b in _blocks(ev):  # snapshots may repeat earlier blocks: count each once
                key = json.dumps(b, sort_keys=True)
                if key not in seen_blocks[mid]:
                    seen_blocks[mid].add(key)
                    weight[mid] += len(b.get("text") or b.get("thinking") or "") + (
                        len(json.dumps(b.get("input"), ensure_ascii=False)) if b.get("type") == "tool_use" else 0)
                if b.get("type") == "tool_use":
                    by_desc[(ctx, (b.get("input") or {}).get("description") or "")] = mid
        elif t == "system" and st in ("task_progress", "task_notification") and ev.get("tool_use_id"):
            total = (ev.get("usage") or {}).get("total_tokens")
            if isinstance(total, int):
                if st == "task_progress":
                    progress.append((ev["tool_use_id"], ev.get("description") or "", total))
                else:
                    notified[ev["tool_use_id"]] = total
        elif t == "user" and isinstance(ev.get("tool_use_result"), dict) and ev["tool_use_result"].get("usage"):
            for b in _blocks(ev):
                if b.get("type") == "tool_result" and b.get("tool_use_id"):
                    reported[b["tool_use_id"]] = _usage(ev["tool_use_result"]["usage"])
        elif t == "result":
            result = ev

    floor = {mid: u.output_tokens for mid, u in snap.items()}
    for ctx, desc, total in progress:  # emitted when the call starts: a lower bound for that turn
        if (mid := by_desc.get((ctx, desc))) is not None:
            floor[mid] = max(floor[mid], total - snap[mid].total_input)
    exact: dict[str, int] = {}
    for ctx, total in notified.items():
        if turns.get(ctx):
            exact[turns[ctx][-1]] = total - snap[turns[ctx][-1]].total_input
    for ctx, u in reported.items():
        if turns.get(ctx) and u.total_input == snap[turns[ctx][-1]].total_input:
            exact[turns[ctx][-1]] = u.output_tokens
    usage = {mid: snap[mid].model_copy(update={"output_tokens": max(o, floor[mid])}) for mid, o in exact.items()}
    turn_ids = list(snap)  # stream order

    def settle(group: set[str], total: int) -> None:
        """Give the turns of `group` not yet resolved what is left of `total`."""
        if not (open_ := [m for m in turn_ids if m in group and m not in usage]):
            return
        parts = _split(total - sum(usage[m].output_tokens for m in group if m in usage), [floor[m] for m in open_],
                       [weight[m] for m in open_])
        for m, p in zip(open_, parts):
            usage[m] = snap[m].model_copy(update={"output_tokens": p, "estimated": len(open_) > 1})

    result = result or {}
    lead, ru = turns.get(None, []), _usage(result.get("usage"))
    if lead and ru.total_input == sum(snap[m].total_input for m in lead):  # result.usage = the lead's turns
        settle(set(lead), ru.output_tokens)
    for name, mu in (result.get("modelUsage") or {}).items():  # the rest: per-model remainder
        group = {m for m in snap if model[m] == name}
        if sum(mu.get(k) or 0 for k in ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens")) \
                == sum(snap[m].total_input for m in group):
            settle(group, mu.get("outputTokens") or 0)
    for mid in snap.keys() - usage.keys():
        usage[mid] = snap[mid].model_copy(update={"output_tokens": floor[mid], "estimated": True})
    return usage, turns


def _result_text(block: dict) -> str:
    c = block.get("content")
    return c if isinstance(c, str) else "\n".join(b.get("text", "") for b in c or [] if b.get("type") == "text")


def unwrap_handback(text: str) -> str:
    """Strip the harness frame Claude Code puts around a subagent's report in the tool_result."""
    if "The report follows:\n" in text:
        text = text.split("The report follows:\n", 1)[1]
        text = re.split(r"\nagentId: ", text, maxsplit=1)[0]
        text = "\n".join(line[2:] if line.startswith("  ") else line for line in text.splitlines())
    return text.strip()


def read_stream(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def parse_stream(events: list[dict], user_prompt: str, lead: str = "lead") -> list[Hop]:
    """Rebuild the message graph of one run.

    Edges come from Agent tool_use (sender -> subagent, text = input.prompt) and from the
    subagent's final assistant message (subagent -> caller). Each message's distance is
    1 + the distance of the last message its sender had received when it sent it, where
    "received" is placed at the caller's tool_result (foreground) or at the
    task_notification (background), so background runs are ordered causally too.
    Usage (see `turn_usage`): a delegation carries the turn that wrote the Agent call, a hand-back /
    final the sum of the writer's other turns in that context.
    """
    calls: dict[str, _Call] = {}
    finals: dict[str | None, tuple[str, list[str], dict]] = {}  # context -> (msg id, texts, msg)
    notified: dict[str, tuple[int, str]] = {}
    usage, turns = turn_usage(events)
    for i, ev in enumerate(events):
        ctx = ev.get("parent_tool_use_id")
        if ev.get("type") == "system" and ev.get("subtype") == "task_started" and ev.get("is_backgrounded"):
            if ev.get("tool_use_id") in calls:
                calls[ev["tool_use_id"]].background = True
        elif ev.get("type") == "system" and ev.get("subtype") == "task_notification" and ev.get("tool_use_id"):
            notified.setdefault(ev["tool_use_id"], (i, ev.get("summary") or ""))
        elif ev.get("type") == "assistant":
            msg = ev["message"]
            for b in _blocks(ev):
                if b.get("type") == "tool_use" and b.get("name") in AGENT_TOOLS:
                    inp = b.get("input") or {}
                    calls[b["id"]] = _Call(b["id"], ctx, inp.get("subagent_type") or "general-purpose",
                                           inp.get("prompt", ""), i, msg.get("model", ""), msg.get("id"))
                elif b.get("type") == "text" and b.get("text", "").strip():
                    mid, texts, _ = finals.get(ctx, (None, [], {}))
                    if mid != msg.get("id"):
                        texts = []
                    finals[ctx] = (msg.get("id"), [*texts, b["text"]], msg)
        elif ev.get("type") == "user":
            for b in _blocks(ev):
                c = calls.get(b.get("tool_use_id", ""))
                if b.get("type") == "tool_result" and c and c.recv_line is None:
                    text = _result_text(b)
                    if text.startswith(_ASYNC_PREFIX):
                        c.background = True
                    else:
                        c.recv_line, c.result_text = i, unwrap_handback(text)

    for c in calls.values():
        if c.recv_line is None and c.id in notified:
            c.recv_line, c.result_text = notified[c.id]
        if c.recv_line is None:  # never handed back (killed / still running): place it at the end
            c.recv_line = len(events)

    def name(ctx: str | None) -> str:
        return lead if ctx is None else calls[ctx].target

    # usage: a delegation carries the turn that wrote its Agent call (once per turn, even with parallel
    # calls); a hand-back / final carries every other turn of the writer's context
    delegating = {c.msg_id for c in calls.values()}
    first_call = {}
    for c in sorted(calls.values(), key=lambda c: c.line):
        first_call.setdefault(c.msg_id, c.id)

    def written(ctx: str | None) -> Usage:
        return sum((usage[m] for m in turns.get(ctx, []) if m not in delegating), Usage())

    hops = [Hop("delegate", topologies.USER, lead, 1, user_prompt, "human", info_distance=1)]
    last: dict[str | None, int] = {None: 1}
    closest: dict[str | None, int] = {None: 1}
    timeline = sorted([(c.line, 1, c.id) for c in calls.values()] + [(c.recv_line, 0, c.id) for c in calls.values()])
    for _, is_send, cid in timeline:
        c = calls[cid]
        if is_send:
            d, info = last.get(c.owner, 1) + 1, closest.get(c.owner, 1) + 1
            last[cid], closest[cid] = d, info
            u = usage.get(c.msg_id, Usage()) if first_call[c.msg_id] == cid else Usage()
            hops.append(Hop("delegate", name(c.owner), c.target, d, c.prompt, c.model, u, cid, "down", info))
        else:
            d, info = last.get(cid, 1) + 1, closest.get(cid, 1) + 1
            last[c.owner], closest[c.owner] = d, min(closest.get(c.owner, info), info)
            _, texts, msg = finals.get(cid, (None, [], {}))
            text = "\n\n".join(texts) if texts else c.result_text
            hops.append(Hop("return", c.target, name(c.owner), d, text, msg.get("model", ""), written(cid), cid, "up", info))
    if None in finals:
        _, texts, msg = finals[None]
        hops.append(Hop("final", lead, topologies.USER, last[None] + 1, "\n\n".join(texts),
                        msg.get("model", ""), written(None), None, "up", closest[None] + 1))
    return hops


def extract_artifact(text: str) -> str:
    """The endpoint artifact from a final message: the body of a single fenced block, else the text."""
    text = text.strip()
    m = _FENCE.match(text)
    if m:
        return m.group(1)
    blocks = re.findall(r"```[\w-]*\n(.*?)\n```", text, re.S)
    return blocks[0] if len(blocks) == 1 else text


# --------------------------------------------------------------------------- validity

def check_run(events: list[dict], hops: list[Hop], allowed: dict[str, list[str]]) -> list[str]:
    """Reasons the run is invalid (empty list = valid)."""
    problems = []
    for ev in events:
        st = ev.get("subtype")
        if ev.get("type") == "system" and st in ("hook_started", "hook_response"):
            problems.append(f"hook ran: {ev.get('hook_name')}")
        elif ev.get("type") == "system" and st == "init":
            if ev.get("apiKeySource") != "none":
                problems.append(f"not billed to the subscription: apiKeySource={ev.get('apiKeySource')!r}")
            if not FIRST_PARTY_MODEL.match(str(ev.get("model", ""))):
                problems.append(f"unexpected model id/provider: {ev.get('model')!r}")
            if ev.get("mcp_servers"):
                problems.append(f"MCP servers loaded: {[s.get('name') for s in ev['mcp_servers']]}")
            user_plugins = [p["name"] for p in ev.get("plugins") or [] if not str(p.get("source", "")).endswith("@builtin")]
            if user_plugins:
                problems.append(f"user plugins loaded: {user_plugins}")
            if ev.get("memory_paths"):
                problems.append("auto-memory enabled")
            if set(ev.get("tools") or []) - AGENT_TOOLS:
                problems.append(f"lead has extra tools: {sorted(set(ev['tools']) - AGENT_TOOLS)}")
        elif ev.get("type") == "system" and st == "task_started" and ev.get("is_backgrounded"):
            problems.append(f"background subagent: {ev.get('subagent_type')}")
        elif ev.get("type") == "assistant":
            if not FIRST_PARTY_MODEL.match(str(ev["message"].get("model", ""))):
                problems.append(f"unexpected model id/provider: {ev['message'].get('model')!r}")
            for b in _blocks(ev):
                if b.get("type") == "tool_use" and b.get("name") not in AGENT_TOOLS:
                    problems.append(f"non-delegation tool used: {b.get('name')}")
        elif ev.get("type") == "result" and ev.get("is_error"):
            problems.append(f"run ended with error: {ev.get('subtype')}")
    seen: dict[tuple[str, str], int] = {}
    for h in hops:
        if h.kind == "delegate" and h.sender != topologies.USER:
            seen[h.sender, h.receiver] = seen.get((h.sender, h.receiver), 0) + 1
            if h.receiver not in allowed.get(h.sender, []):
                problems.append(f"unexpected delegation {h.sender} -> {h.receiver}")
    for a, bs in allowed.items():
        for b in bs:
            if seen.get((a, b), 0) != 1:
                problems.append(f"delegation {a} -> {b} happened {seen.get((a, b), 0)} times (expected 1)")
    return list(dict.fromkeys(problems))


# --------------------------------------------------------------------------- run

@dataclass
class CCRunResult:
    run_id: str
    messages: list[Message]
    artifact_path: Path | None
    stream_path: Path
    problems: list[str]

    @property
    def valid(self) -> bool:
        return not self.problems


def _assert_outside_repos(path: Path) -> None:
    for p in [path, *path.parents]:
        if (p / ".git").exists() or (p / "CLAUDE.md").exists() or ((p / ".claude").is_dir() and p != Path.home()):
            raise RuntimeError(f"refusing to run claude inside {p} (repo or Claude config found)")


def to_messages(hops: list[Hop], run_id: str, scenario: Scenario, condition: Condition) -> list[Message]:
    return [Message(run_id=run_id, scenario=scenario.id, condition=condition, edge=(h.sender, h.receiver),
                    distance=h.distance, sender=h.sender, receiver=h.receiver, text=h.text,
                    model=h.model, usage=h.usage, direction=h.direction, info_distance=h.info_distance) for h in hops]


def record(events: list[dict], scenario: Scenario, condition: Condition, log_path: str | Path, run_id: str,
           stream_path: Path) -> CCRunResult:
    """Parse a finished run's stream, append its rows to the log and write the artifact."""
    roles = [r.id for r in scenario.roles]
    allowed = routes(condition.topology, roles)
    terminal = roles[-1]
    hops = parse_stream(events, scenario.source_brief(condition.noise), lead=roles[0])
    problems = check_run(events, hops, allowed)
    ret = next((h for h in reversed(hops) if h.kind == "return" and h.sender == terminal), None)
    artifact_path = None
    if ret is None:
        problems.append(f"no hand-back from terminal role {terminal}")
    else:  # the terminal role's hand-back *is* the artifact; logged once more as [terminal, endpoint]
        artifact = extract_artifact(ret.text)
        hops.append(Hop("endpoint", terminal, ENDPOINT, ret.distance, artifact, ret.model, Usage(), ret.tool_use_id,
                        "down", ret.info_distance))
        artifact_path = Path(log_path).with_name(f"{run_id}.artifact.{ENDPOINT_INSTRUCTIONS[scenario.domain][1]}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(artifact, encoding="utf-8")
    messages = to_messages(hops, run_id, scenario, condition)
    log = MessageLog(log_path)
    for m in messages:
        log.append(m)
    return CCRunResult(run_id, messages, artifact_path, stream_path, problems)


def run(scenario: Scenario, condition: Condition, log_path: str | Path, models: dict[str, str] | None = None,
        run_id: str | None = None, workdir: str | Path | None = None, timeout: float = 1800,
        claude_bin: str = "claude", oauth_token_file: str | Path | None = None,
        runner=subprocess.run) -> CCRunResult:
    run_id = run_id or uuid.uuid4().hex[:12]
    wd = Path(workdir or tempfile.mkdtemp(prefix="agent-telephone-cc-")).resolve()
    _assert_outside_repos(wd)
    stream_path = Path(log_path).with_name(f"{run_id}.stream.jsonl")
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(scenario, condition, models, claude_bin)
    with stream_path.open("w", encoding="utf-8") as out:
        proc = runner(cmd, cwd=wd, env=isolated_env(oauth_token_file=oauth_token_file), stdin=subprocess.DEVNULL,
                      stdout=out, stderr=subprocess.PIPE, text=True, timeout=timeout)
    res = record(read_stream(stream_path), scenario, condition, log_path, run_id, stream_path)
    if proc.returncode:
        res.problems.append(f"claude exited {proc.returncode}: {(proc.stderr or '')[-500:]}")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agent_telephone.runner_cc")
    ap.add_argument("--scenario", required=True, help="scenario dir or scenario.yaml")
    ap.add_argument("--topology", choices=["hier_chain", "hub_relay"], default="hier_chain")
    ap.add_argument("--model", default="haiku", help="model for every role (alias or full id)")
    ap.add_argument("--role-model", action="append", default=[], metavar="ROLE=MODEL")
    ap.add_argument("--format", dest="handoff_format", choices=list(formats.INSTRUCTIONS), default="free")
    ap.add_argument("--noise", default="clean")
    ap.add_argument("--log", default="runs/cc_messages.jsonl")
    ap.add_argument("--oauth-token-file", help="file with a `claude setup-token` token (fallback auth)")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args(argv)

    scenario = Scenario.load(args.scenario)
    cond = Condition(topology=args.topology, model=args.model, handoff_format=args.handoff_format, noise=args.noise)
    models = dict(kv.split("=", 1) for kv in args.role_model)
    if args.dry_run:
        print(json.dumps(build_command(scenario, cond, models), indent=1))
        return
    res = run(scenario, cond, args.log, models, oauth_token_file=args.oauth_token_file)
    print(f"run {res.run_id}: {len(res.messages)} messages -> {args.log}; artifact {res.artifact_path}; "
          f"{'VALID' if res.valid else 'INVALID: ' + '; '.join(res.problems)}")


if __name__ == "__main__":
    main()
