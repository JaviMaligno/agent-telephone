# Running Claude Code isolated on the subscription

Verified 2026-10-12, Claude Code 2.1.296, model `haiku` (`claude-haiku-5-5`), six tiny
`claude -p` calls, each in a fresh directory under `$TMPDIR` (never in a repo).
Implemented by `agent_telephone/runner_cc.py` (`build_command`, `isolated_env`).

## Working invocation

```bash
cd "$(mktemp -d)"   # fresh dir, outside any repo
env -u ANTHROPIC_API_KEY \
  CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 \
  CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=1 \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude -p "<brief>" --model haiku \
  --setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' \
  --disable-slash-commands --system-prompt "<lead role prompt>" \
  --agents '<{role: {description, prompt, tools, model}}>' \
  --tools Agent --disallowedTools "Agent(general-purpose)" "Agent(Explore)" ... \
  --permission-mode dontAsk --no-session-persistence \
  --output-format stream-json --verbose --forward-subagent-text < /dev/null
```

Auth is the normal subscription login read from the macOS keychain (`apiKeySource: "none"`
in the init event). No API key is used; `isolated_env` removes `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL` and `CLAUDE_CONFIG_DIR`.

## Evidence

| Check | Contaminated capture (`fixtures/cc_nested_stream.jsonl`) | Isolated (`fixtures/cc_isolated_*.jsonl`) |
|---|---|---|
| Reply language (English question) | Spanish | English ("I'm replying in English.") |
| `hook_started` / `hook_response` events | 8 (SessionStart) | 0 |
| init `mcp_servers` | 12 (user, plugin, claude.ai) | `[]` |
| init `plugins` | 13 user plugins + builtins | only `cc-plugin-agents-md@builtin`, `cc-plugin-plugin-authoring@builtin` |
| init `memory_paths` | auto-memory dir set | absent |
| init `tools` (lead) | 264 | `["Task"]` (= Agent) |
| init `agents` | 15, incl. user-level `pm`/`implementer` and plugin agents | only the `--agents` roles |
| subagents in background | 2 of 2 (`is_backgrounded: true`) | 0 (`started_in_background: 0`) |
| files written by agents | `greeting.py`, `test_greeting.py` in the repo | none; workdir left empty |

`check_run` turns each row of this table into an invalidity check.

## What did not work

- **`--bare`**: ignores OAuth/keychain (API key only), so it can't use the subscription.
- **`--safe-mode`**: isolates fully (English, no hooks/MCP/plugins), but also drops the
  `--agents` definitions: `Agent type 'pm' not found. Available agents: none`.
- **Fresh `CLAUDE_CONFIG_DIR`**: `claude auth status` reports `loggedIn: false` because
  the keychain entry is tied to the config dir. The fallback is to run `claude setup-token`
  once, store the token in a file and pass `--oauth-token-file` (`runner_cc` sets
  `CLAUDE_CODE_OAUTH_TOKEN` and a fresh `CLAUDE_CONFIG_DIR`; the token is never printed).
  This path is implemented but **not exercised** (not needed, the default config dir with
  `--setting-sources ""` is clean).

Not separated (all applied together, so which flag removes which item is not attributed):
`--setting-sources ""` vs `CLAUDE_CODE_DISABLE_CLAUDE_MDS` for the global CLAUDE.md, and
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` as the cause of foreground subagents.

## Forced routing

| Mechanism | Result |
|---|---|
| subagent `tools: ["Agent(implementer)"]` | grants the Agent tool but **does not restrict the type**: the pm reached `reviewer` |
| `--allowedTools "Agent(pm)"` + `dontAsk` (lead) | **does not restrict**: the lead reached `reviewer` |
| `--disallowedTools "Agent(reviewer)"` (CLI) | **enforced** for the lead and inside subagents: `Agent type 'reviewer' has been denied by permission rule 'Agent(reviewer)' from cliArg.` |
| subagent `disallowedTools: ["Agent(reviewer)"]` | removes the **whole** Agent tool from that subagent ("would be spawned with zero tools — refusing") |
| subagent `tools: []` | no tools at all (the implementer reported NONE and made 0 tool calls) |
| `CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=1` | `general-purpose`, `Explore`, `Plan` are gone from the agent list |

So per-role allow-listing is not available. The runner:
1. denies session-wide every agent type that no role may call (built-ins, the lead's id);
2. gives the Agent tool only to roles that must delegate, `tools: []` to the rest
   (no file tools, so nothing touches disk; the endpoint is the terminal role's text);
3. states the route in each role's prompt;
4. marks the run invalid post hoc (`check_run`) if any delegation leaves the topology, an
   expected delegation is missing or repeated, a non-Agent tool is used, a subagent runs in
   the background, or any hook/MCP/plugin/memory shows up.

## Known side effects

- Subagents still see the environment block (cwd): in the hub_relay capture the architect
  wrote the temp workdir path into its hand-off. Harmless for scoring, but it is text the
  brief did not contain.
- Per-message `usage` in stream-json is a snapshot taken while streaming: `input_tokens`
  (incl. cache) is right, `output_tokens` is a lower bound. Run totals per model are in the
  final `result` event (`modelUsage`).
- A model with no tools can still *write* a fake tool call as text (seen once); it has no
  effect, but it ends up in the logged text.
