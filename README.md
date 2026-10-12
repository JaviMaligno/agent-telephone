# agent-telephone

Does a request survive being passed through a team of agents? A client brief goes
user → lead → role agents (PM, architect, implementer / account manager, logistics, vendor).
For every message and every atomic fact of the brief we measure whether the fact arrives
**preserved**, **lost**, **distorted** or **inverted**, count **invented** claims, and check
whether the final artifact still does what the client asked (hidden tests / plan checks).

The pre-registered design (hypotheses H1–H7, factors, scoring) is in [`DESIGN.md`](DESIGN.md).

## Layout

| Path | What |
|---|---|
| `agent_telephone/schema.py` | scenarios, facts, noise variants, conditions, the JSONL message log |
| `agent_telephone/topologies.py` | `hier_chain`, `hub_relay`, `peer_route` as edge lists (the last role always writes the endpoint) |
| `agent_telephone/runner_api.py` | API harness: one model call per hand-off over any `Backend` |
| `agent_telephone/runner_cc.py` | Claude Code runner (`claude -p --agents`, isolated; see `docs/claude_code_isolation.md`) |
| `agent_telephone/backends/` | `FakeBackend` / `FakeJudgeBackend` (Phase 0), LiteLLM gateway, Anthropic API (budget-guarded) |
| `agent_telephone/scoring/` | deterministic checks, cross-family LLM judge, agreement (Cohen's κ) |
| `agent_telephone/endpoint.py` | endpoint evaluation: hidden tests in a temp dir + subprocess, or JSON plan checks |
| `agent_telephone/analysis.py` | survival by distance / type / condition, invented claims, noise propagation classes |
| `agent_telephone/pilot.py` | end-to-end pilot grid → `runs/<backend>/summary.md` |
| `scenarios/software_pilot` | opticians' appointment reminders; 22 facts; hidden tests; reference solution |
| `scenarios/event_pilot` | partner dinner plan; 22 facts; JSON schema + plan checks; reference plan |
| `scenarios/software_2` | seat-upgrade proration (billing); 21 facts; hidden tests; reference solution |
| `scenarios/software_3` | GDPR account clean-up job; 19 facts; hidden tests; reference solution |
| `scenarios/event_2` | bilingual wedding (Spanish/Finnish families); 21 facts; JSON schema + plan checks |
| `scenarios/event_3` | one-day scientific symposium (speakers, AV); 21 facts; JSON schema + plan checks |
| `fixtures/` | captured Claude Code streams (one contaminated, two isolated) |
| `scripts/cc_pilot.py` | Phase 1 grid on Claude Code (subscription) → `runs/cc_pilot/` |
| `scripts/backfill_usage.py` | re-derive per-message usage of logged Claude Code runs from their streams |
| `scripts/score_real.py` | score real runs: deterministic + endpoints; judge behind `--judge gateway` (`--dry-run` = plan + cost) |

## Phases

0. **Fake backend, zero cost** — the whole pipeline is validated against injected ground truth. *(current)*
1. **Claude Code with the subscription** — `runner_cc`, Haiku / Sonnet, isolated temp dirs.
2. **API harness through the LiteLLM gateway** — gpt-6-luna.
3. **Anthropic API credits (optional)** — `BudgetGuard`, hard cap; every paid batch estimated and approved first.

## Running Phase 0

```bash
uv sync
uv run pytest -q                                      # full suite, no network
uv run python -m agent_telephone.pilot --backend fake # 48 runs -> runs/fake/summary.md
```

The pilot runs both scenarios × 2 topologies (`hier_chain`, `hub_relay`) × 3 hand-off formats
(`free`, `structured`, `forward`) × (clean + 3 noise variants) with a seeded `FakeTeam`, scores
every message deterministically plus with the fake judge in two views (`note` = what the agent
wrote, `full` = what the receiver saw), evaluates every endpoint artifact and writes
`summary.md`, score JSONL, endpoint CSVs and the per-view analysis CSVs under `runs/fake/`.
The numbers are synthetic: they check the plumbing, not the hypotheses.

Single pieces:

```bash
uv run python -m agent_telephone.runner_api --scenario scenarios/event_pilot --backend fake --topology hub_relay --format forward
uv run python -m agent_telephone.analysis --scenario scenarios/event_pilot --log runs/messages.jsonl --out runs/analysis
uv run python -m agent_telephone.runner_cc --scenario scenarios/software_pilot --topology hier_chain --model haiku --dry-run
```

## Status

- Phase 0 complete: deterministic scorer recovers the injected ground truth exactly (accuracy 1.000
  on the pilots); the fake judge reaches 0.967 (it cannot emit `distorted`); judge pass-to-pass κ is
  reported; every endpoint fact has a mutant that its check catches.
- Claude Code isolation and forced routing confirmed on real (subscription) runs; captures in `fixtures/`.
- H4 is scored on the endpoint (DESIGN §7); forward hand-offs are scored as contradiction.
- Paid backends need a price in `config/prices.yaml`, a gateway URL (`--gateway-url` or
  `AGENT_TELEPHONE_GATEWAY_URL`) and a budget ledger outside the repo (created once with `--init-ledger`).
- Deterministic scorer revised after the first real runs (DESIGN changelog, 2026-10-12): numbers before
  that entry are void; validate against the blind label sheet (`runs/cc_pilot/labels/`) before quoting H1/H7.
- Open before Phase 1: judge sampling for a genuine second pass; noise variants are hand-written (no TTS → Whisper yet); `peer_route` has no Claude Code runner.
