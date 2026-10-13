"""API harness: walk a topology, one model call per hand-off, log every message.

Each node gets its role as system prompt, everything it has received so far (in
<received> blocks, latest last) and the hand-off format instruction. The receiver of
the last edge then produces the endpoint artifact, logged as edge [node, "endpoint"]
at distance last+1 and written next to the log.
"""
from __future__ import annotations

import argparse
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import formats, topologies
from .backends.base import Backend
from .schema import Condition, Message, MessageLog, Scenario, Usage

ENDPOINT = "endpoint"
ENDPOINT_INSTRUCTIONS = {
    "software": ("Produce the final implementation as a single Python module. "
                 "Reply with the code only, no prose and no markdown fences.", "py"),
    "event": ("Produce the final event plan as a single JSON object. "
              "Reply with the JSON only, no prose and no markdown fences.", "json"),
}


@dataclass
class RunResult:
    run_id: str
    messages: list[Message]
    artifact_path: Path


def _system(scenario: Scenario, role_id: str) -> str:
    role = scenario.role(role_id)
    return f"You are the {role.title} in a team working for a client.\n\n{role.instructions}"


def _addressee(scenario: Scenario, e: topologies.Edge) -> str:
    """Who the message is for, as Claude Code's Agent call tells the sender (parity with runner_cc)."""
    title = scenario.role(e.receiver).title
    if e.direction == "up":
        return f"Your reply goes back to the {title} who delegated this step to you. "
    return f"Your message goes to the {title} ({e.receiver}), who only sees your message. "


def _prompt(inbox: list[tuple[str, str]], instruction: str) -> str:
    blocks = "\n\n".join(f'<received from="{s}">\n{t}\n</received>' for s, t in inbox)
    return f"{blocks}\n\n<instruction>\n{instruction}\n</instruction>"


def run(scenario: Scenario, condition: Condition, backend: Backend, log_path: str | Path,
        max_tokens: int = 4096, run_id: str | None = None) -> RunResult:
    run_id = run_id or uuid.uuid4().hex[:12]
    log = MessageLog(log_path)
    original = scenario.source_brief(condition.noise)
    edges = topologies.build(condition.topology, [r.id for r in scenario.roles])
    inbox: dict[str, list[tuple[str, str]]] = {}
    messages: list[Message] = []

    def emit(e: topologies.Edge, text: str, usage: Usage, model: str) -> None:
        sender, receiver = e.sender, e.receiver
        msg = Message(run_id=run_id, scenario=scenario.id, condition=condition, edge=(sender, receiver),
                      distance=e.distance, sender=sender, receiver=receiver, text=text, model=model, usage=usage,
                      direction=e.direction, info_distance=e.info_distance)
        log.append(msg)
        messages.append(msg)
        inbox.setdefault(receiver, []).append((sender, text))

    def ask(node: str, instruction: str):
        return backend.complete(_system(scenario, node), [{"role": "user", "content": _prompt(inbox[node], instruction)}],
                                condition.model, max_tokens)

    for e in edges:
        if e.sender == topologies.USER:
            emit(e, original, Usage(), "human")
            continue
        c = ask(e.sender, _addressee(scenario, e) + formats.instruction(condition.handoff_format))
        emit(e, formats.compose(condition.handoff_format, c.text, original), c.usage, condition.model)

    final, (instruction, ext) = edges[-1].receiver, ENDPOINT_INSTRUCTIONS[scenario.domain]
    c = ask(final, instruction)
    emit(topologies.Edge(final, ENDPOINT, edges[-1].distance + 1, "down", topologies.min_info_distance(edges, final)),
         c.text, c.usage, condition.model)
    artifact = Path(log_path).with_name(f"{run_id}.artifact.{ext}")
    artifact.write_text(c.text, encoding="utf-8")
    return RunResult(run_id, messages, artifact)


def _backend(args, scenario: Scenario) -> Backend:
    if args.backend == "fake":
        from .backends.fake import FakeBackend, FakeConfig

        cfg = FakeConfig.model_validate(yaml.safe_load(Path(args.fake_config).read_text())) if args.fake_config else None
        return FakeBackend(scenario, cfg, seed=args.seed)
    from .budget import DEFAULT_LEDGER, BudgetGuard

    args.ledger = args.ledger or DEFAULT_LEDGER
    guard = BudgetGuard(args.budget_limit, args.ledger, args.prices, init_ledger=args.init_ledger)
    if args.backend == "gateway":
        from .backends.gateway import GatewayBackend

        return GatewayBackend(args.gateway_url, args.key_file, guard, reasoning_effort=args.reasoning_effort)
    from .backends.anthropic_api import AnthropicBackend

    return AnthropicBackend(guard, args.key_file)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agent_telephone.runner_api")
    ap.add_argument("--scenario", required=True, help="scenario dir or scenario.yaml")
    ap.add_argument("--backend", choices=["fake", "gateway", "anthropic"], default="fake")
    ap.add_argument("--topology", choices=list(topologies.TOPOLOGIES), default="hier_chain")
    ap.add_argument("--model", default="fake")
    ap.add_argument("--format", dest="handoff_format", choices=list(formats.INSTRUCTIONS), default="free")
    ap.add_argument("--noise", default="clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="runs/messages.jsonl")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--fake-config", help="YAML with a `distortions:` list (fake backend)")
    ap.add_argument("--gateway-url", default=os.environ.get("AGENT_TELEPHONE_GATEWAY_URL", ""))
    ap.add_argument("--reasoning-effort", help="sent to the gateway for reasoning models")
    ap.add_argument("--key-file", help="file holding the API key (never printed)")
    ap.add_argument("--budget-limit", default="400", help="USD; at most budget.HARD_CAP_USD")
    ap.add_argument("--ledger", default=None, help="budget ledger (default: one absolute path outside the repo)")
    ap.add_argument("--init-ledger", action="store_true", help="create a NEW ledger if none exists (fresh budget)")
    ap.add_argument("--prices", default=str(Path(__file__).resolve().parent.parent / "config" / "prices.yaml"))
    args = ap.parse_args(argv)

    scenario = Scenario.load(args.scenario)
    cond = Condition(topology=args.topology, model=args.model, handoff_format=args.handoff_format,
                     noise=args.noise, seed=args.seed)
    res = run(scenario, cond, _backend(args, scenario), args.log, args.max_tokens)
    print(f"run {res.run_id}: {len(res.messages)} messages -> {args.log}; artifact {res.artifact_path}")


if __name__ == "__main__":
    main()
