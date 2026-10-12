"""Data model: scenarios, conditions and the message log."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

FactType = Literal["name", "number", "negation", "conditional", "priority", "rationale", "low_salience"]


class Check(BaseModel):
    kind: Literal["exact", "regex", "fuzzy", "judge"]
    pattern: str | None = None
    aliases: list[str] = Field(default_factory=list)
    # Optional regex(es) with a named group `value` (or value2, value3... in alternations) locating
    # the fact's slot ("host is (?P<value>[^.]+)"). The expected value is read off the clean brief;
    # a list means independent parts (any order, any line) that must all carry their value.
    slot: str | list[str] | None = None
    # Slot values that belong to other facts (e.g. the PL signer in the default signer's slot).
    ignore: list[str] = Field(default_factory=list)
    # Also ask the judge: for facts whose polarity/behaviour the anchor cannot see.
    also_judge: bool = False


class Fact(BaseModel):
    id: str
    type: FactType
    text: str
    check: Check


class NoiseVariant(BaseModel):
    id: str
    kind: Literal["asr_name", "homophone", "punct"]
    brief: str  # full noisy brief that replaces the clean one
    affected_fact_ids: list[str]
    wrong_reading: str  # prose description, for humans
    # Machine-checkable wrong reading: regex over the number-normalised text (scoring.deterministic.canon).
    wrong_pattern: str
    # Regex for the context where the wrong reading legitimately sits; the wrong reading appearing
    # in a sentence without it has spread to another fact (amplify).
    wrong_context: str
    # False when nothing in the brief allows restoring the right reading (e.g. an ASR-mangled name).
    repairable: bool = True


class Role(BaseModel):
    id: str
    title: str
    instructions: str


class Endpoint(BaseModel):
    kind: Literal["hidden_tests", "plan_checks"]
    path: str
    # Facts every endpoint check depends on (e.g. the entry point): losing one fails everything.
    universal: list[str] = Field(default_factory=list)
    # fact -> facts its endpoint check also depends on (e.g. "CEO at the left end" needs the CEO seated).
    depends: dict[str, list[str]] = Field(default_factory=dict)
    # Facts whose endpoint check passes by default when the fact is lost (a natural artifact
    # satisfies a lost negation anyway): they cannot show H7 and are excluded from it.
    loss_insensitive: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    id: str
    domain: Literal["software", "event"]
    brief: str
    facts: list[Fact]
    roles: list[Role]
    noise: list[NoiseVariant] = Field(default_factory=list)
    endpoint: Endpoint

    @model_validator(mode="after")
    def _consistent(self) -> Scenario:
        ids = [f.id for f in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate fact ids")
        if len({r.id for r in self.roles}) != len(self.roles):
            raise ValueError("duplicate role ids")
        for n in self.noise:
            if missing := set(n.affected_fact_ids) - set(ids):
                raise ValueError(f"noise {n.id} references unknown facts {sorted(missing)}")
        ep = self.endpoint
        named = {*ep.universal, *ep.loss_insensitive, *ep.depends, *(f for fs in ep.depends.values() for f in fs)}
        if missing := named - set(ids):
            raise ValueError(f"endpoint references unknown facts {sorted(missing)}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        path = Path(path)
        if path.is_dir():
            path = path / "scenario.yaml"
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def fact(self, fact_id: str) -> Fact:
        return next(f for f in self.facts if f.id == fact_id)

    def role(self, role_id: str) -> Role:
        return next(r for r in self.roles if r.id == role_id)

    def source_brief(self, noise: str) -> str:
        """The original message as the lead receives it (clean or with a noise variant)."""
        if noise == "clean":
            return self.brief
        for n in self.noise:
            if n.id == noise:
                return n.brief
        raise ValueError(f"unknown noise variant {noise!r} for scenario {self.id}")


class Condition(BaseModel):
    topology: Literal["hier_chain", "hub_relay", "peer_route"]
    model: str
    handoff_format: Literal["free", "structured", "forward"]
    noise: str = "clean"
    seed: int = 0


class Usage(BaseModel):
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    # Output split heuristically rather than reported per turn (Claude Code lead turns, see runner_cc).
    estimated: bool = False

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(input_tokens=self.input_tokens + other.input_tokens,
                     output_tokens=self.output_tokens + other.output_tokens,
                     cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
                     cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
                     estimated=self.estimated or other.estimated)


class Message(BaseModel):
    run_id: str
    scenario: str
    condition: Condition
    edge: tuple[str, str]
    distance: int
    sender: str
    receiver: str
    text: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    # down = delegation/brief towards the endpoint; up = hand-back to a caller (None in old logs = down).
    direction: Literal["down", "up"] | None = None
    # 1 + the smallest distance at which the writer had received source-derived content; differs
    # from `distance` when the writer still holds a closer copy (the hub_relay lead holds the brief).
    info_distance: int | None = None


class MessageLog:
    """Append-only JSONL log of Message rows."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, msg: Message) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg.model_dump_json() + "\n")

    def read(self) -> list[Message]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [Message.model_validate_json(line) for line in lines if line.strip()]


def attempt_blocks(messages: list[Message]) -> dict[str, list[list[int]]]:
    """run id -> row indices per logged attempt (each starts with the brief). Run ids can repeat
    (re-runs append to the same log); the last block is the latest attempt."""
    out: dict[str, list[list[int]]] = {}
    for i, m in enumerate(messages):
        blocks = out.setdefault(m.run_id, [])
        if m.distance == 1 and m.direction != "up" or not blocks:
            blocks.append([])
        blocks[-1].append(i)
    return out


def latest_attempts(messages: list[Message]) -> list[Message]:
    return [messages[i] for blocks in attempt_blocks(messages).values() for i in blocks[-1]]
