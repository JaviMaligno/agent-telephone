"""FakeBackend: deterministic scripted hand-offs with known distortions (Phase 0).

Each call takes the latest incoming message (the last <received> block of the user turn,
minus any forwarded attachment), applies the configured distortions, and returns the
result. `applied[i]` lists the distortions actually applied on call i: the ground truth
that scoring is validated against.
"""
from __future__ import annotations

import random
import re
from typing import Literal

from pydantic import BaseModel, Field

from ..formats import strip_attachment
from ..schema import Fact, Scenario, Usage
from .base import Completion

RECEIVED_RE = re.compile(r"<received from=\"[^\"]*\">\n(.*?)\n</received>", re.S)
NEGATIONS = [(r"\bdo not\b", "do"), (r"\bdon't\b", "do"), (r"\bnever\b", "always"),
             (r"\bnot\b", ""), (r"\bno\b", "some"), (r"\bunless\b", "if")]


class Distortion(BaseModel):
    op: Literal["drop_fact", "swap_name", "flip_negation", "invent"]
    fact_id: str | None = None
    new: str | None = None   # swap_name replacement
    text: str | None = None  # invent: claim to add
    at: int | None = None    # call index to apply on; None = every call where applicable
    p: float = 1.0           # application probability (seeded)


class Applied(BaseModel):
    call: int
    op: str
    fact_id: str | None = None
    before: str | None = None
    after: str | None = None


class FakeConfig(BaseModel):
    distortions: list[Distortion] = Field(default_factory=list)


def flip_negation(span: str) -> str | None:
    """Remove/invert the first negation in `span` (None if there is none)."""
    for pat, rep in NEGATIONS:
        keep_case = lambda m, rep=rep: rep[:1].upper() + rep[1:] if m.group(0)[0].isupper() else rep
        flipped, n = re.subn(pat, keep_case, span, count=1, flags=re.I)
        if n:
            return re.sub(r" {2,}", " ", flipped)
    return None


def _find(text: str, fact: Fact, prefer_name: bool = False) -> re.Match | None:
    """Locate a fact in text: its full text, its check pattern, or an alias (case-insensitive)."""
    literal = [fact.text, *fact.check.aliases]
    if prefer_name:
        literal = [*fact.check.aliases, fact.text]
    pats = [re.escape(t) for t in literal if t]
    if fact.check.pattern and fact.check.kind in ("regex", "exact", "fuzzy"):
        pat = fact.check.pattern if fact.check.kind == "regex" else re.escape(fact.check.pattern)
        pats.insert(0 if prefer_name else 1, pat)
    for pat in pats:
        if m := re.search(pat, text, re.I):
            return m
    return None


class FakeBackend:
    def __init__(self, scenario: Scenario, config: FakeConfig | None = None, seed: int = 0):
        self.scenario = scenario
        self.config = config or FakeConfig()
        self.rng = random.Random(seed)
        self.calls = 0
        self.applied: list[list[Applied]] = []

    @staticmethod
    def incoming(messages: list[dict]) -> str:
        content = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        blocks = RECEIVED_RE.findall(content)
        return strip_attachment(blocks[-1] if blocks else content)

    def _apply(self, d: Distortion, text: str, call: int) -> tuple[str, Applied | None]:
        if d.op == "invent":
            if not d.text or d.text in text:
                return text, None
            return f"{text.rstrip()}\n{d.text}", Applied(call=call, op=d.op, after=d.text)
        fact = self.scenario.fact(d.fact_id)
        m = _find(text, fact, prefer_name=d.op == "swap_name")
        if not m:
            return text, None
        span = m.group(0)
        if d.op == "drop_fact":
            out = (text[: m.start()] + text[m.end():]).replace("  ", " ")
            return out, Applied(call=call, op=d.op, fact_id=fact.id, before=span)
        if d.op == "swap_name":
            out = text[: m.start()] + d.new + text[m.end():]
            return out, Applied(call=call, op=d.op, fact_id=fact.id, before=span, after=d.new)
        # flip_negation: remove/invert the first negation inside the fact span
        if (flipped := flip_negation(span)) is None:
            return text, None
        return text[: m.start()] + flipped + text[m.end():], Applied(
            call=call, op=d.op, fact_id=fact.id, before=span, after=flipped)

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> Completion:
        call, self.calls = self.calls, self.calls + 1
        text, applied = self.incoming(messages), []
        for d in self.config.distortions:
            if d.at is not None and d.at != call:
                continue
            if d.p < 1.0 and self.rng.random() >= d.p:
                continue
            text, a = self._apply(d, text, call)
            if a:
                applied.append(a)
        self.applied.append(applied)
        return Completion(text=text, usage=Usage(input_tokens=len(text.split()), output_tokens=len(text.split())))


GROUND_TRUTH = {"drop_fact": "lost", "swap_name": "distorted", "flip_negation": "inverted"}


def expected_labels(scenario: Scenario, applied: list[list[Applied]], noise: str = "clean"
                    ) -> tuple[list[dict[str, str]], list[set[str]]]:
    """Ground truth per message of one run: fact labels and invented claims present.

    Message 0 is the source; message i+1 is the output of call i. A noisy source starts its
    affected facts as `distorted`. Valid for the agent-written part of each message (i.e. any
    format scored with view="note", or non-forward formats with view="full").
    """
    state = {f.id: "preserved" for f in scenario.facts}
    if noise != "clean":
        state |= dict.fromkeys(next(n for n in scenario.noise if n.id == noise).affected_fact_ids, "distorted")
    invented: set[str] = set()
    labels, inv = [dict(state)], [set()]
    for call in applied:
        for a in call:
            if a.op == "invent":
                invented.add(a.after)
            else:
                state[a.fact_id] = GROUND_TRUTH[a.op]
        labels.append(dict(state))
        inv.append(set(invented))
    return labels, inv
