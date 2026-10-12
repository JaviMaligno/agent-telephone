"""LLM judge over a Backend, cross-family with respect to the agents it judges.

Output contract (validated):
    {"facts": [{"fact_id": str, "label": preserved|lost|distorted|inverted, "evidence_span": str|null}],
     "invented": [{"span": str, "rationale": str}]}
Every requested fact id exactly once; for any label other than `lost` the evidence span must
be a verbatim (whitespace/case-insensitive) substring of the message. Invalid output is
retried once with the validation error; a second failure marks every fact `unscored`.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable

from pydantic import BaseModel, Field, ValidationError

from ..backends.base import Backend
from ..schema import Fact, Message, Scenario
from .deterministic import needs_judge
from .rows import InventedClaim, Label, ScoreRow, View, row, view_text

FAMILY_PREFIXES = [("claude", "anthropic"), ("anthropic", "anthropic"), ("gpt", "openai"), ("o1", "openai"),
                   ("o3", "openai"), ("o4", "openai"), ("openai", "openai"), ("gemini", "google"),
                   ("llama", "meta"), ("mistral", "mistral"), ("qwen", "alibaba"), ("deepseek", "deepseek")]

SYSTEM = """You are a meticulous annotator. You compare a message passed between agents against the \
original client brief and label, for each listed fact, whether the message still carries it.

Labels:
- preserved: present with the same meaning (paraphrase is fine).
- lost: absent.
- distorted: present with a different meaning (wrong name, number, scope, priority, reason...).
- inverted: polarity flipped (a negation dropped or added, a condition reversed). Use this rather \
than distorted when the polarity is what changed.

evidence_span: copy the exact substring of the message that supports the label; null only for lost.
invented: claims in the message that are not traceable to the brief and that a downstream agent \
would act on. Copy the exact span and give a one-line rationale. Do not list paraphrases of the brief.

Reply with a single JSON object and nothing else:
{"facts": [{"fact_id": "...", "label": "...", "evidence_span": "..."}], \
"invented": [{"span": "...", "rationale": "..."}]}"""


class CrossFamilyError(ValueError):
    pass


def model_family(model: str) -> str:
    """Provider family of a model id. Each `fake*` id is its own family; unknown ids are refused."""
    name = model.lower().rsplit("/", 1)[-1]
    if name.startswith("fake"):
        return name
    for prefix, family in FAMILY_PREFIXES:
        if name.startswith(prefix):
            return family
    raise ValueError(f"unknown model family for {model!r}; add it to FAMILY_PREFIXES")


class FactJudgement(BaseModel):
    fact_id: str
    label: Label
    evidence_span: str | None = None


class InventedSpan(BaseModel):
    span: str
    rationale: str


class JudgeOutput(BaseModel):
    facts: list[FactJudgement]
    invented: list[InventedSpan] = Field(default_factory=list)


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def parse_output(raw: str, fact_ids: list[str], message: str) -> JudgeOutput:
    """Parse and validate judge output; raises ValueError with a reason the judge can act on."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object found")
    try:
        out = JudgeOutput.model_validate(json.loads(m.group(0)))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"invalid JSON or schema: {e}") from e
    got = [f.fact_id for f in out.facts]
    if sorted(got) != sorted(fact_ids):
        raise ValueError(f"fact ids must be exactly {sorted(fact_ids)}, got {sorted(got)}")
    hay = _squash(message)
    for f in out.facts:
        if f.label != "lost" and (not f.evidence_span or _squash(f.evidence_span) not in hay):
            raise ValueError(f"evidence_span for {f.fact_id} is not a verbatim substring of the message")
    for inv in out.invented:
        if _squash(inv.span) not in hay:
            raise ValueError(f"invented span {inv.span[:40]!r} is not a verbatim substring of the message")
    return out


def build_prompt(brief: str, facts: list[Fact], message: str) -> str:
    listed = json.dumps([{"fact_id": f.id, "type": f.type, "text": f.text} for f in facts], ensure_ascii=False)
    return (f"<brief>\n{brief}\n</brief>\n\n<facts>\n{listed}\n</facts>\n\n"
            f"<message>\n{message}\n</message>\n\nLabel every fact and list invented claims. JSON only.")


class Judge:
    def __init__(self, backend: Backend, model: str, agent_models: Iterable[str], max_tokens: int = 2048):
        self.backend, self.model, self.max_tokens = backend, model, max_tokens
        self.family = model_family(model)
        self.agent_families = set()
        for m in agent_models:
            self.check_agent(m)

    def check_agent(self, agent_model: str) -> None:
        if agent_model == "human":
            return
        fam = model_family(agent_model)
        if fam == self.family:
            raise CrossFamilyError(f"judge {self.model!r} and agent {agent_model!r} are both {fam!r}")
        self.agent_families.add(fam)

    def judge(self, brief: str, facts: list[Fact], message: str) -> tuple[JudgeOutput | None, int]:
        """Returns (validated output or None if unscored, attempts used)."""
        ids = [f.id for f in facts]
        convo = [{"role": "user", "content": build_prompt(brief, facts, message)}]
        for attempt in (1, 2):
            raw = self.backend.complete(SYSTEM, convo, self.model, self.max_tokens).text
            try:
                return parse_output(raw, ids, message), attempt
            except ValueError as e:
                convo += [{"role": "assistant", "content": raw},
                          {"role": "user", "content": f"Your reply was invalid: {e}. Reply with the corrected JSON only."}]
        return None, 2

    def score(self, scenario: Scenario, msg: Message, view: View = "full", facts: list[Fact] | None = None,
              judge_pass: int = 0) -> tuple[list[ScoreRow], list[InventedClaim]]:
        """Judge one message. Default facts: those without a deterministic check, plus `also_judge` ones."""
        self.check_agent(msg.model)
        facts = facts if facts is not None else [f for f in scenario.facts if needs_judge(f)]
        if not facts:
            return [], []
        text = view_text(msg, view)
        out, _ = self.judge(scenario.brief, facts, text)
        if out is None:
            return [row(msg, f, "unscored", "judge", judge_pass=judge_pass, view=view) for f in facts], []
        by_id = {j.fact_id: j for j in out.facts}
        rows = [row(msg, f, by_id[f.id].label, "judge", by_id[f.id].evidence_span, judge_pass, view) for f in facts]
        inv = [InventedClaim(run_id=msg.run_id, distance=msg.distance, sender=msg.sender, receiver=msg.receiver,
                             span=i.span, rationale=i.rationale, judge_pass=judge_pass) for i in out.invented]
        return rows, inv
