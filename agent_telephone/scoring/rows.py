"""Score rows: one per (fact, message), plus invented claims per message."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..formats import strip_attachment
from ..schema import Condition, FactType, Message
from ..topologies import USER

Label = Literal["preserved", "lost", "distorted", "inverted"]
ScoreLabel = Literal["preserved", "lost", "distorted", "inverted", "unscored"]
LABELS: tuple[str, ...] = ("preserved", "lost", "distorted", "inverted")
View = Literal["full", "note"]
ENDPOINT = "endpoint"


class ScoreRow(BaseModel):
    run_id: str
    scenario: str
    condition: Condition
    distance: int
    sender: str
    receiver: str
    kind: Literal["source", "handoff", "endpoint"]
    fact_id: str
    fact_type: FactType
    label: ScoreLabel
    method: Literal["deterministic", "judge"]
    evidence: str | None = None
    judge_pass: int = 0
    view: View = "full"
    direction: Literal["down", "up"] = "down"
    info_distance: int | None = None

    @property
    def key(self) -> tuple:
        """Identity of the (message, fact) unit, shared across scoring passes."""
        return (self.run_id, self.distance, self.sender, self.receiver, self.fact_id)


class InventedClaim(BaseModel):
    run_id: str
    distance: int
    sender: str
    receiver: str
    span: str
    rationale: str
    judge_pass: int = 0


def message_kind(msg: Message) -> str:
    if msg.sender == USER:
        return "source"
    return "endpoint" if msg.receiver == ENDPOINT else "handoff"


def view_text(msg: Message, view: View = "full") -> str:
    """`full`: what the receiver sees (incl. a forwarded attachment); `note`: the agent-written part."""
    return msg.text if view == "full" or msg.sender == USER else strip_attachment(msg.text)


def row(msg: Message, fact, label: str, method: str, evidence: str | None = None, judge_pass: int = 0,
        view: View = "full") -> ScoreRow:
    return ScoreRow(run_id=msg.run_id, scenario=msg.scenario, condition=msg.condition, distance=msg.distance,
                    sender=msg.sender, receiver=msg.receiver, kind=message_kind(msg), fact_id=fact.id,
                    fact_type=fact.type, label=label, method=method, evidence=evidence, judge_pass=judge_pass,
                    view=view, direction=msg.direction or "down",
                    info_distance=msg.info_distance if msg.info_distance is not None else msg.distance)


def save_jsonl(rows: list[BaseModel], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")


def load_scores(path: str | Path) -> list[ScoreRow]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ScoreRow.model_validate_json(line) for line in lines if line.strip()]
