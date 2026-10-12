"""Scoring stack: deterministic checks first, then the cross-family LLM judge for the rest."""
from __future__ import annotations

from ..schema import Message, Scenario
from .deterministic import DeterministicScorer, is_deterministic
from .judge import Judge
from .rows import LABELS, InventedClaim, ScoreRow, View, load_scores, message_kind, save_jsonl, view_text

__all__ = ["LABELS", "DeterministicScorer", "InventedClaim", "Judge", "ScoreRow", "load_scores", "message_kind",
           "save_jsonl", "score_messages", "view_text"]


def score_messages(scenarios: dict[str, Scenario], messages: list[Message], judge: Judge | None = None,
                   view: View = "full", judge_pass: int = 0) -> tuple[list[ScoreRow], list[InventedClaim]]:
    """Score every (fact, message). Judge facts are skipped when no judge is given.

    `also_judge` facts get one row: the deterministic label, overridden by the judge when the
    anchor is present but the judge sees a distortion/inversion (or the rest of the fact lost)."""
    det = {sid: DeterministicScorer(s) for sid, s in scenarios.items()}
    rows: list[ScoreRow] = []
    invented: list[InventedClaim] = []
    for msg in messages:
        drows = {r.fact_id: r for r in det[msg.scenario].score(msg, view)}
        if judge is not None:
            jrows, inv = judge.score(scenarios[msg.scenario], msg, view, judge_pass=judge_pass)
            invented += inv
            for j in jrows:
                if not is_deterministic(scenarios[msg.scenario].fact(j.fact_id)):
                    rows.append(j)
                elif (d := drows[j.fact_id]).label == "preserved" and j.label in ("distorted", "inverted", "lost"):
                    label = "distorted" if j.label == "lost" else j.label
                    drows[j.fact_id] = j.model_copy(update={"label": label, "evidence": f"{d.evidence} | judge: {j.label}"})
        rows += drows.values()
    return rows, invented
