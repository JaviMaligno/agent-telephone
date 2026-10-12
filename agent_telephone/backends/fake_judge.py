"""FakeJudgeBackend: a scripted judge for Phase 0, speaking the judge's JSON contract.

Labels each listed fact by string matching against the message: fact text present ->
preserved; the fact text with its negation flipped (FakeBackend's rule) -> inverted;
otherwise lost. `inventions` lists claims to report when present. `bad_calls` makes the
given call indices return invalid output (to exercise retry/unscored); `noise_p` relabels a
fact at random (seeded) to simulate a disagreeing second pass.
"""
from __future__ import annotations

import json
import random
import re

from ..schema import Usage
from .base import Completion
from .fake import flip_negation

LABELS = ["preserved", "lost", "distorted", "inverted"]


def _find(needle: str, hay: str) -> str | None:
    m = re.search(re.escape(needle).replace(r"\ ", r"\s+"), hay, re.I)
    return m.group(0) if m else None


class FakeJudgeBackend:
    def __init__(self, inventions: list[str] | None = None, bad_calls: set[int] | None = None,
                 noise_p: float = 0.0, seed: int = 0):
        self.inventions = inventions or []
        self.bad_calls = bad_calls or set()
        self.noise_p, self.rng = noise_p, random.Random(seed)
        self.calls = 0

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> Completion:
        call, self.calls = self.calls, self.calls + 1
        if call in self.bad_calls:
            return Completion(text="Sure! Here are the labels: preserved, lost.")
        prompt = messages[0]["content"]
        facts = json.loads(re.search(r"<facts>\n(.*?)\n</facts>", prompt, re.S).group(1))
        message = re.search(r"<message>\n(.*?)\n</message>", prompt, re.S).group(1)
        out = []
        for f in facts:
            label, span = "lost", None
            if span := _find(f["text"], message):
                label = "preserved"
            elif (flipped := flip_negation(f["text"])) and (span := _find(flipped, message)):
                label = "inverted"
            if self.noise_p and span and self.rng.random() < self.noise_p:
                label = self.rng.choice([lab for lab in LABELS if lab != label])
                span = None if label == "lost" else span
            out.append({"fact_id": f["fact_id"], "label": label, "evidence_span": span})
        invented = [{"span": s, "rationale": "not in the brief"} for i in self.inventions if (s := _find(i, message))]
        text = json.dumps({"facts": out, "invented": invented}, ensure_ascii=False)
        return Completion(text=f"```json\n{text}\n```", usage=Usage(input_tokens=len(prompt.split()),
                                                                    output_tokens=len(text.split())))
