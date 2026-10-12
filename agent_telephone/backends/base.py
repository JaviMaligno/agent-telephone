from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from ..schema import Usage


class Completion(BaseModel):
    text: str
    usage: Usage = Field(default_factory=Usage)


class Backend(Protocol):
    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> Completion: ...
