"""Hand-off format instructions.

`forward` is applied mechanically: the agent writes only its own note and the harness
attaches the original brief verbatim (see `compose`).
"""
from __future__ import annotations

ATTACH_MARKER = "----- ORIGINAL BRIEF (verbatim) -----"

INSTRUCTIONS = {
    "free": (
        "Write the message you send to the next agent in your own words, as prose. "
        "Include whatever they need to do their part."
    ),
    "structured": (
        "Write the message you send to the next agent as a numbered requirement list. "
        "Give each requirement a stable ID (R1, R2, ...) and keep IDs you received unchanged."
    ),
    "forward": (
        "Write a short note for the next agent with your own guidance. "
        "Do not restate the original brief: it is attached verbatim automatically."
    ),
}


def instruction(fmt: str) -> str:
    return INSTRUCTIONS[fmt]


def compose(fmt: str, note: str, original: str) -> str:
    """Build the outgoing message from the agent's output."""
    if fmt != "forward":
        return note
    return f"{note.strip()}\n\n{ATTACH_MARKER}\n{original}"


def strip_attachment(text: str) -> str:
    """The agent-written part of a message (drops a forwarded attachment)."""
    return text.split(ATTACH_MARKER, 1)[0].strip()
