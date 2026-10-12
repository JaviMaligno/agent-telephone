"""Deterministic endpoint checks for event_3: one check per fact id over the final plan.

`validate(plan)` returns schema errors (a tiny validator for the subset of JSON Schema used
in plan_schema.json, so no extra dependency). `run_checks(plan)` returns {fact_id: passed};
a check that raises (missing key, wrong type) counts as failed.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

SCHEMA = json.loads((Path(__file__).with_name("plan_schema.json")).read_text(encoding="utf-8"))
JUDGE_ONLY = {"c_rat_record", "c_rat_coffee"}
NO_DECOMPOSITION = str.maketrans("łŁøØđĐ", "lLoOdD")  # letters NFKD does not split
TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "null": type(None),
         "boolean": bool}


def _type_ok(value: Any, t: str) -> bool:
    return isinstance(value, TYPES[t]) and not (t in ("integer", "number") and isinstance(value, bool))


def validate(plan: Any, schema: dict = SCHEMA, path: str = "$") -> list[str]:
    if "$ref" in schema:
        schema = SCHEMA["$defs"][schema["$ref"].rsplit("/", 1)[-1]]
    types = schema.get("type")
    if types and not any(_type_ok(plan, t) for t in ([types] if isinstance(types, str) else types)):
        return [f"{path}: expected {types}, got {type(plan).__name__}"]
    errors = []
    if "enum" in schema and plan not in schema["enum"]:
        errors.append(f"{path}: {plan!r} not in {schema['enum']}")
    if "pattern" in schema and isinstance(plan, str) and not re.search(schema["pattern"], plan):
        errors.append(f"{path}: {plan!r} does not match {schema['pattern']}")
    if isinstance(plan, dict):
        errors += [f"{path}.{k}: missing" for k in schema.get("required", []) if k not in plan]
        for k, sub in schema.get("properties", {}).items():
            if k in plan:
                errors += validate(plan[k], sub, f"{path}.{k}")
    if isinstance(plan, list) and "items" in schema:
        for i, item in enumerate(plan):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors


def norm(s: str) -> str:
    """Casefold and strip accents, so 'Ó Dálaigh' == 'o dalaigh' (a dropped fada is not a lost name)."""
    s = unicodedata.normalize("NFKD", s.translate(NO_DECOMPOSITION))
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def _has(names: list[str], name: str) -> bool:
    return any(norm(name) in norm(n) for n in names)


def _speakers(p: dict) -> list[str]:
    return [v["name"] for v in p["speakers"]]


def _names_panel(entry: str) -> bool:
    """An entry that schedules the panel; "all sessions except the afternoon panel" does not."""
    n = norm(entry)
    return "panel" in n and not re.search(r"\b(?:except|excluding|excl|not|no|without|but)\b[^,;]*\bpanel", n)


def _panel_not_recorded(p: dict) -> bool:
    av = p["av"]
    return not any(_names_panel(s) for s in av["recorded_sessions"] + av["streamed_sessions"])


def _coffee_in_foyer(p: dict) -> bool:
    """Coffee in the foyer, not inside the hall; "foyer of the Aula Leopoldina" is fine."""
    loc = norm(p["venue"]["coffee_location"])
    inside = re.search(r"(?<!not )(?<!never )\b(?:inside|in|within|at the back of|back of)\s+(?:the\s+)?"
                       r"(?:aula|lecture hall|hall)\b", loc)
    return "foyer" in loc and not inside


def _dinner_before_captioning(p: dict) -> bool:
    order = p["budget"]["cut_order"]
    return "dinner" in order and ("captioning" not in order or order.index("dinner") < order.index("captioning"))


def _lapel_for_zsofia(p: dict) -> bool:
    return any("zsofia" in norm(m["speaker"]) and m["type"] == "lapel" for m in p["av"]["microphones"])


CHECKS: dict[str, Callable[[dict], bool]] = {
    "c_date": lambda p: p["date"] == "2027-03-09",
    "c_venue": lambda p: "aula leopoldina" in norm(p["venue"]["name"]),
    "c_delegates": lambda p: p["delegate_count"] == 238,
    "c_keynote_time": lambda p: p["schedule"]["keynote_start"] == "09:15",
    "c_posters": lambda p: p["schedule"]["poster_session_start"] == "15:30",
    "c_keynote_speaker": lambda p: _has(_speakers(p), "Chiamaka Nwachukwu-Eze"),
    "c_chair": lambda p: _has(_speakers(p), "Zsófia Szőke-Gyöngyösi"),
    "c_av_contact": lambda p: norm("Bartłomiej Żółkiewski") in norm(p["av"]["contact"]),
    "c_lunch": lambda p: p["catering"]["lunch_counts"] == {"vegetarian": 31, "halal": 9, "gluten_free": 4},
    "c_no_record": _panel_not_recorded,
    "c_no_flash": lambda p: p["photography"]["flash_allowed"] is False,
    "c_no_coffee": _coffee_in_foyer,
    "c_cond_overflow": lambda p: p["rooms"]["overflow_booked"] is False and p["rooms"]["book_overflow_if_over"] == 260,
    "c_cond_strike": lambda p: p["contingency"]["rail_strike_delay_minutes"] == 45,
    "c_budget": lambda p: p["budget"]["cap_eur"] == 31750,  # the fact is the cap; the total is checked below
    "c_priority_caption": _dinner_before_captioning,
    "c_priority_questions": lambda p: p["contingency"]["talk_overrun"] == "shorten_coffee",
    "c_low_ratio": lambda p: p["av"]["aspect_ratio"].replace(" ", "") == "4:3",
    "c_low_lapel": _lapel_for_zsofia,
}


def run_checks(plan: Any) -> dict[str, bool]:
    results = {}
    for fact_id, check in CHECKS.items():
        try:
            results[fact_id] = bool(check(plan))
        except (KeyError, TypeError, IndexError, AttributeError):
            results[fact_id] = False
    return results


def _total_within_cap(p: dict):
    total = p["budget"].get("total_eur")
    return None if not total else total <= p["budget"]["cap_eur"]  # no quotes yet (0 / null): skipped


# Untagged consistency checks (no fact): None = not applicable. Not part of the endpoint score.
CONSISTENCY: dict[str, Callable[[dict], bool | None]] = {"budget_total_within_cap": _total_within_cap}


def run_consistency(plan: Any) -> dict[str, bool]:
    out = {}
    for name, check in CONSISTENCY.items():
        try:
            if (ok := check(plan)) is not None:
                out[name] = bool(ok)
        except (KeyError, TypeError, IndexError, AttributeError):
            out[name] = False
    return out
