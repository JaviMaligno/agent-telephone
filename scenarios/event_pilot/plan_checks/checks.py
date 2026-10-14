"""Deterministic endpoint checks for event_pilot: one check per fact id over the final plan.

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
JUDGE_ONLY = {"e_rat_alcohol", "e_rat_coachline"}
SHELLFISH = re.compile(r"shellfish|prawn|shrimp|crab|lobster|langoustine|scampi|mussel|oyster|scallop|clam|crayfish")
NO_DECOMPOSITION = str.maketrans("łŁøØđĐ", "lLoOdD")  # letters NFKD does not split
TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "null": type(None)}


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

NEGATION = re.compile(r"\b(?:not|no|non|never|nor|without|avoid\w*|exclu\w*|prohibit\w*|declin\w*|reject\w*|"
                      r"ruled out|bann\w*|blacklist\w*|instead|other than|alternative\w*|replacement|anyone but|"
                      r"except|descartad\w*|distint\w*|excluid\w*|prohibid\w*|sin)\b")


def books(text: str, name: str) -> bool:
    """True if the provider field names `name` as the choice. The field holds one provider, so a
    negation or exclusion marker anywhere in it ("TBC (not Coachline)", "Coachline excluded",
    "must NOT be booked", "descartada") is the client's exclusion being honoured, not a booking."""
    t = norm(text or "")
    return name in t and not NEGATION.search(t)



def _has(names: list[str], name: str) -> bool:
    return any(norm(name) in norm(n) for n in names)


def _vips(p: dict) -> list[str]:
    return [v["name"] for v in p["vips"]]


def _no_shellfish(p: dict) -> bool:
    c = p["catering"]
    dishes = " ".join(f"{m['dish']} {' '.join(m['contains'])}" for m in c["menu"])
    return any("shellfish" in norm(a) for a in c["excluded_allergens"]) and not SHELLFISH.search(norm(dishes))


def _band_before_coaches(p: dict) -> bool:
    order = p["budget"]["cut_order"]
    return "band" in order and ("transport" not in order or order.index("band") < order.index("transport"))


CHECKS: dict[str, Callable[[dict], bool]] = {
    "e_date": lambda p: p["date"] == "2026-11-19",
    "e_venue": lambda p: "ardgillan orangery" in norm(p["venue"]["name"]),
    "e_guests": lambda p: p["guest_count"] == 142,
    "e_reception": lambda p: p["schedule"]["reception_start"] == "18:30"
    and "terrace" in norm(p["venue"]["reception_location"]),
    "e_dinner": lambda p: p["schedule"]["dinner_start"] == "20:45",
    "e_coach_time": lambda p: p["schedule"]["coach_departure"] == "23:15",
    "e_speaker": lambda p: _has(_vips(p), "Caoimhe Ní Fhlannagáin"),
    "e_vip_cfo": lambda p: _has(_vips(p), "Przemysław Grzybowski"),
    "e_ceo": lambda p: _has(p["seating"]["top_table"], "Fionnuala Ó Dálaigh"),
    "e_no_shellfish": _no_shellfish,
    "e_diet": lambda p: p["catering"]["dietary_counts"] == {"vegetarian": 18, "vegan": 4},
    "e_no_alcohol": lambda p: p["bar"]["alcohol_from"] >= "20:30",
    "e_no_coachline": lambda p: not books(p["transport"]["provider"], "coachline"),
    "e_cond_coach": lambda p: p["transport"]["coaches"] == 1 and p["transport"]["second_coach_if_over"] == 90,
    "e_cond_rain": lambda p: p["contingency"]["rain"]["threshold_percent"] == 40
    and "long gallery" in norm(p["contingency"]["rain"]["location"] or ""),
    "e_budget": lambda p: p["budget"]["cap_eur"] == 18500,  # the fact is the cap; the total is checked below
    "e_priority_band": _band_before_coaches,
    "e_priority_keynote": lambda p: p["contingency"]["keynote_overrun"] == "delay_dinner",
    "e_low_lefty": lambda p: norm("Fionnuala Ó Dálaigh") in norm(p["seating"]["top_table"][0]),
    "e_low_step_free": lambda p: any(re.search(r"step[- ]?free", norm(a)) for a in p["accessibility"]),
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
