"""Deterministic endpoint checks for event_2: one check per fact id over the final plan.

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
JUDGE_ONLY = {"w_rat_nuts", "w_rat_lumiere"}
NUTS = re.compile(r"almond|hazelnut|walnut|pistachio|pecan|cashew|peanut|macadamia|pine nut|praline|marzipan"
                  r"|frangipane|pesto|nougat|amaretto|amaretti")
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


def _people(p: dict) -> list[str]:
    return [v["name"] for v in p["people"]]


def _no_nuts(p: dict) -> bool:
    c = p["catering"]
    dishes = " ".join(f"{m['dish']} {' '.join(m['contains'])}" for m in c["menu"])
    return any("nut" in norm(a) for a in c["excluded_allergens"]) and not NUTS.search(norm(dishes))


def _cards(p: dict) -> bool:
    langs, m = [norm(x) for x in p["menu_cards"]["languages"]], p["menu_cards"]
    return (any("spanish" in x for x in langs) and any("english" in x for x in langs)
            and not any("finnish" in x for x in langs) and m["add_finnish_if_over"] == 25)


def _fireworks_before_interpreter(p: dict) -> bool:
    order = p["budget"]["cut_order"]
    return "fireworks" in order and ("interpreter" not in order or order.index("fireworks") < order.index("interpreter"))


def _next_to_interpreter(p: dict) -> bool:
    row = [norm(n) for n in p["seating"]["front_row"]]
    gran = [i for i, n in enumerate(row) if norm("Aino-Kaarina Väänänen") in n]
    interp = [i for i, n in enumerate(row) if norm("Päivikki Järvenpää") in n]
    return any(abs(a - b) == 1 for a in gran for b in interp)


CHECKS: dict[str, Callable[[dict], bool]] = {
    "w_date": lambda p: p["date"] == "2026-09-12",
    "w_venue": lambda p: "quinta do covelo velho" in norm(p["venue"]["name"]),
    "w_guests": lambda p: p["guest_count"] == 96,
    "w_ceremony": lambda p: p["schedule"]["ceremony_start"] == "16:00" and "chapel" in norm(p["venue"]["ceremony_location"]),
    "w_dinner": lambda p: p["schedule"]["dinner_start"] == "19:30",
    "w_shuttle": lambda p: p["schedule"]["last_shuttle"] == "01:30",
    "w_interpreter": lambda p: _has(_people(p), "Päivikki Järvenpää"),
    "w_grandmother": lambda p: _has(_people(p), "Aino-Kaarina Väänänen"),
    "w_no_nuts": _no_nuts,
    "w_diet": lambda p: p["catering"]["dietary_counts"] == {"vegetarian": 11, "children": 4},
    "w_no_lumiere": lambda p: not books(p["photographer"], "lumiere"),
    "w_no_macarena": lambda p: any("macarena" in norm(s) for s in p["music"]["banned_songs"]),
    "w_cond_cards": _cards,
    "w_cond_heat": lambda p: p["contingency"]["heat"]["threshold_celsius"] == 32
    and "wine cellar" in norm(p["contingency"]["heat"]["location"] or ""),
    "w_budget": lambda p: p["budget"]["cap_eur"] == 42000,  # the fact is the cap; the total is checked below
    "w_priority_interp": _fireworks_before_interpreter,
    "w_priority_late": lambda p: p["contingency"]["late_ceremony"] == "shorten_cocktails",
    "w_low_hearing": _next_to_interpreter,
    "w_low_dog": lambda p: any(re.search(r"\bdogs?\b", norm(n)) for n in p["venue_notes"]),
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
