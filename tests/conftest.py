import pytest

from agent_telephone.schema import Scenario

SCENARIO = {
    "id": "tiny",
    "domain": "event",
    "brief": ("Dinner for 142 guests on 17 May at 17:45. The host is Siobhán Ní Dhúill. "
              "Do not serve shellfish. Oh, and the CEO is left-handed."),
    "facts": [
        {"id": "f_guests", "type": "number", "text": "142 guests", "check": {"kind": "regex", "pattern": r"\b142\b",
                                                                     "slot": r"for (?P<value>\d[\d,.]*) guests"}},
        {"id": "f_host", "type": "name", "text": "The host is Siobhán Ní Dhúill.",
         "check": {"kind": "fuzzy", "pattern": "Siobhán Ní Dhúill", "aliases": ["Siobhan Ni Dhuill"],
                   "slot": r"host is (?P<value>[^.\n]+)"}},
        {"id": "f_shellfish", "type": "negation", "text": "Do not serve shellfish.", "check": {"kind": "judge"}},
        {"id": "f_ceo", "type": "low_salience", "text": "Oh, and the CEO is left-handed.", "check": {"kind": "judge"}},
    ],
    "roles": [
        {"id": "am", "title": "account manager", "instructions": "Talk to the client and brief logistics."},
        {"id": "logistics", "title": "logistics coordinator", "instructions": "Plan logistics, brief vendors."},
        {"id": "vendor", "title": "vendor liaison", "instructions": "Produce the final plan."},
    ],
    "noise": [{"id": "asr_host", "kind": "asr_name",
               "brief": ("Dinner for 142 guests on 17 May at 17:45. The host is Chevonne Nee Dool. "
                         "Do not serve shellfish. Oh, and the CEO is left-handed."),
               "affected_fact_ids": ["f_host"], "wrong_reading": "Chevonne Nee Dool",
               "wrong_pattern": r"\bChevonne\b", "wrong_context": r"\bhost", "repairable": False}],
    "endpoint": {"kind": "plan_checks", "path": "checks.py"},
}


@pytest.fixture
def scenario() -> Scenario:
    return Scenario.model_validate(SCENARIO)
