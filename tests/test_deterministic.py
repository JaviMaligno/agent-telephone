import pytest

from agent_telephone.schema import Scenario
from agent_telephone.scoring.deterministic import DeterministicScorer, FactMatcher

from .conftest import SCENARIO


def _labels(scenario, text):
    return {k: v[0] for k, v in DeterministicScorer(scenario).label_text(text).items()}


def test_only_deterministic_facts_and_clean_brief_preserved(scenario):
    assert _labels(scenario, scenario.brief) == {"f_guests": "preserved", "f_host": "preserved"}


@pytest.mark.parametrize("text,host", [
    ("THE HOST IS  SIOBHÁN NÍ DHÚILL, who arrives early.", "preserved"),   # case/space insensitive
    ("The host is Siobhan Ni Dhuill.", "distorted"),                        # accent drop, even if listed alias
    ("The host is Chevonne.", "distorted"),                                 # other value in the slot
    ("Siobhán Ní Dhúill will host.", "preserved"),                          # no slot match, exact anywhere
    ("Siobhan Ni Dhuill will host.", "distorted"),                          # no slot, accent-folded match
    ("Shiobhan Ni Dhuil will host.", "distorted"),                          # no slot, fuzzy match
    ("Chevonne will host.", "lost"),                                        # no slot, nothing close
    ("Nothing about anyone.", "lost"),
])
def test_names(scenario, text, host):
    assert _labels(scenario, text)["f_host"] == host


@pytest.mark.parametrize("text,guests", [
    ("Dinner for 142 guests.", "preserved"),
    ("Dinner for 124 guests.", "distorted"),
    ("We expect 142 people.", "preserved"),   # no slot match, regex anywhere
    ("We expect 124 people.", "lost"),
    ("Dinner on 17 May.", "lost"),
])
def test_numbers(scenario, text, guests):
    assert _labels(scenario, text)["f_guests"] == guests


def test_number_value_and_unit():
    data = {**SCENARIO, "brief": SCENARIO["brief"] + " Budget of 1,500 EUR.",
            "facts": [*SCENARIO["facts"], {"id": "f_budget", "type": "number", "text": "Budget of 1,500 EUR",
                                           "check": {"kind": "exact", "pattern": "1,500 EUR",
                                                     "slot": r"budget of (?P<value>[\d,.]+ ?\w+)"}}]}
    s = Scenario.model_validate(data)
    lab = lambda t: _labels(s, t)["f_budget"]
    assert lab("budget of 1500 EUR") == "preserved"
    assert lab("Budget of 1500.0 eur") == "preserved"
    assert lab("budget of 1500 USD") == "distorted"
    assert lab("budget of 15,000 EUR") == "distorted"


def test_non_accent_alias_is_preserved(scenario):
    fact = scenario.fact("f_host").model_copy(deep=True)
    fact.check.aliases.append("Ms Ní Dhúill")
    fact.check.slot = None
    m = FactMatcher(fact, scenario.brief)
    assert m.label("Ms Ní Dhúill will host.")[0] == "preserved"
    assert m.label("Siobhan Ni Dhuill will host.")[0] == "distorted"


def test_bad_checks_are_rejected():
    def with_check(check):
        return Scenario.model_validate({**SCENARIO, "facts": [{**SCENARIO["facts"][0], "check": check},
                                                              *SCENARIO["facts"][1:]]})

    with pytest.raises(ValueError, match="slot"):
        DeterministicScorer(with_check({"kind": "regex", "pattern": "142", "slot": "nope (?P<value>x)"}))
    with pytest.raises(ValueError, match="clean brief"):
        DeterministicScorer(with_check({"kind": "regex", "pattern": "999"}))
