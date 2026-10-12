"""Pilot scenarios: well-formed briefs, consistent noise, full endpoint coverage, and
endpoint checks that pass on a reference and fail when a negation is flipped."""
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from agent_telephone import runner_api
from agent_telephone.backends.fake import FakeBackend
from agent_telephone.schema import Condition, Scenario
from agent_telephone.scoring.deterministic import canon

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"
SOFTWARE, EVENT = SCENARIOS / "software_pilot", SCENARIOS / "event_pilot"
HIDDEN = SOFTWARE / "hidden_tests"
REFERENCE_PY = SOFTWARE / "reference_solution" / "notifications.py"
PLAN_CHECKS = EVENT / "plan_checks"
ALL_TYPES = {"name", "number", "negation", "conditional", "priority", "rationale", "low_salience"}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


checks = _module(PLAN_CHECKS / "checks.py", "event_pilot_checks")


@pytest.fixture(params=[SOFTWARE, EVENT], ids=lambda p: p.name)
def pilot(request) -> Scenario:
    return Scenario.load(request.param)


def test_load_and_shape(pilot):
    assert 18 <= len(pilot.facts) <= 22
    types = [f.type for f in pilot.facts]
    assert set(types) == ALL_TYPES
    assert types.count("negation") >= 3 and types.count("conditional") >= 2
    assert sum(f.type == "name" for f in pilot.facts) >= 3
    assert {n.kind for n in pilot.noise} == {"asr_name", "homophone", "punct"}


def test_fact_texts_and_patterns_are_in_the_brief(pilot):
    for f in pilot.facts:
        assert f.text in pilot.brief, f.id
        if f.check.kind == "regex":  # number facts are matched on canonical text
            assert re.search(f.check.pattern, canon(f.text) if f.type == "number" else f.text, re.I), f.id
        elif f.check.kind in ("exact", "fuzzy"):
            assert f.check.pattern in f.text, f.id


def test_noise_changes_exactly_the_affected_facts(pilot):
    for n in pilot.noise:
        assert n.brief != pilot.brief and n.wrong_reading
        for f in pilot.facts:
            assert (f.text in n.brief) == (f.id not in n.affected_fact_ids), (n.id, f.id)
        if n.kind == "asr_name":
            assert n.wrong_reading in n.brief and n.wrong_reading not in pilot.brief
        # the machine-checkable wrong reading is in the noisy source only, in its legitimate context
        assert re.search(n.wrong_pattern, canon(n.brief), re.I), n.id
        assert not re.search(n.wrong_pattern, canon(pilot.brief), re.I), n.id


# --- software: hidden tests -------------------------------------------------------------

def _hidden_markers() -> dict[str, set[str]]:
    src = (HIDDEN / "test_notifications.py").read_text(encoding="utf-8")
    pairs = re.findall(r"@facts\(([^)]*)\)\s*\ndef (test_\w+)", src)
    return {name: set(re.findall(r'"(\w+)"', ids)) for ids, name in pairs}


def run_hidden(artifact: Path, tmp_path: Path) -> dict[str, bool]:
    """Run the hidden tests against a module file; return {test name: passed}."""
    xml = tmp_path / f"{artifact.stem}.xml"
    env = os.environ | {"AGENT_TELEPHONE_ARTIFACT": str(artifact), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", str(HIDDEN)],
                   env=env, capture_output=True, check=False)
    return {tc.get("name"): len(tc) == 0 for tc in ET.parse(xml).iter("testcase")}


def test_every_software_fact_is_covered():
    sc = Scenario.load(SOFTWARE)
    judge_only = _module(HIDDEN / "conftest.py", "hidden_conftest").JUDGE_ONLY
    tested = set().union(*_hidden_markers().values())
    ids = {f.id for f in sc.facts}
    assert tested | judge_only == ids and not tested & judge_only


def test_hidden_tests_pass_on_reference(tmp_path):
    results = run_hidden(REFERENCE_PY, tmp_path)
    assert set(results) == set(_hidden_markers()) and all(results.values()), results


@pytest.mark.parametrize("before, after, must_fail", [
    ('appt.get("status") == "cancelled"', 'appt.get("status") != "cancelled"', "test_no_reminders_for_cancelled"),
    ('and not appt.get("optout_sms_v2")', 'and appt.get("optout_sms_v2")', "test_email_when_opted_out"),
    ('appt.get("optout_sms_v2")', 'appt.get("opt_out")', "test_sms_when_phone_and_not_opted_out"),
])
def test_hidden_tests_catch_mutants(tmp_path, before, after, must_fail):
    src = REFERENCE_PY.read_text(encoding="utf-8")
    assert before in src
    mutant = tmp_path / "mutant.py"
    mutant.write_text(src.replace(before, after), encoding="utf-8")
    results = run_hidden(mutant, tmp_path)
    assert results[must_fail] is False and not all(results.values())


def test_hidden_tests_fail_on_garbage(tmp_path):
    junk = tmp_path / "junk.py"
    junk.write_text("this is not python", encoding="utf-8")
    assert not any(run_hidden(junk, tmp_path).values())


# --- event: plan checks -----------------------------------------------------------------

@pytest.fixture
def plan() -> dict:
    return json.loads((PLAN_CHECKS / "reference_plan.json").read_text(encoding="utf-8"))


def test_every_event_fact_is_covered():
    ids = {f.id for f in Scenario.load(EVENT).facts}
    assert set(checks.CHECKS) | checks.JUDGE_ONLY == ids and not set(checks.CHECKS) & checks.JUDGE_ONLY


def test_vendor_role_describes_the_schema():
    instructions = Scenario.load(EVENT).role("vendor").instructions
    assert all(f'"{k}"' in instructions for k in checks.SCHEMA["properties"])


def test_reference_plan_passes(plan):
    assert checks.validate(plan) == []
    assert all(checks.run_checks(plan).values())


def _mutate(plan: dict, path: str, value) -> dict:
    plan = copy.deepcopy(plan)
    *keys, last = path.split(".")
    node = plan
    for k in keys:
        node = node[int(k)] if isinstance(node, list) else node[k]
    node[last] = value
    return plan


@pytest.mark.parametrize("path, value, must_fail", [
    ("bar.alcohol_from", "18:30", "e_no_alcohol"),
    ("catering.excluded_allergens", ["nuts"], "e_no_shellfish"),
    ("catering.menu", [{"course": "starter", "dish": "Dublin Bay prawn cocktail", "contains": []}], "e_no_shellfish"),
    ("transport.provider", "Coach Line Ltd", "e_no_coachline"),
    ("contingency.keynote_overrun", "shorten_keynote", "e_priority_keynote"),
    ("budget.cut_order", ["transport", "band"], "e_priority_band"),
    ("guest_count", 140, "e_guests"),
    ("vips.1.name", "Pshemislav Gribovsky", "e_vip_cfo"),
    ("seating.top_table", ["Orla", "Fionnuala Ó Dálaigh"], "e_low_lefty"),
])
def test_plan_checks_catch_mutants(plan, path, value, must_fail):
    results = checks.run_checks(_mutate(plan, path, value))
    assert [k for k, ok in results.items() if not ok] == [must_fail]


def test_plan_checks_are_robust_to_bad_plans(plan):
    assert checks.validate({"date": 19}) and not any(checks.run_checks({}).values())
    assert checks.validate(_mutate(plan, "schedule.dinner_start", "8.45pm"))
    assert checks.run_checks(_mutate(plan, "vips.1.name", "Przemyslaw Grzybowski"))["e_vip_cfo"]  # accents optional


# --- both: the harness runs the pilots end to end ----------------------------------------

@pytest.mark.parametrize("path", [SOFTWARE, EVENT], ids=lambda p: p.name)
@pytest.mark.parametrize("topology", ["hier_chain", "hub_relay"])
def test_pilots_run_on_fake_backend(tmp_path, path, topology):
    sc = Scenario.load(path)
    noise = sc.noise[0].id
    cond = Condition(topology=topology, model="fake", handoff_format="forward", noise=noise)
    res = runner_api.run(sc, cond, FakeBackend(sc), tmp_path / "log.jsonl")
    assert res.messages[0].text == sc.source_brief(noise) and res.artifact_path.exists()
