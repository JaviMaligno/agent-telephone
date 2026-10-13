"""Scenarios (the pilots and every other dir under scenarios/): well-formed briefs, consistent
noise, full endpoint coverage, and endpoint checks that pass on a reference and fail when a
negation is flipped."""
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
import yaml

from agent_telephone import endpoint, runner_api
from agent_telephone.backends.fake import FakeBackend
from agent_telephone.schema import Condition, Scenario
from agent_telephone.scoring.deterministic import DeterministicScorer, canon

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"
SOFTWARE, EVENT = SCENARIOS / "software_pilot", SCENARIOS / "event_pilot"
ALL = sorted(p.parent for p in SCENARIOS.glob("*/scenario.yaml"))
SOFTWARE_ALL = [p for p in ALL if Scenario.load(p).domain == "software"]
EVENT_ALL = [p for p in ALL if Scenario.load(p).domain == "event"]
HIDDEN = SOFTWARE / "hidden_tests"
REFERENCE_PY = SOFTWARE / "reference_solution" / "notifications.py"
ALL_TYPES = {"name", "number", "negation", "conditional", "priority", "rationale", "low_salience"}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(sdir: Path) -> Path:
    spec = yaml.safe_load((sdir / "fake_endpoint.yaml").read_text(encoding="utf-8"))
    return sdir / spec["reference"]


CHECKS = {d: _module(d / "plan_checks" / "checks.py", f"{d.name}_checks") for d in EVENT_ALL}
checks = CHECKS[EVENT]
name = lambda p: p.name  # noqa: E731  (pytest ids)


def test_all_scenarios_are_found():
    assert {SOFTWARE, EVENT} < set(ALL) and set(ALL) == {*SOFTWARE_ALL, *EVENT_ALL}
    assert all(Scenario.load(d).id == d.name for d in ALL)  # run ids and --scenarios use the dir name


@pytest.fixture(params=ALL, ids=name)
def pilot(request) -> Scenario:
    return Scenario.load(request.param)


def test_same_team_as_the_pilot_of_the_domain(pilot):
    twin = Scenario.load(SOFTWARE if pilot.domain == "software" else EVENT)
    assert [(r.id, r.title) for r in pilot.roles] == [(r.id, r.title) for r in twin.roles]


def test_deterministic_checks_validate_on_the_clean_brief(pilot):
    DeterministicScorer(pilot)  # raises if a pattern/slot misses or is ambiguous on the clean brief


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

def _hidden_markers(sdir: Path = SOFTWARE) -> dict[str, set[str]]:
    return endpoint.facts_by_test(sdir / "hidden_tests")


def run_hidden(artifact: Path, tmp_path: Path, hidden: Path = HIDDEN) -> dict[str, bool]:
    """Run the hidden tests against a module file; return {test name: passed}."""
    xml = tmp_path / f"{artifact.stem}.xml"
    env = os.environ | {"AGENT_TELEPHONE_ARTIFACT": str(artifact), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", str(hidden)],
                   env=env, capture_output=True, check=False)
    return {tc.get("name"): len(tc) == 0 for tc in ET.parse(xml).iter("testcase")}


@pytest.mark.parametrize("sdir", SOFTWARE_ALL, ids=name)
def test_every_software_fact_is_covered(sdir):
    sc = Scenario.load(sdir)
    judge_only = _module(sdir / "hidden_tests" / "conftest.py", f"{sdir.name}_hidden_conftest").JUDGE_ONLY
    tested = set().union(*_hidden_markers(sdir).values())
    ids = {f.id for f in sc.facts}
    assert tested | judge_only == ids and not tested & judge_only


@pytest.mark.parametrize("sdir", SOFTWARE_ALL, ids=name)
def test_hidden_tests_pass_on_reference(sdir, tmp_path):
    results = run_hidden(reference(sdir), tmp_path, sdir / "hidden_tests")
    assert set(results) == set(_hidden_markers(sdir)) and all(results.values()), results


@pytest.mark.parametrize("sdir, before, after, must_fail", [
    (SOFTWARE, 'appt.get("status") == "cancelled"', 'appt.get("status") != "cancelled"', "test_no_reminders_for_cancelled"),
    (SOFTWARE, 'and not appt.get("optout_sms_v2")', 'and appt.get("optout_sms_v2")', "test_email_when_opted_out"),
    (SOFTWARE, 'appt.get("optout_sms_v2")', 'appt.get("opt_out")', "test_sms_when_phone_and_not_opted_out"),
    (SCENARIOS / "software_2", '== "trialing"', '!= "trialing"', "test_no_lines_for_trialing"),
    (SCENARIOS / "software_2", "if new < old:", "if new > old:", "test_downgrade_returns_no_lines"),
    (SCENARIOS / "software_2", '"vat_cents": 0,', '"vat_cents": 1,', "test_credits_carry_no_vat"),
    (SCENARIOS / "software_3", 'if acc.get("litigation_hold")', 'if not acc.get("litigation_hold")',
     "test_litigation_hold_never_touched"),
    (SCENARIOS / "software_3", '!= "member"', '== "member"', "test_no_warning_when_the_member_closed_it"),
])
def test_hidden_tests_catch_mutants(tmp_path, sdir, before, after, must_fail):
    src = reference(sdir).read_text(encoding="utf-8")
    assert before in src
    mutant = tmp_path / "mutant.py"
    mutant.write_text(src.replace(before, after), encoding="utf-8")
    results = run_hidden(mutant, tmp_path, sdir / "hidden_tests")
    assert results[must_fail] is False and not all(results.values())


@pytest.mark.parametrize("sdir", SOFTWARE_ALL, ids=name)
def test_hidden_tests_fail_on_garbage(sdir, tmp_path):
    junk = tmp_path / "junk.py"
    junk.write_text("this is not python", encoding="utf-8")
    assert not any(run_hidden(junk, tmp_path, sdir / "hidden_tests").values())


# --- event: plan checks -----------------------------------------------------------------

def _plan(sdir: Path) -> dict:
    return json.loads(reference(sdir).read_text(encoding="utf-8"))


@pytest.fixture
def plan() -> dict:
    return _plan(EVENT)


@pytest.mark.parametrize("sdir", EVENT_ALL, ids=name)
def test_every_event_fact_is_covered(sdir):
    ids, c = {f.id for f in Scenario.load(sdir).facts}, CHECKS[sdir]
    assert set(c.CHECKS) | c.JUDGE_ONLY == ids and not set(c.CHECKS) & c.JUDGE_ONLY


@pytest.mark.parametrize("sdir", EVENT_ALL, ids=name)
def test_vendor_role_describes_the_schema(sdir):
    instructions = Scenario.load(sdir).role("vendor").instructions
    assert all(f'"{k}"' in instructions for k in CHECKS[sdir].SCHEMA["properties"])


@pytest.mark.parametrize("sdir", EVENT_ALL, ids=name)
def test_reference_plan_passes(sdir):
    plan, c = _plan(sdir), CHECKS[sdir]
    assert c.validate(plan) == []
    assert all(c.run_checks(plan).values())


def _mutate(plan: dict, path: str, value) -> dict:
    plan = copy.deepcopy(plan)
    *keys, last = path.split(".")
    node = plan
    for k in keys:
        node = node[int(k)] if isinstance(node, list) else node[k]
    node[last] = value
    return plan


W, C = SCENARIOS / "event_2", SCENARIOS / "event_3"


@pytest.mark.parametrize("sdir, path, value, must_fail", [
    (EVENT, "bar.alcohol_from", "18:30", "e_no_alcohol"),
    (EVENT, "catering.excluded_allergens", ["nuts"], "e_no_shellfish"),
    (EVENT, "catering.menu", [{"course": "starter", "dish": "Dublin Bay prawn cocktail", "contains": []}], "e_no_shellfish"),
    (EVENT, "transport.provider", "Coach Line Ltd", "e_no_coachline"),
    (EVENT, "contingency.keynote_overrun", "shorten_keynote", "e_priority_keynote"),
    (EVENT, "budget.cut_order", ["transport", "band"], "e_priority_band"),
    (EVENT, "guest_count", 140, "e_guests"),
    (EVENT, "vips.1.name", "Pshemislav Gribovsky", "e_vip_cfo"),
    (EVENT, "seating.top_table", ["Orla", "Fionnuala Ó Dálaigh"], "e_low_lefty"),
    (W, "catering.menu", [{"course": "dessert", "dish": "Almond and orange cake", "contains": []}], "w_no_nuts"),
    (W, "photographer", "Lumiere Fotografia Lda", "w_no_lumiere"),
    (W, "music.banned_songs", [], "w_no_macarena"),
    (W, "menu_cards.languages", ["Spanish", "English", "Finnish"], "w_cond_cards"),
    (W, "contingency.late_ceremony", "shorten_ceremony", "w_priority_late"),
    (W, "seating.front_row", ["Aino-Kaarina Väänänen", "Xóchitl Huerta Ibáñez", "Päivikki Järvenpää"], "w_low_hearing"),
    (W, "people.1.name", "Paivikki Jarvenpaa", None),  # accents optional
    (C, "av.streamed_sessions", ["keynote", "Afternoon panel"], "c_no_record"),
    (C, "venue.coffee_location", "back of the Aula", "c_no_coffee"),
    (C, "photography.flash_allowed", True, "c_no_flash"),
    (C, "rooms.overflow_booked", True, "c_cond_overflow"),
    (C, "contingency.talk_overrun", "shorten_questions", "c_priority_questions"),
    (C, "delegate_count", 38, "c_delegates"),
    (C, "av.contact", "Bartlomiej Zolkiewski", None),  # accents optional, ł included
])
def test_plan_checks_catch_mutants(sdir, path, value, must_fail):
    results = CHECKS[sdir].run_checks(_mutate(_plan(sdir), path, value))
    assert [k for k, ok in results.items() if not ok] == ([must_fail] if must_fail else [])


@pytest.mark.parametrize("sdir, path, value, must_fail", [
    # a correct plan that says where the foyer is, or excludes the panel by name, still passes
    (C, "venue.coffee_location", "foyer of the Aula Leopoldina", None),
    (C, "venue.coffee_location", "Foyer (outside the Aula)", None),
    (C, "venue.coffee_location", "foyer, not inside the Aula", None),
    (C, "venue.coffee_location", "inside the Aula, by the foyer doors", "c_no_coffee"),
    (C, "av.recorded_sessions", ["all sessions except the afternoon panel"], None),
    (C, "av.streamed_sessions", ["keynote", "everything but the panel"], None),
    (C, "av.recorded_sessions", ["keynote", "afternoon panel"], "c_no_record"),
    # the overflow threshold sits above the 238 registered: not booking is right under any reading
    (C, "rooms.book_overflow_if_over", 200, "c_cond_overflow"),
    # the budget fact is the cap: an honest "no quotes yet" total passes
    (EVENT, "budget.total_eur", 0, None),
    (EVENT, "budget.total_eur", None, None),
    (EVENT, "budget.total_eur", 19000, None),  # over the cap: the consistency check, not the fact
    (W, "budget.total_eur", None, None),
    (C, "budget.total_eur", 0, None),
])
def test_plan_checks_accept_correct_phrasings(sdir, path, value, must_fail):
    results = CHECKS[sdir].run_checks(_mutate(_plan(sdir), path, value))
    assert [k for k, ok in results.items() if not ok] == ([must_fail] if must_fail else [])


@pytest.mark.parametrize("sdir", EVENT_ALL, ids=name)
def test_budget_consistency_is_separate_and_skips_missing_totals(sdir):
    c, plan = CHECKS[sdir], _plan(sdir)
    cap = plan["budget"]["cap_eur"]
    assert c.run_consistency(plan) == {"budget_total_within_cap": True}
    assert c.run_consistency(_mutate(plan, "budget.total_eur", cap + 1)) == {"budget_total_within_cap": False}
    assert c.run_consistency(_mutate(plan, "budget.total_eur", 0)) == {}
    assert c.run_consistency(_mutate(plan, "budget.total_eur", None)) == {}


def test_endpoint_reports_consistency_apart(tmp_path):
    plan = _mutate(_plan(EVENT), "budget.total_eur", 99999)
    art = tmp_path / "plan.json"
    art.write_text(json.dumps(plan), encoding="utf-8")
    ep = endpoint.evaluate(Scenario.load(EVENT), EVENT, art)
    assert ep.passed == ep.total and ep.consistency == {"budget_total_within_cap": False}


def test_billing_tests_do_not_depend_on_the_day_count_convention(tmp_path):
    """NOW and END share the time of day, so whole-date, rounded-up and fractional day counts agree."""
    sdir = SCENARIOS / "software_2"
    src = reference(sdir).read_text(encoding="utf-8")
    before = "(end.date() - changed_at.date()).days"
    assert before in src
    for after in ("-(-(end - changed_at).total_seconds() // 86400)", "(end - changed_at).total_seconds() / 86400"):
        variant = tmp_path / "variant.py"
        variant.write_text(src.replace(before, after), encoding="utf-8")
        results = run_hidden(variant, tmp_path, sdir / "hidden_tests")
        assert results["test_prorated_over_30_days_on_billable_seats"] and results["test_rounding_in_the_customers_favour"]


@pytest.mark.parametrize("sdir", EVENT_ALL, ids=name)
def test_plan_checks_fail_on_empty_plans(sdir):
    c = CHECKS[sdir]
    assert c.validate({"date": 19}) and not any(c.run_checks({}).values())
    assert c.validate(_mutate(_plan(sdir), "date", "12/09/2026"))


def test_plan_checks_are_robust_to_bad_plans(plan):
    assert checks.validate({"date": 19}) and not any(checks.run_checks({}).values())
    assert checks.validate(_mutate(plan, "schedule.dinner_start", "8.45pm"))
    assert checks.run_checks(_mutate(plan, "vips.1.name", "Przemyslaw Grzybowski"))["e_vip_cfo"]  # accents optional


# --- both: the harness runs the pilots end to end ----------------------------------------

@pytest.mark.parametrize("path", ALL, ids=name)
@pytest.mark.parametrize("topology", ["hier_chain", "hub_relay"])
def test_pilots_run_on_fake_backend(tmp_path, path, topology):
    sc = Scenario.load(path)
    noise = sc.noise[0].id
    cond = Condition(topology=topology, model="fake", handoff_format="forward", noise=noise)
    res = runner_api.run(sc, cond, FakeBackend(sc), tmp_path / "log.jsonl")
    assert res.messages[0].text == sc.source_brief(noise) and res.artifact_path.exists()


def test_cc_pilot_scenarios_option():
    """scripts/cc_pilot.py --scenarios: names or paths -> clean + the ASR-name variants (no run)."""
    cc = _module(SCENARIOS.parent / "scripts" / "cc_pilot.py", "cc_pilot_script")
    assert cc.SCENARIOS == {"scenarios/software_pilot": ["clean", "asr_signer"], "scenarios/event_pilot": ["clean", "asr_cfo"]}
    assert cc.scenario_noises(["software_2", "scenarios/event_3"]) == {
        "scenarios/software_2": ["clean", "asr_approver"], "scenarios/event_3": ["clean", "asr_av"]}


def test_cc_pilot_full_grid_options():
    """--all-noise adds homophone/punct; replicate 0 keeps the pilot id, later ones get -s<k>."""
    cc = _module(SCENARIOS.parent / "scripts" / "cc_pilot.py", "cc_pilot_script")
    assert cc.scenario_noises(["software_2"], all_noise=True) == {
        "scenarios/software_2": ["clean", "asr_approver", "homophone_period", "punct_trial"]}
    assert cc.run_name("scenarios/software_2", "clean", "hub_relay", "free", "haiku", 0) == "software_2-hub_relay-free-clean-haiku"
    assert cc.run_name("scenarios/software_2", "clean", "hub_relay", "free", "haiku", 2).endswith("-haiku-s2")
