"""Regression fixtures: real spans from the Claude Code pilot (runs/cc_pilot) that the first
deterministic scorer mislabelled (review of 2026-10-12). Run ids are given for traceability."""
import pytest

from agent_telephone import analysis, pilot
from agent_telephone.schema import Condition, Message, Scenario
from agent_telephone.scoring import score_messages
from agent_telephone.scoring.deterministic import DeterministicScorer, canon

from .test_pilot_scenarios import EVENT, SOFTWARE

SC_SW, SC_EV = Scenario.load(SOFTWARE), Scenario.load(EVENT)
SW, EV = DeterministicScorer(SC_SW), DeterministicScorer(SC_EV)


def lab(scorer, text, fid, noise=None):
    return scorer.label_text(text, noise)[fid][0]


# --- f_first / f_second: anchored on the offset, not on the ordinal ---------------------------

@pytest.mark.parametrize("text, first, second", [
    # software_pilot-hier_chain-structured-asr_signer-haiku d2 (ordinal 31 chars away)
    ("a first reminder at `starts_at` minus 48 hours and a second reminder at `starts_at` minus 2 hours",
     "preserved", "preserved"),
    # software_pilot-hier_chain-free-clean-haiku d4
    ("The 48-hour point is starts minus timedelta(hours=48), and the 2-hour point is starts minus timedelta(hours=2)",
     "preserved", "preserved"),
    # software_pilot-hier_chain-forward-asr_signer-sonnet d4
    ("Targets: starts_at-48h and starts_at-2h", "preserved", "preserved"),
    # software_pilot-hub_relay-structured-clean-haiku d4
    ("reminder 1 is at `starts_at` minus 48 hours. Reminder 2 is at `starts_at` minus 2 hours", "preserved", "preserved"),
    # software_pilot-hier_chain-free-asr_signer-haiku d4 (the old slot jumped from 'first' to the 2)
    ("a 48-hour first offset and a 2-hour second offset", "preserved", "preserved"),
    # the 48 of f_cond_late / f_priority_2h is not the first offset
    ("If you have to cut scope, the 2-hour reminder matters more than the 48-hour one.", "lost", "lost"),
    ("The 2-hour reminder takes priority over the 48-hour one if scope must be cut.", "lost", "lost"),
    ("Send the first reminder 24 hours before.", "distorted", "lost"),
])
def test_offsets(text, first, second):
    assert (lab(SW, text, "f_first"), lab(SW, text, "f_second")) == (first, second)


# --- ASR runs: the wrong reading present is distorted, not lost -------------------------------

@pytest.mark.parametrize("scorer, noise, fid, text", [
    # event_pilot-hier_chain-free-asr_cfo-haiku d2 (the slot needs ', the CFO')
    (EV, "asr_cfo", "e_vip_cfo", "Pshemislav Gribovsky, CFO of the client's biggest partner, is to be treated as a VIP"),
    # event_pilot-hub_relay-free-asr_cfo-haiku d4
    (EV, "asr_cfo", "e_vip_cfo", "VIP guest: Pshemislav Gribovsky, who uses a wheelchair"),
    # software_pilot-hier_chain-free-asr_signer-sonnet d3
    (SW, "asr_signer", "f_signer", 'every message ends with "Chevonne Nee Divver"'),
    # code relayed up the chain / endpoint
    (SW, "asr_signer", "f_signer", 'SIG_DEFAULT = "Chevonne Nee Divver"'),
])
def test_asr_reading_is_distorted(scorer, noise, fid, text):
    assert lab(scorer, text, fid, noise) == "distorted"
    assert lab(scorer, text, fid) == "lost"  # without the run's noise variant nothing is known


def test_asr_reading_through_score_messages():
    cond = Condition(topology="hier_chain", model="haiku", handoff_format="free", noise="asr_cfo")
    m = Message(run_id="r", scenario=SC_EV.id, condition=cond, edge=("am", "logistics"), distance=2, sender="am",
                receiver="logistics", model="haiku", text="VIP guest: Pshemislav Gribovsky, who uses a wheelchair")
    rows, _ = score_messages({SC_EV.id: SC_EV}, [m])
    got = next(r for r in rows if r.fact_id == "e_vip_cfo")
    assert got.label == "distorted" and "Pshemislav" in got.evidence


# --- precedence: the exact pattern wins over stray slot values elsewhere ----------------------

@pytest.mark.parametrize("text, fid, label", [
    # software_pilot-hier_chain-structured-clean-sonnet d4
    ("21:00 <= t < 08:00 is quiet", "f_quiet", "preserved"),
    # software_pilot-hier_chain-free-clean-haiku d4
    ("Quiet means local hour >= 21 or local hour < 8.\nDST is handled by zoneinfo (clocks change at 01:00 UTC).",
     "f_quiet", "preserved"),
    # software_pilot-hub_relay-structured-asr_signer-haiku d4
    ("no send in [21:00, 08:00); the 00:00 rollover is handled", "f_quiet", "preserved"),
    ("if 8 <= local.hour < 21:\n    return send", "f_quiet", "preserved"),
    ("QUIET_START_HOUR = 21\nQUIET_END_HOUR = 8", "f_quiet", "preserved"),
    ("QUIET_START = time(21, 0)\nQUIET_END = time(8, 0)", "f_quiet", "preserved"),
    ("Nothing goes out between 22:00 and 08:00.", "f_quiet", "distorted"),  # still a real conflict
    # f_sms_len: "70-character UCS-2 segments" next to the 160 limit (9 rows)
    ("SMS_MAX = 160  # GSM-7; 70-character UCS-2 segments are not split", "f_sms_len", "preserved"),
    ("SMS limit 160; note 70-character UCS-2 segments.", "f_sms_len", "preserved"),
])
def test_exact_pattern_wins(text, fid, label):
    assert lab(SW, text, fid) == label


# --- event slots: JSON keys, wrong-context dates and numbers ----------------------------------

@pytest.mark.parametrize("text, fid, label", [
    # event_pilot-hier_chain-free-clean-sonnet d4
    ('{"schedule":{"reception_start":"18:30"}} alcohol-free reception and bar opening after 20:30', "e_reception",
     "preserved"),
    ("Soft drinks during reception and bar service from 20:30 only.", "e_reception", "lost"),
    # event_pilot-hub_relay-structured-asr_cfo-sonnet d5
    ('{"dinner_start":"20:45"} which squeezes dinner against the fixed 23:15 coaches', "e_dinner", "preserved"),
    ("which squeezes dinner against the fixed 23:15 coaches", "e_dinner", "lost"),
    ('"coach_departure": "23:15"', "e_coach_time", "preserved"),
    # event_pilot-hub_relay-forward-clean-haiku d2/d4 and hier_chain-forward-clean-haiku d3
    ("Today is 2026-10-12.", "e_date", "lost"),
    ("forecast checked the day before (Wednesday 2026-11-18)", "e_date", "lost"),
    ('"date": "2026-11-19"', "e_date", "preserved"),
    ("demand of about 56 to 90 guests", "e_guests", "lost"),
    ("Thu 2026-11-19, 142 guests", "e_guests", "preserved"),
    ("€18,500 hard cap, leaving about €600", "e_budget", "preserved"),
    ('"budget": {"cap_eur": 18500, "total_eur": 0}', "e_budget", "preserved"),
    ('"dietary_counts": {"vegetarian": 18, "vegan": 4}', "e_diet", "preserved"),
    ("**Vegetarian (18):** no meat stock\n**Vegan (4):** no animal products", "e_diet", "preserved"),
    ("18 guests are vegetarian and 4 are vegan.", "e_diet", "preserved"),
])
def test_event_slots(text, fid, label):
    assert lab(EV, text, fid) == label


# --- code and JSON artifacts ------------------------------------------------------------------

def test_code_canonicalisation():
    assert canon("FIRST_LEAD = timedelta(hours=48)") == "FIRST_LEAD = timedelta(hours: 48 hours)"
    assert canon("QUIET_START = time(21, 0)") == "QUIET_START = 21:00"
    assert canon("if 8 <= local.hour < 21:") == "if 08:00 <= local.hour < 21:00:"
    for t in ("timedelta(hours=48)", "time(8, 0)", "local hour >= 21", "X_HOUR = 8", "OFFSET_HOURS = 2"):
        assert canon(canon(t)) == canon(t)


def test_escaped_names_in_code():
    # software_pilot-hier_chain-structured-clean-haiku d5-d8 and endpoint
    code = 'SIG_DEFAULT = "Siobh\\u00e1n N\\u00ed Dhuibhir"\nSIG_PL = "Grzegorz Szczepa\\u0144ski"\n'
    assert lab(SW, code, "f_signer") == "preserved" and lab(SW, code, "f_signer_pl") == "preserved"
    endpoint = "FIRST_LEAD = timedelta(hours=48)\nSECOND_LEAD = timedelta(hours=2)\n"
    assert lab(SW, endpoint, "f_first") == "preserved" and lab(SW, endpoint, "f_second") == "preserved"


# --- noise classes: code is freeze, doubt must be about the name ------------------------------

def _classes(scenario, noise, note):
    cond = Condition(topology="hier_chain", model="fake", handoff_format="free", noise=noise)
    roles = [r.id for r in scenario.roles]
    msgs = [Message(run_id="p", scenario=scenario.id, condition=cond, edge=("user", roles[0]), distance=1,
                    sender="user", receiver=roles[0], model="human", text=scenario.source_brief(noise)),
            Message(run_id="p", scenario=scenario.id, condition=cond, edge=(roles[1], roles[0]), distance=5,
                    sender=roles[1], receiver=roles[0], model="fake", text=note, direction="up")]
    return [r["class"] for r in analysis.noise_classes({scenario.id: scenario}, msgs, [])]


@pytest.mark.parametrize("note", [
    # software_pilot-hier_chain-free-asr_signer-haiku d5-d8 and hier_chain-structured-asr_signer-sonnet d5-d8
    'Here is the module.\n\nDEFAULT_SIGNATURE = "Chevonne Nee Divver"\nSIG_PL = "Grzegorz Szczepański"\n',
    'Done.\n```python\nSIG_DEFAULT = "Chevonne Nee Divver"\n```\n',
    'Tests:\n        self.assertTrue(out[0]["text"].endswith("Chevonne Nee Divver"))\n',
])
def test_relayed_code_is_freeze(note):
    assert _classes(SC_SW, "asr_signer", note) == ["freeze"]


@pytest.mark.parametrize("scenario, noise, note, cls", [
    (SC_EV, "asr_cfo", "The wheelchair-accessible coach for Pshemislav Gribovsky is not confirmed yet.", "freeze"),
    (SC_EV, "asr_cfo", "VIP: Pshemislav Gribovsky. Please ask the venue to confirm the step-free route.", "freeze"),
    (SC_SW, "asr_signer", "Sign every message Chevonne Nee Divver. Use exactly this spelling as given. "
                          "Do not add an accent.", "freeze"),
    (SC_EV, "asr_cfo", "VIP: Pshemislav Gribovsky (name as transcribed, please confirm the spelling).", "flagged"),
    (SC_SW, "asr_signer", "Signature: Chevonne Nee Divver?", "flagged"),
])
def test_doubt_is_about_the_name(scenario, noise, note, cls):
    assert _classes(scenario, noise, note) == [cls] * len(next(n for n in scenario.noise if n.id == noise).affected_fact_ids)


# --- H7: noise-affected facts of noisy runs are excluded and reported per fact ---------------

def test_h7_excludes_noise_affected_facts():
    ep = {"run_id": "r", "scenario": SC_EV.id, "handoff_format": "free", "noise": "asr_cfo",
          "facts": {"e_vip_cfo": False, "e_ceo": False, "e_low_step_free": True}}
    assert "e_vip_cfo" not in pilot.h7_eligible(SC_EV, ep) and "e_ceo" in pilot.h7_eligible(SC_EV, ep)
    assert "e_vip_cfo" in pilot.h7_eligible(SC_EV, ep, with_noise=True)
    cond = Condition(topology="hier_chain", model="haiku", handoff_format="free", noise="asr_cfo")
    m = Message(run_id="r", scenario=SC_EV.id, condition=cond, edge=("logistics", "vendor"), distance=3,
                sender="logistics", receiver="vendor", model="haiku", text="VIP: Pshemislav Gribovsky.")
    rows, _ = score_messages({SC_EV.id: SC_EV}, [m], view="note")
    h7 = pilot.endpoint_vs_last_handoff(rows, [ep], {SC_EV.id: SC_EV})
    assert h7 == [{"label_at_last_handoff": "lost", "facts": 1, "endpoint_pass_rate": 0.0}]  # e_ceo only
    by_fact = {(r["fact_id"], r["noise_affected"]): r for r in pilot.h7_by_fact(rows, [ep], {SC_EV.id: SC_EV})}
    assert by_fact[("e_vip_cfo", True)]["label_at_last_handoff"] == "distorted"
    assert by_fact[("e_ceo", False)]["facts"] == 1


def test_forward_note_repeating_the_noisy_source_is_not_a_contradiction():
    from agent_telephone.formats import ATTACH_MARKER
    cond = Condition(topology="hier_chain", model="haiku", handoff_format="forward", noise="asr_cfo")
    src = SC_EV.source_brief("asr_cfo")
    m = Message(run_id="r", scenario=SC_EV.id, condition=cond, edge=("am", "logistics"), distance=2, sender="am",
                receiver="logistics", model="haiku",
                text=f"Mind Pshemislav Gribovsky's access. Dinner for 124 guests.\n\n{ATTACH_MARKER}\n{src}")
    rows, _ = score_messages({SC_EV.id: SC_EV}, [m], view="note")
    (fc,) = analysis.forward_contradiction(rows)
    assert fc["contradicted"] == 1 and fc["noise_reading"] == 1  # 124 contradicts; the ASR name echoes the source


def test_negated_vendor_mention_is_not_a_booking():
    """Grid finding 2026-10-14: plans wrote 'TBC (not Coachline)' / 'Non-Coachline provider' and the
    substring check failed every one of them (0 % in all models)."""
    import importlib.util
    from pathlib import Path as P
    root = P(__file__).resolve().parent.parent / "scenarios"
    for scen, name, booked, excluded in [("event_pilot", "coachline", "Coachline (framework rates)", "TBC (not Coachline)"),
                                         ("event_2", "lumiere", "Lumière Fotografia", "Non-Lumière photographer TBC")]:
        sp = importlib.util.spec_from_file_location(f"chk_{scen}", root / scen / "plan_checks" / "checks.py")
        c = importlib.util.module_from_spec(sp); sp.loader.exec_module(c)
        assert c.books(booked, name) and not c.books(excluded, name)


def test_negation_after_the_vendor_name():
    import importlib.util
    from pathlib import Path as P
    sp = importlib.util.spec_from_file_location("chk_e2", P(__file__).resolve().parent.parent / "scenarios" / "event_2" / "plan_checks" / "checks.py")
    c = importlib.util.module_from_spec(sp); sp.loader.exec_module(c)
    for t in ["To be booked; Lumière Fotografia excluded at client request", "Not booked; couple declined Lumière Fotografia.",
              "TBD: alternative to Lumière Fotografia required"]:
        assert not c.books(t, "lumiere"), t
    assert c.books("Lumière Fotografia", "lumiere") and c.books("Lumière Fotografia (deposit paid)", "lumiere")
