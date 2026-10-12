"""Deterministic checks of the pilot scenarios on realistic paraphrases (review probes)."""
import pytest

from agent_telephone.schema import Scenario
from agent_telephone.scoring.deterministic import DeterministicScorer, canon, words_to_int

from .test_pilot_scenarios import EVENT, SOFTWARE

SW, EV = DeterministicScorer(Scenario.load(SOFTWARE)), DeterministicScorer(Scenario.load(EVENT))


def lab(scorer, text, fid):
    return scorer.label_text(text)[fid][0]


@pytest.mark.parametrize("text, out", [
    ("a hundred and forty-two guests", "142 guests"), ("one hundred forty-two guests", "142 guests"),
    ("forty eight hours", "48 hours"), ("one hundred sixty characters", "160 characters"),
    ("the second one two hours before", "the second one 2 hours before"), ("from 6:30pm", "from 18:30"),
    ("at 18.30", "at 18:30"), ("at 18h30", "at 18:30"), ("until 8:00", "until 08:00"), ("8am", "08:00"),
    ("November 19th, 2026", "2026-11-19"), ("19/11/2026", "2026-11-19"), ("Thursday 19 November 2026", "Thursday 2026-11-19"),
    ("€18.5k", "€18500"), ("EUR 18 500", "EUR 18500"), ("Python 3.12", "Python 3.12"),
])
def test_canon(text, out):
    assert canon(text) == out and canon(out) == out


def test_words_to_int():
    assert words_to_int("a hundred and forty-two") == 142 == words_to_int("one hundred forty two")
    assert words_to_int("eighteen thousand five hundred") == 18500


# --- finding: number/time facts need a distorted path and format-robust matching ------------

@pytest.mark.parametrize("text, fid, label", [
    ("Nothing goes out between 22:00 and 08:00.", "f_quiet", "distorted"),
    ("- No sends after 21:00\n- No sends before 08:00", "f_quiet", "preserved"),
    ("quiet window 21:00 to 8:00", "f_quiet", "preserved"),
    ("one hundred sixty characters max per SMS", "f_sms_len", "preserved"),
    ("SMS: at most 140 characters", "f_sms_len", "distorted"),
    ("first reminder forty eight hours before", "f_first", "preserved"),
])
def test_software_numbers(text, fid, label):
    assert lab(SW, text, fid) == label


@pytest.mark.parametrize("text, fid, label", [
    ("Drinks reception from 19:30.", "e_reception", "distorted"),
    ("Drinks reception from 6:30pm on the terrace.", "e_reception", "preserved"),
    ("Reception at 18.30.", "e_reception", "preserved"),
    ("Event date: 20 November 2026.", "e_date", "distorted"),
    ("Event date: 19 November 2025.", "e_date", "distorted"),
    ("Event date: November 19th, 2026.", "e_date", "preserved"),
    ("Event date: 19/11/2026.", "e_date", "preserved"),
    ("Hard budget €15,000.", "e_budget", "distorted"),
    ("Budget cap €18.5k incl. VAT.", "e_budget", "preserved"),
    ("Budget: EUR 18 500.", "e_budget", "preserved"),
    ("We have 20 vegetarians and 4 vegans.", "e_diet", "distorted"),
    ("Dietary:\n- Vegetarian: 18\n- Vegan: 4", "e_diet", "preserved"),
    ("4 vegans and 18 vegetarians", "e_diet", "preserved"),
    ("Dietary:\n- Vegetarian: 4\n- Vegan: 18", "e_diet", "distorted"),
    ("18 vegetarians.", "e_diet", "distorted"),  # partial: the vegans are gone
    ("Dinner at 20:15.", "e_dinner", "distorted"),
    ("Coaches leave at 23:45.", "e_coach_time", "distorted"),
    ("a hundred and forty two guests", "e_guests", "preserved"),
    ("one hundred forty-two guests", "e_guests", "preserved"),
    ("Second coach only if more than 90 guests confirm.", "e_guests", "lost"),  # 90 belongs to another fact
])
def test_event_numbers(text, fid, label):
    assert lab(EV, text, fid) == label


# --- finding: slot matching must search every match and compare numbers by value -----------

@pytest.mark.parametrize("text, fid, label", [
    ("Keep the signature short to fit 160 chars. Sign every message Siobhán Ní Dhuibhir.", "f_signer", "preserved"),
    ("For PL customers the signature is Grzegorz Szczepański; everyone else gets the signature Siobhán Ní Dhuibhir.",
     "f_signer", "preserved"),
    ("For PL customers the signature is Grzegorz Szczepański.", "f_signer", "lost"),
    ("Sign every message Siobhan Ni Dhuibhir.", "f_signer", "distorted"),
    ("Send the second one 2 hours before.", "f_second", "preserved"),
    ("Second reminder: two hours before.", "f_second", "preserved"),
    ("Send the second one 3 hours before.", "f_second", "distorted"),
])
def test_slots_search_all_matches(text, fid, label):
    assert lab(SW, text, fid) == label


# --- finding: checks must not give false `preserved` -----------------------------------------

def test_first_reminder_anchored_on_first():
    assert lab(SW, "first one 24 hours before; skip it if the 48-hour point passed", "f_first") == "distorted"
    assert lab(SW, "If the 48-hour point passed, send immediately.", "f_first") == "lost"


def test_polarity_and_behaviour_facts_go_to_the_judge():
    sc = Scenario.load(SOFTWARE)
    assert sc.fact("f_optout_field").check.also_judge and sc.fact("f_quiet").check.also_judge
    assert Scenario.load(EVENT).fact("e_reception").check.also_judge


def test_also_judge_overrides_a_preserved_anchor():
    from agent_telephone.backends.base import Completion
    from agent_telephone.schema import Condition, Message
    from agent_telephone.scoring import score_messages
    from agent_telephone.scoring.judge import Judge
    import json, re

    class InvertingJudge:  # labels f_optout_field inverted, everything else preserved
        def complete(self, system, messages, model, max_tokens):
            prompt = messages[0]["content"]
            facts = json.loads(re.search(r"<facts>\n(.*?)\n</facts>", prompt, re.S).group(1))
            span = re.search(r"<message>\n(.*?)\n</message>", prompt, re.S).group(1)[:10]
            out = [{"fact_id": f["fact_id"], "label": "inverted" if f["fact_id"] == "f_optout_field" else "preserved",
                    "evidence_span": span} for f in facts]
            return Completion(text=json.dumps({"facts": out, "invented": []}))

    sc = Scenario.load(SOFTWARE)
    msg = Message(run_id="r", scenario=sc.id, distance=2, sender="lead", receiver="pm", model="fake", edge=("lead", "pm"),
                  condition=Condition(topology="hier_chain", model="fake", handoff_format="free"),
                  text="Use the `opt_out` field; optout_sms_v2 is stale.")
    assert lab(SW, msg.text, "f_optout_field") == "preserved"  # the anchor alone cannot see the inversion
    rows, _ = score_messages({sc.id: sc}, [msg], Judge(InvertingJudge(), "fake-judge", ["fake"]))
    got = {r.fact_id: r.label for r in rows}
    assert got["f_optout_field"] == "inverted" and len(rows) == len(sc.facts)
