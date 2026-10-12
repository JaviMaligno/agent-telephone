import csv

from agent_telephone import analysis
from agent_telephone.formats import ATTACH_MARKER
from agent_telephone.schema import Condition, Message, MessageLog
from agent_telephone.scoring import score_messages


def _run(scenario, texts, noise="asr_host", run_id="n1", fmt="free"):
    cond = Condition(topology="hier_chain", model="fake", handoff_format=fmt, noise=noise)
    hops = [("user", "am"), ("am", "logistics"), ("logistics", "vendor"), ("vendor", "endpoint")]
    return [Message(run_id=run_id, scenario=scenario.id, condition=cond, edge=e, distance=i + 1, sender=e[0],
                    receiver=e[1], text=t, model="human" if i == 0 else "fake") for i, (e, t) in enumerate(zip(hops, texts))]


def test_noise_classes(scenario):
    src = scenario.source_brief("asr_host")
    msgs = _run(scenario, [src, "The host is Chevonne Nee Dool.",
                           "Chevonne Nee Dool hosts; seat Chevonne Nee Dool at the head.",
                           "The host is Siobhán Ní Dhúill."])
    msgs += _run(scenario, [src, "Nobody named.", "The host is Shona.",
                            f"note\n\n{ATTACH_MARKER}\n{src}"], run_id="n2", fmt="forward")
    msgs += _run(scenario, [scenario.brief, "x", "y", "z"], noise="clean", run_id="c1")
    scores, _ = score_messages({scenario.id: scenario}, msgs)
    rows = analysis.noise_classes({scenario.id: scenario}, msgs, scores)
    assert [(r["run_id"], r["distance"], r["class"]) for r in rows] == [
        ("n1", 2, "freeze"), ("n1", 3, "amplify"), ("n1", 4, "repair"),
        ("n2", 2, "lost"), ("n2", 3, "drift"), ("n2", 4, "lost")]  # forwarded attachment is not the agent's reading
    summary = analysis.noise_summary(rows)
    assert sum(r["freeze"] + r["amplify"] + r["repair"] + r["lost"] + r["drift"] for r in summary) == 6


def test_survival_and_outputs(scenario, tmp_path):
    msgs = _run(scenario, [scenario.brief, "Dinner for 124 guests. The host is Siobhán Ní Dhúill.",
                           "The host is Siobhán Ní Dhúill.", "{}"], noise="clean")
    scores, _ = score_messages({scenario.id: scenario}, msgs)
    by_d = {(r["kind"], r["distance"]): r for r in analysis.survival(scores)}
    assert by_d[("source", 1)]["survival"] == 1.0
    assert by_d[("handoff", 2)]["survival"] == 0.5 and by_d[("handoff", 2)]["distorted_rate"] == 0.5
    assert by_d[("handoff", 3)]["lost_rate"] == 0.5 and by_d[("endpoint", 4)]["survival"] == 0.0
    by_t = {(r["fact_type"], r["distance"]): r["survival"] for r in analysis.survival(scores, ("fact_type",))}
    assert by_t[("name", 3)] == 1.0 and by_t[("number", 3)] == 0.0

    log = tmp_path / "log.jsonl"
    for m in msgs:
        MessageLog(log).append(m)
    (tmp_path / "s").mkdir()
    import yaml
    (tmp_path / "s" / "scenario.yaml").write_text(yaml.safe_dump(scenario.model_dump(), allow_unicode=True))
    out = tmp_path / "out"
    analysis.main(["--scenario", str(tmp_path / "s"), "--log", str(log), "--out", str(out)])
    for name in ("survival_by_distance", "survival_by_type", "survival_by_condition", "noise_summary"):
        assert (out / f"{name}.csv").exists()
    rows = list(csv.DictReader((out / "survival_by_condition.csv").open()))
    assert rows[0]["topology"] == "hier_chain" and (out / "scores.jsonl").exists()
    md = (out / "summary.md").read_text()
    assert "## survival by distance" in md and "| handoff | 2 |" in md


# --- noise classes on the pilots: machine-checkable wrong readings -------------------------

from pathlib import Path  # noqa: E402

from agent_telephone.schema import Scenario  # noqa: E402
from agent_telephone.scoring.rows import row  # noqa: E402

PILOTS = Path(__file__).resolve().parent.parent / "scenarios"


def _pilot_classes(name, noise, notes, fmt="free", scores=None):
    sc = Scenario.load(PILOTS / name)
    roles = [r.id for r in sc.roles]
    cond = Condition(topology="hier_chain", model="fake", handoff_format=fmt, noise=noise)
    path = ["user", *roles, "endpoint"]
    texts = [sc.source_brief(noise), *notes]
    msgs = [Message(run_id="p", scenario=sc.id, condition=cond, edge=(a, b), distance=i + 1, sender=a, receiver=b,
                    text=t, model="human" if a == "user" else "fake") for i, ((a, b), t) in enumerate(zip(zip(path, path[1:]), texts))]
    return [(r["fact_id"], r["class"]) for r in analysis.noise_classes({sc.id: sc}, msgs, scores(sc, msgs) if scores else [])]


def test_frozen_number_is_freeze_not_lost():
    assert _pilot_classes("event_pilot", "homophone_guests", ["Plan for 140 guests."]) == [("e_guests", "freeze")]


def test_questioned_wrong_reading_is_flagged():
    note = "Sign every message Chevonne Nee Divver (looks like a transcription error, please confirm)."
    assert _pilot_classes("software_pilot", "asr_signer", [note]) == [("f_signer", "flagged")]


def test_repetition_in_context_is_freeze_spread_is_amplify():
    structured = "R7: signature Chevonne Nee Divver.\nAcceptance: every message ends with the signature Chevonne Nee Divver."
    spread = "Signature: Chevonne Nee Divver. Chevonne also approves the PL wording."
    assert _pilot_classes("software_pilot", "asr_signer", [structured, spread]) == [
        ("f_signer", "freeze"), ("f_signer", "amplify")]


def test_judge_labels_only_from_the_note_view():
    """A full-view judge label (read off a noisy attachment) must not classify a silent note."""
    def full_view_inverted(sc, msgs):
        return [row(m, sc.fact("f_cond_channel"), "inverted", "judge", view="full") for m in msgs[1:]]

    note = f"Please implement it.\n\n{ATTACH_MARKER}\nwhatever"
    got = _pilot_classes("software_pilot", "punct_channel", [note], fmt="forward", scores=full_view_inverted)
    assert got == [("f_cond_channel", "unscored")]


def test_asr_names_are_marked_unrepairable():
    sc = Scenario.load(PILOTS / "event_pilot")
    assert {n.id: n.repairable for n in sc.noise} == {"asr_cfo": False, "homophone_guests": True, "punct_alcohol": True}


# --- pooling: H1/H5 tables exclude upward hand-backs and forward notes ------------------------

def _hub_scores(scenario, tmp_path, fmt):
    from agent_telephone import runner_api
    from agent_telephone.backends.fake import FakeBackend

    cond = Condition(topology="hub_relay", model="fake", handoff_format=fmt)
    msgs = runner_api.run(scenario, cond, FakeBackend(scenario), tmp_path / f"{fmt}.jsonl").messages
    return msgs, score_messages({scenario.id: scenario}, msgs, view="note")[0]


def test_primary_pool_and_info_distance(scenario, tmp_path):
    msgs, scores = _hub_scores(scenario, tmp_path, "free")
    assert [(m.distance, m.direction, m.info_distance) for m in msgs] == [
        (1, "down", 1), (2, "down", 2), (3, "up", 3), (4, "down", 2), (5, "down", 3)]
    main = analysis.primary(scores)
    assert {(r.distance, r.direction) for r in main if r.kind == "handoff"} == {(2, "down"), (4, "down")}
    assert {r.distance for r in analysis.upward(scores)} == {3}
    by_info = {r["info_distance"] for r in analysis.survival([r for r in main if r.kind == "handoff"],
                                                              distance_key="info_distance")}
    assert by_info == {2}  # lead -> logistics and lead -> vendor are both one hop from the brief


def test_forward_notes_are_scored_as_contradiction(scenario, tmp_path):
    _, scores = _hub_scores(scenario, tmp_path, "forward")
    assert not [r for r in analysis.primary(scores) if r.kind == "handoff"]
    src = scenario.brief
    msgs = _run(scenario, [src, f"Handle it.\n\n{ATTACH_MARKER}\n{src}",
                           f"Dinner for 124 guests.\n\n{ATTACH_MARKER}\n{src}", "{}"], noise="clean", fmt="forward")
    rows = {r["distance"]: r for r in analysis.forward_contradiction(score_messages({scenario.id: scenario}, msgs,
                                                                                    view="note")[0])}
    assert rows[2]["contradiction_rate"] == 0 and rows[2]["silent_rate"] == 1  # a silent note is not a loss
    assert rows[3]["contradicted"] == 1  # the note contradicts the attachment (124 vs 142)


import pytest  # noqa: E402

SCENARIO_DIRS = sorted(p.parent for p in PILOTS.glob("*/scenario.yaml"))


@pytest.mark.parametrize("sdir, noise", [(d, n.id) for d in SCENARIO_DIRS for n in Scenario.load(d).noise],
                         ids=lambda x: x.name if isinstance(x, Path) else x)
def test_relaying_the_noisy_brief_verbatim_is_freeze(sdir, noise):
    """Every variant's wrong reading sits in its own context in the noisy brief: a lead that passes
    the brief on unquestioned freezes each affected fact (no amplify, nothing flagged)."""
    sc = Scenario.load(sdir)
    affected = next(n for n in sc.noise if n.id == noise).affected_fact_ids
    assert _pilot_classes(sdir.name, noise, [sc.source_brief(noise)]) == [(f, "freeze") for f in affected]


# --- punct wrong patterns: correct paraphrases are repairs, not the wrong reading -------------

@pytest.mark.parametrize("sid, noise, note, wrong", [
    ("event_2", "punct_cards", "Menu cards: Spanish and English, add Finnish when more than 25 of Eero's family confirm.", 0),
    ("event_2", "punct_cards", "Print menu cards in Spanish and English, and Finnish provided over 25 Finnish relatives attend.", 0),
    ("event_2", "punct_cards", "Print the menu cards in Spanish and English and add Finnish.", 1),
    ("event_3", "punct_record", "Do not record the afternoon panel; it is not to be streamed either.", 0),
    ("event_3", "punct_record", "Do not stream the afternoon panel live.", 0),
    ("event_3", "punct_record", "The afternoon panel will be streamed live.", 1),
])
def test_punct_paraphrases(sid, noise, note, wrong):
    v = next(n for n in Scenario.load(PILOTS / sid).noise if n.id == noise)
    assert analysis.wrong_reading(note, v.wrong_pattern, v.wrong_context)[0] == wrong
