from agent_telephone.backends.fake import Distortion, FakeBackend, FakeConfig
from agent_telephone.formats import ATTACH_MARKER


def _msgs(text):
    return [{"role": "user", "content": f'<received from="user">\n{text}\n</received>\n\n<instruction>\nx\n</instruction>'}]


CFG = FakeConfig(distortions=[
    Distortion(op="drop_fact", fact_id="f_ceo", at=0),
    Distortion(op="swap_name", fact_id="f_host", new="Chevonne", at=1),
    Distortion(op="flip_negation", fact_id="f_shellfish", at=1),
    Distortion(op="invent", text="Budget is capped at 5000 EUR.", at=2),
    Distortion(op="drop_fact", fact_id="f_guests", p=0.5),
])


def _chain(scenario, seed, n=3):
    fb, text, outs = FakeBackend(scenario, CFG, seed=seed), scenario.brief, []
    for _ in range(n):
        text = fb.complete("sys", _msgs(text), "fake", 100).text
        outs.append(text)
    return fb, outs


def test_distortions_and_ground_truth(scenario):
    fb, (o0, o1, o2) = _chain(scenario, seed=0)
    assert "left-handed" not in o0 and "Siobhán" in o0
    assert "Chevonne" in o1 and "Siobhán" not in o1
    assert "Do serve shellfish." in o1
    assert o2.endswith("Budget is capped at 5000 EUR.")
    ops = [[(a.op, a.fact_id) for a in call if a.fact_id != "f_guests"] for call in fb.applied]
    assert ops == [[("drop_fact", "f_ceo")],
                   [("swap_name", "f_host"), ("flip_negation", "f_shellfish")],
                   [("invent", None)]]
    swap = fb.applied[1][0]
    assert (swap.before, swap.after) == ("Siobhán Ní Dhúill", "Chevonne")
    # ground truth for the probabilistic drop matches the observable text
    guests_dropped = [any(a.fact_id == "f_guests" for a in call) for call in fb.applied]
    assert [("142" not in o) for o in (o0, o1, o2)] == [any(guests_dropped[: i + 1]) for i in range(3)]


def test_deterministic_by_seed(scenario):
    runs = {seed: _chain(scenario, seed, n=6) for seed in range(20)}
    for seed in range(20):
        fb, outs = _chain(scenario, seed, n=6)
        assert outs == runs[seed][1] and fb.applied == runs[seed][0].applied
    # the seed actually matters for p < 1
    assert len({tuple(o) for _, o in runs.values()}) > 1


def test_incoming_ignores_forward_attachment_and_uses_latest_block(scenario):
    content = (f'<received from="user">\nold\n</received>\n\n<received from="pm">\nnote\n\n{ATTACH_MARKER}\n'
               f'{scenario.brief}\n</received>\n\n<instruction>\nx\n</instruction>')
    assert FakeBackend.incoming([{"role": "user", "content": content}]) == "note"
