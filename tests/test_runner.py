import json
from pathlib import Path

import pytest

from agent_telephone import runner_api
from agent_telephone.backends.fake import Distortion, FakeBackend, FakeConfig
from agent_telephone.formats import ATTACH_MARKER
from agent_telephone.schema import Condition, MessageLog


def _cond(**kw):
    return Condition(**{"topology": "hier_chain", "model": "fake", "handoff_format": "free", "noise": "clean",
                        "seed": 0, **kw})


def test_hier_chain_end_to_end(tmp_path, scenario):
    cfg = FakeConfig(distortions=[Distortion(op="drop_fact", fact_id="f_ceo", at=0),
                                  Distortion(op="swap_name", fact_id="f_host", new="Chevonne", at=1)])
    fb = FakeBackend(scenario, cfg)
    log = tmp_path / "runs" / "log.jsonl"
    res = runner_api.run(scenario, _cond(), fb, log, run_id="r1")
    rows = MessageLog(log).read()
    assert rows == res.messages
    assert [(m.sender, m.receiver, m.distance) for m in rows] == [
        ("user", "am", 1), ("am", "logistics", 2), ("logistics", "vendor", 3), ("vendor", "endpoint", 4)]
    assert rows[0].text == scenario.brief and rows[0].model == "human"
    assert "left-handed" not in rows[1].text and "Siobhán" in rows[1].text
    assert "Chevonne" in rows[2].text and "Chevonne" in rows[3].text
    assert fb.calls == 3 and [len(a) for a in fb.applied] == [1, 1, 0]
    assert res.artifact_path == log.with_name("r1.artifact.json")
    assert res.artifact_path.read_text() == rows[-1].text
    assert all(m.run_id == "r1" and m.scenario == "tiny" for m in rows)


def test_hub_relay_and_forward_attaches_original(tmp_path, scenario):
    fb = FakeBackend(scenario, FakeConfig(distortions=[Distortion(op="drop_fact", fact_id="f_guests", at=0)]))
    res = runner_api.run(scenario, _cond(topology="hub_relay", handoff_format="forward", noise="asr_host"),
                         fb, tmp_path / "log.jsonl")
    edges = [(m.sender, m.receiver, m.distance) for m in res.messages]
    assert edges == [("user", "am", 1), ("am", "logistics", 2), ("logistics", "am", 3), ("am", "vendor", 4),
                     ("vendor", "endpoint", 5)]
    assert [(m.direction, m.info_distance) for m in res.messages] == [
        ("down", 1), ("down", 2), ("up", 3), ("down", 2), ("down", 3)]
    noisy = scenario.source_brief("asr_host")
    for m in res.messages[1:4]:  # inter-agent hand-offs carry the original verbatim
        note, original = m.text.split(f"\n\n{ATTACH_MARKER}\n")
        assert original == noisy and "142" not in note
    assert ATTACH_MARKER not in res.messages[-1].text


def test_cli_fake(tmp_path, scenario, capsys):
    import yaml

    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(scenario.model_dump(), allow_unicode=True))
    (tmp_path / "fake.yaml").write_text(yaml.safe_dump({"distortions": [{"op": "flip_negation", "fact_id": "f_shellfish"}]}))
    log = tmp_path / "log.jsonl"
    runner_api.main(["--scenario", str(tmp_path), "--backend", "fake", "--topology", "peer_route",
                     "--fake-config", str(tmp_path / "fake.yaml"), "--log", str(log)])
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["distance"] for r in rows] == [1, 2, 3, 4]
    assert "Do serve shellfish." in rows[1]["text"]
    assert "messages" in capsys.readouterr().out


def test_paid_backend_refused_with_null_prices(tmp_path, scenario):
    from agent_telephone.backends.gateway import GatewayBackend
    from agent_telephone.budget import BudgetExceeded, BudgetGuard

    key = tmp_path / "key"
    key.write_text("sk-secret")
    gw = GatewayBackend("http://127.0.0.1:9", key,  # no price for the model: independent of config/prices.yaml
                        BudgetGuard(400, tmp_path / "l.ledger.json", prices={"gpt-6-luna": {"input": None, "output": None}},
                                    init_ledger=True))
    assert "sk-secret" not in repr(gw)
    with pytest.raises(BudgetExceeded):  # refused before any network I/O
        runner_api.run(scenario, _cond(model="gpt-6-luna"), gw, tmp_path / "log.jsonl")


def test_api_sender_is_told_the_receiver(tmp_path):
    """Parity with Claude Code (the Agent call names the subagent): every hand-off prompt says who reads it."""
    from agent_telephone import runner_api
    from agent_telephone.backends.base import Completion
    from agent_telephone.schema import Condition, Scenario

    sc = Scenario.load(Path(__file__).resolve().parent.parent / "scenarios" / "software_pilot")
    seen = []

    class Rec:
        def complete(self, system, messages, model, max_tokens):
            seen.append(messages[0]["content"])
            return Completion(text="x")

    runner_api.run(sc, Condition(topology="hub_relay", model="m", handoff_format="free", noise="clean"), Rec(),
                   tmp_path / "log.jsonl")
    assert "goes to the product manager (pm)" in seen[0]
    assert "goes back to the" in seen[1] and "goes to the Python developer (implementer)" in seen[2]
