import pytest
import yaml

from agent_telephone.schema import Condition, Message, MessageLog, Scenario, Usage

from .conftest import SCENARIO


def test_scenario_yaml_round_trip(tmp_path, scenario):
    d = tmp_path / "tiny"
    d.mkdir()
    (d / "scenario.yaml").write_text(yaml.safe_dump(scenario.model_dump(), allow_unicode=True), encoding="utf-8")
    assert Scenario.load(d) == scenario
    assert Scenario.load(d / "scenario.yaml") == scenario


def test_scenario_validation():
    bad = {**SCENARIO, "noise": [{**SCENARIO["noise"][0], "affected_fact_ids": ["nope"]}]}
    with pytest.raises(ValueError):
        Scenario.model_validate(bad)
    with pytest.raises(ValueError):
        Scenario.model_validate({**SCENARIO, "domain": "cooking"})


def test_source_brief(scenario):
    assert scenario.source_brief("clean") == scenario.brief
    assert "Chevonne" in scenario.source_brief("asr_host")
    with pytest.raises(ValueError):
        scenario.source_brief("missing")


def test_message_log_round_trip(tmp_path):
    cond = Condition(topology="hier_chain", model="m", handoff_format="free", noise="clean", seed=1)
    msgs = [Message(run_id="r", scenario="s", condition=cond, edge=("user", "lead"), distance=1, sender="user",
                    receiver="lead", text="hola\nmundo", model="human", usage=Usage(input_tokens=3, output_tokens=4))] * 2
    log = MessageLog(tmp_path / "log.jsonl")
    for m in msgs:
        log.append(m)
    assert log.read() == msgs
