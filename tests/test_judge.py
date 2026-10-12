import json

import pytest

from agent_telephone.backends.fake_judge import FakeJudgeBackend
from agent_telephone.schema import Condition, Message
from agent_telephone.scoring.judge import CrossFamilyError, Judge, model_family, parse_output

COND = Condition(topology="hier_chain", model="fake", handoff_format="free")


def _msg(text, model="fake"):
    return Message(run_id="r", scenario="tiny", condition=COND, edge=("am", "logistics"), distance=2,
                   sender="am", receiver="logistics", text=text, model=model)


def test_model_family():
    assert model_family("claude-haiku-5-5") == model_family("anthropic/claude-sonnet-5-5") == "anthropic"
    assert model_family("gpt-6-luna-tst") == model_family("azure/gpt-5.4") == "openai"
    assert model_family("fake") != model_family("fake-judge")
    with pytest.raises(ValueError):
        model_family("mystery-model")


def test_cross_family_rule():
    with pytest.raises(CrossFamilyError):
        Judge(FakeJudgeBackend(), "claude-sonnet-5-5", ["claude-haiku-5-5"])
    j = Judge(FakeJudgeBackend(), "gpt-6-luna", ["claude-haiku-5-5", "human"])
    with pytest.raises(CrossFamilyError):  # also enforced per judged message
        j.check_agent("gpt-5.4")


def test_judge_labels_and_invented(scenario):
    text = "Dinner for 142 guests. Do serve shellfish. Budget is capped at 5000 EUR."
    j = Judge(FakeJudgeBackend(inventions=["Budget is capped at 5000 EUR."]), "fake-judge", ["fake"])
    rows, inv = j.score(scenario, _msg(text))
    assert {r.fact_id: r.label for r in rows} == {"f_shellfish": "inverted", "f_ceo": "lost"}
    assert all(r.method == "judge" for r in rows)
    assert [c.span for c in inv] == ["Budget is capped at 5000 EUR."]


def test_retry_once_then_unscored(scenario):
    text = "Do not serve shellfish."
    be = FakeJudgeBackend(bad_calls={0})
    rows, _ = Judge(be, "fake-judge", ["fake"]).score(scenario, _msg(text))
    assert be.calls == 2 and {r.label for r in rows} == {"preserved", "lost"}
    be = FakeJudgeBackend(bad_calls={0, 1})
    rows, inv = Judge(be, "fake-judge", ["fake"]).score(scenario, _msg(text))
    assert be.calls == 2 and {r.label for r in rows} == {"unscored"} and inv == []


def test_parse_output_validation():
    ok = {"facts": [{"fact_id": "a", "label": "preserved", "evidence_span": "the  CAT"}], "invented": []}
    assert parse_output(json.dumps(ok), ["a"], "I saw the cat").facts[0].label == "preserved"
    bad = [
        {"facts": [], "invented": []},                                                         # missing id
        {"facts": [{"fact_id": "a", "label": "kept", "evidence_span": "cat"}]},                # bad label
        {"facts": [{"fact_id": "a", "label": "distorted", "evidence_span": "dog"}]},           # not verbatim
        {"facts": [{"fact_id": "a", "label": "lost"}], "invented": [{"span": "dog", "rationale": "x"}]},
    ]
    for b in bad:
        with pytest.raises(ValueError):
            parse_output(json.dumps(b), ["a"], "I saw the cat")
    with pytest.raises(ValueError):
        parse_output("no json here", ["a"], "x")
