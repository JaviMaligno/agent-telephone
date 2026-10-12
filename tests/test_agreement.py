import pytest

from agent_telephone import runner_api
from agent_telephone.backends.fake import Distortion, FakeBackend, FakeConfig
from agent_telephone.backends.fake_judge import FakeJudgeBackend
from agent_telephone.schema import Condition
from agent_telephone.scoring import score_messages
from agent_telephone.scoring.agreement import (cohen_kappa, kappa_against_manual, kappa_between, load_manual,
                                               precision_recall)
from agent_telephone.scoring.judge import Judge


def test_cohen_kappa_known_values():
    assert cohen_kappa(list("yynn"), list("ynnn")).kappa == pytest.approx(0.5)
    assert cohen_kappa(list("abc"), list("abc")).kappa == 1.0
    assert cohen_kappa(list("aa"), list("aa")).kappa == 1.0
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_precision_recall():
    pr = precision_recall(["a", "a", "b"], ["a", "b", "b"])
    assert pr["a"] == (1.0, 0.5, 2) and pr["b"] == (0.5, 1.0, 1)


def _judged_runs(scenario, tmp_path, noise_p, seed):
    msgs = []
    for i in range(4):
        fb = FakeBackend(scenario, FakeConfig(distortions=[Distortion(op="flip_negation", fact_id="f_shellfish", p=0.5),
                                                           Distortion(op="drop_fact", fact_id="f_ceo", p=0.4)]), seed=i)
        cond = Condition(topology="hier_chain", model="fake", handoff_format="free", seed=i)
        msgs += runner_api.run(scenario, cond, fb, tmp_path / "log.jsonl", run_id=f"r{i}").messages
    judge = Judge(FakeJudgeBackend(noise_p=noise_p, seed=seed), "fake-judge", ["fake"])
    rows, _ = score_messages({scenario.id: scenario}, msgs, judge)
    return [r for r in rows if r.method == "judge"]


def test_two_passes_and_manual_anchor(scenario, tmp_path):
    a = _judged_runs(scenario, tmp_path, 0.0, 0)
    b = _judged_runs(scenario, tmp_path, 0.3, 1)
    assert kappa_between(a, a).kappa == 1.0
    ag = kappa_between(a, b)
    assert ag.n == len(a) and ag.kappa < 1.0
    csv = tmp_path / "manual.csv"
    csv.write_text("run_id,distance,sender,receiver,fact_id,label\n"
                   + "".join(f"{r.run_id},{r.distance},{r.sender},{r.receiver},{r.fact_id},{r.label}\n" for r in a[:6]))
    manual = load_manual(csv)
    assert len(manual) == 6 and kappa_against_manual(a, manual).kappa == 1.0
    assert kappa_against_manual(a, manual).n == 6
