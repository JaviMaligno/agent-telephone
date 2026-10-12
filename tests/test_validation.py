"""Phase 0 instrument validation: inject known distortions with FakeBackend, run the API
runner, score (deterministic + FakeJudgeBackend judge) and check the scorer recovers the
injected ground truth. Run with `-s` to see the confusion matrices."""
import itertools

from agent_telephone import analysis, runner_api
from agent_telephone.backends.fake import Distortion, FakeBackend, FakeConfig, expected_labels
from agent_telephone.backends.fake_judge import FakeJudgeBackend
from agent_telephone.schema import Condition
from agent_telephone.scoring import score_messages
from agent_telephone.scoring.agreement import format_confusion, precision_recall
from agent_telephone.scoring.judge import Judge

INVENT = "Budget is capped at 5000 EUR."

SCRIPTED = [  # (topology, format, noise, distortions)
    ("hier_chain", "free", "clean", [Distortion(op="drop_fact", fact_id="f_ceo", at=0),
                                     Distortion(op="swap_name", fact_id="f_host", new="Chevonne", at=1),
                                     Distortion(op="flip_negation", fact_id="f_shellfish", at=1),
                                     Distortion(op="invent", text=INVENT, at=2)]),
    ("hub_relay", "structured", "clean", [Distortion(op="swap_name", fact_id="f_guests", new="124", at=0),
                                          Distortion(op="swap_name", fact_id="f_host", new="Siobhan Ni Dhuill", at=1),
                                          Distortion(op="drop_fact", fact_id="f_shellfish", at=2),
                                          Distortion(op="drop_fact", fact_id="f_host", at=3)]),
    ("peer_route", "forward", "asr_host", [Distortion(op="drop_fact", fact_id="f_guests", at=1),
                                           Distortion(op="flip_negation", fact_id="f_shellfish", at=0)]),
]
RANDOM = [Distortion(op="drop_fact", fact_id="f_guests", p=0.15), Distortion(op="swap_name", fact_id="f_guests", new="1420", p=0.15),
          Distortion(op="swap_name", fact_id="f_host", new="Siobhan Ni Dhuill", p=0.2),
          Distortion(op="swap_name", fact_id="f_host", new="Shivaun Ní Dhúill", p=0.1),
          Distortion(op="drop_fact", fact_id="f_host", p=0.1), Distortion(op="flip_negation", fact_id="f_shellfish", p=0.2),
          Distortion(op="drop_fact", fact_id="f_shellfish", p=0.1), Distortion(op="drop_fact", fact_id="f_ceo", p=0.2),
          Distortion(op="invent", text=INVENT, p=0.2)]


def _cells():
    yield from SCRIPTED
    topos, fmts, noises = ["hier_chain", "hub_relay", "peer_route"], ["free", "structured", "forward"], ["clean", "asr_host"]
    for i, (t, f, n) in enumerate(itertools.product(topos, fmts, noises)):
        for seed in range(4):
            yield t, f, n, RANDOM, i * 10 + seed


def test_scorer_recovers_injected_ground_truth(scenario, tmp_path):
    scenarios = {scenario.id: scenario}
    judge = Judge(FakeJudgeBackend(inventions=[INVENT]), "fake-judge", ["fake"])
    truth, pred, all_msgs, all_scores = {"deterministic": [], "judge": []}, {"deterministic": [], "judge": []}, [], []
    for n, cell in enumerate(_cells()):
        topo, fmt, noise, distortions, *seed = cell
        seed = seed[0] if seed else 0
        fb = FakeBackend(scenario, FakeConfig(distortions=distortions), seed=seed)
        cond = Condition(topology=topo, model="fake", handoff_format=fmt, noise=noise, seed=seed)
        msgs = runner_api.run(scenario, cond, fb, tmp_path / "log.jsonl", run_id=f"run{n}").messages
        gt, gt_inv = expected_labels(scenario, fb.applied, noise)
        assert len(gt) == len(msgs)
        # forward attaches the original verbatim; ground truth describes the agent-written part
        scores, invented = score_messages(scenarios, msgs, judge, view="note")
        index = {(m.distance, m.sender, m.receiver): i for i, m in enumerate(msgs)}
        for r in scores:
            i = index[(r.distance, r.sender, r.receiver)]
            truth[r.method].append(gt[i][r.fact_id])
            pred[r.method].append(r.label)
        got_inv = [set() for _ in msgs]
        for c in invented:
            got_inv[index[(c.distance, c.sender, c.receiver)]].add(c.span)
        assert got_inv == gt_inv
        all_msgs += msgs
        all_scores += scores

    for method in ("deterministic", "judge"):
        print(f"\n[{method}] {len(truth[method])} units\n" + format_confusion(truth[method], pred[method]))
    # every label actually occurs, so the check is not vacuous
    assert {"preserved", "lost", "distorted"} <= set(truth["deterministic"])
    assert {"preserved", "lost", "inverted"} <= set(truth["judge"])
    # deterministic labels must match ground truth exactly on name/number facts
    assert pred["deterministic"] == truth["deterministic"]
    for lab, (p, r, _) in precision_recall(truth["deterministic"], pred["deterministic"]).items():
        assert p == r == 1.0, lab
    assert pred["judge"] == truth["judge"]

    # analysis runs end to end on the validated scores; noisy runs show the frozen name
    out = analysis.analyse(scenarios, all_msgs, all_scores, [], tmp_path / "analysis")
    noise_rows = analysis.noise_classes(scenarios, all_msgs, all_scores)
    assert noise_rows and {r["class"] for r in noise_rows} <= {"freeze", "lost", "drift"}
    assert out["summary"].read_text().startswith("# agent-telephone analysis")


def test_negative_control_unrecorded_edit_is_caught(scenario, tmp_path):
    """An edit the ground truth does not know about must show up as a mismatch."""
    fb = FakeBackend(scenario, FakeConfig())
    cond = Condition(topology="hier_chain", model="fake", handoff_format="free")
    msgs = runner_api.run(scenario, cond, fb, tmp_path / "log.jsonl", run_id="ctl").messages
    msgs[2] = msgs[2].model_copy(update={"text": msgs[2].text.replace("142", "241")})
    gt, _ = expected_labels(scenario, fb.applied)
    scores, _ = score_messages({scenario.id: scenario}, msgs)
    wrong = [(r.distance, r.fact_id, r.label) for r in scores if r.label != gt[r.distance - 1][r.fact_id]]
    assert wrong == [(3, "f_guests", "distorted")]
