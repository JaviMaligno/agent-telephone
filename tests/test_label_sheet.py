"""Manual labelling sheet (scripts/make_label_sheet.py) and its agreement with the scorers."""
import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

from agent_telephone import runner_api
from agent_telephone.backends.fake import Distortion, FakeBackend, FakeConfig
from agent_telephone.backends.fake_judge import FakeJudgeBackend
from agent_telephone.schema import Condition, MessageLog
from agent_telephone.scoring import save_jsonl, score_messages
from agent_telephone.scoring.agreement import against_manual_by_method, load_sheet, main as agreement_main
from agent_telephone.scoring.judge import Judge

from .conftest import SCENARIO

_spec = importlib.util.spec_from_file_location("make_label_sheet",
                                               Path(__file__).resolve().parent.parent / "scripts" / "make_label_sheet.py")
mls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mls)


@pytest.fixture
def pilot_dir(tmp_path, scenario):
    """Fake pilot output: runs.csv (one invalid run) + messages.jsonl + scenarios/<id>/scenario.yaml."""
    runs, sdir = tmp_path / "runs", tmp_path / "scenarios" / "tiny"
    sdir.mkdir(parents=True)
    (sdir / "scenario.yaml").write_text(yaml.safe_dump(SCENARIO), encoding="utf-8")
    rows = []
    for i, (model, topo, fmt) in enumerate([("fake-haiku", "hier_chain", "free"), ("fake-sonnet", "hub_relay", "structured"),
                                            ("fake-haiku", "hub_relay", "forward"), ("fake-sonnet", "hier_chain", "free")]):
        fb = FakeBackend(scenario, FakeConfig(distortions=[Distortion(op="flip_negation", fact_id="f_shellfish", p=0.5),
                                                           Distortion(op="drop_fact", fact_id="f_ceo", p=0.5),
                                                           Distortion(op="swap_name", fact_id="f_host", new="Chevonne",
                                                                      p=0.5)]), seed=i)
        cond = Condition(topology=topo, model=model, handoff_format=fmt, seed=i)
        runner_api.run(scenario, cond, fb, runs / "messages.jsonl", run_id=f"r{i}")
        rows.append({"run_id": f"r{i}", "valid": "False" if i == 3 else "True"})
    with (runs / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "valid"])
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def _sheet(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sheet_is_blind_stratified_and_seeded(pilot_dir):
    sheet = mls.build(pilot_dir / "runs", pilot_dir / "scenarios", 12, 0, {"free", "structured"})
    rows = _sheet(sheet)
    assert list(rows[0]) == mls.COLUMNS and len(rows) == 12
    assert all(r["label"] == "" and r["notes"] == "" for r in rows)
    assert not any(c in mls.COLUMNS for c in ("method", "evidence", "deterministic", "judge"))
    assert {r["run_id"] for r in rows} <= {"r0", "r1"}  # r2 is forward, r3 invalid
    assert all(r["sender"] != "user" and r["receiver"] != "endpoint" for r in rows)
    assert len({(r["run_id"], r["distance"], r["sender"], r["receiver"], r["fact_id"]) for r in rows}) == 12
    # stratified: one unit per stratum per round, so every fact type appears
    assert {r["fact_type"] for r in rows} == {"number", "name", "negation", "low_salience"}
    for r in rows:
        text = (sheet.parent / r["message_file"]).read_text(encoding="utf-8")
        assert r["fact_text"] in text and f"distance {r['distance']}" in text
    with pytest.raises(FileExistsError):
        mls.build(pilot_dir / "runs", pilot_dir / "scenarios", 12, 0, {"free", "structured"})
    again = mls.build(pilot_dir / "runs", pilot_dir / "scenarios", 12, 0, {"free", "structured"}, pilot_dir / "again")
    assert _sheet(again) == rows


def test_excerpt_finds_the_fact_window(scenario):
    text = "Intro line.\nVenue is booked.\nThe host is Siobhán Ní Dhúill, seat her centrally.\nClosing remark."
    ex = mls.excerpt(text, scenario.fact("f_host"), k=1)
    assert "Siobhán" in ex and "Intro line" not in ex
    assert "no lexical match" in mls.excerpt("Nothing relevant.", scenario.fact("f_ceo"))
    assert "forty-two" in mls.excerpt("Dinner for a hundred and forty-two people.", scenario.fact("f_guests"))  # canon match


def test_filled_sheet_kappa_vs_deterministic_and_judge(pilot_dir, scenario, capsys):
    runs = pilot_dir / "runs"
    sheet = mls.build(runs, pilot_dir / "scenarios", 16, 1, {"free", "structured"})
    msgs = MessageLog(runs / "messages.jsonl").read()
    det, _ = score_messages({"tiny": scenario}, msgs, None, view="note")
    judged, _ = score_messages({"tiny": scenario}, msgs, Judge(FakeJudgeBackend(), "fake-judge", ["fake"]), view="note")
    truth = {r.key: r.label for r in det + [j for j in judged if j.method == "judge"]}
    # synthetic labeller: agrees with the scorers except one flipped deterministic item; one row left blank
    rows = _sheet(sheet)
    for r in rows:
        r["label"] = truth[(r["run_id"], int(r["distance"]), r["sender"], r["receiver"], r["fact_id"])]
    det_rows = [r for r in rows if r["fact_type"] in ("number", "name")]
    det_rows[0]["label"] = "lost" if det_rows[0]["label"] != "lost" else "preserved"
    next(r for r in rows if r is not det_rows[0])["label"] = ""  # not yet labelled: skipped
    det_rows[0]["label"] = det_rows[0]["label"].upper()  # case-insensitive
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mls.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    manual = load_sheet(sheet)
    assert len(manual) == 15
    res = against_manual_by_method(det + judged, manual)
    assert set(res) == {"deterministic", "judge"}
    assert res["judge"][0].kappa == 1.0 and res["deterministic"][0].kappa < 1.0
    assert sum(a.n for a, _, _ in res.values()) <= len(manual)
    # CLI: deterministic recomputed from the log, judge from a score file
    save_jsonl([j for j in judged if j.method == "judge"], pilot_dir / "judge.jsonl")
    agreement_main(["--sheet", str(sheet), "--log", str(runs / "messages.jsonl"), "--scenarios",
                    str(pilot_dir / "scenarios"), "--scores", str(pilot_dir / "judge.jsonl")])
    out = capsys.readouterr().out
    assert "manual vs deterministic" in out and "manual vs judge: kappa=1.000" in out


def test_sheet_rejects_bad_labels(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("item_id,run_id,distance,sender,receiver,fact_id,label\nL001,r0,2,am,logistics,f_ceo,maybe\n")
    with pytest.raises(ValueError, match="L001"):
        load_sheet(p)
