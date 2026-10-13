import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

from agent_telephone import runner_cc
from agent_telephone.backends.fake_judge import FakeJudgeBackend
from agent_telephone.schema import MessageLog
from tests.test_runner_cc import HUB, SMOKE, _cond

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("score_real", ROOT / "scripts" / "score_real.py")
score_real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_real)


@pytest.fixture
def real_dir(tmp_path):
    """A run dir like runs/cc_pilot: two attempts of r1 (only the second valid) and an invalid r2."""
    run_dir, sc_dir = tmp_path / "run", tmp_path / "scenarios" / SMOKE.id
    sc_dir.mkdir(parents=True)
    (sc_dir / "scenario.yaml").write_text(yaml.safe_dump(SMOKE.model_dump(mode="json")), encoding="utf-8")
    log = run_dir / "messages.jsonl"
    events = runner_cc.read_stream(HUB)
    for run_id in ("r1", "r1", "r2"):
        runner_cc.record(events, SMOKE, _cond(topology="hub_relay"), log, run_id, HUB)
    with (run_dir / "runs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerows([["run_id", "valid"], ["r1", "False"], ["r2", "False"], ["r1", "True"]])
    return run_dir, tmp_path / "scenarios"


def test_load_runs_keeps_latest_valid_attempt(real_dir):
    runs, messages = score_real.load_runs(real_dir[0])
    assert list(runs) == ["r1"]
    per_attempt = len(MessageLog(real_dir[0] / "messages.jsonl").read()) // 3
    assert len(messages) == per_attempt and {m.run_id for m in messages} == {"r1"}


def test_judge_plan_counts_unique_prompts_and_prices(real_dir, tmp_path):
    runs, messages = score_real.load_runs(real_dir[0])
    scenarios = score_real.load_scenarios(messages, real_dir[1])
    with pytest.raises(SystemExit, match="non-Claude judge"):
        score_real.judge_plan(scenarios, messages, "claude-haiku-5-5", 2, 2048)
    prices = tmp_path / "prices.yaml"
    prices.write_text(yaml.safe_dump({"models": {"gpt-x": {"input": None, "output": None},
                                                 "gpt-y": {"input": 1.0, "output": 10.0}}}))
    plan = score_real.judge_plan(scenarios, messages, "gpt-x", 2, 1000, prices)
    assert plan["cost_usd"] is None and "fill it" in plan["cost_note"]
    note, full, second = (p["calls"] for p in plan["passes"])
    assert note == second > 0 and full == 0  # full view only adds texts that differ (forward runs)
    plan = score_real.judge_plan(scenarios, messages, "gpt-y", 2, 1000, prices)
    tin = plan["input_tokens_per_call"]["total"]
    assert plan["cost_usd"]["worst_no_retries"] == pytest.approx((tin * 1.0 + 1000 * plan["calls"] * 10.0) / 1e6)

    # the real judge path sends exactly the planned number of calls
    sent = []

    class Counting(FakeJudgeBackend):
        def complete(self, *a, **kw):
            sent.append(1)
            return super().complete(*a, **kw)

    out = score_real.run_judge(scenarios, messages, Counting(), "fake-judge", 2, 1000)
    assert len(sent) == plan["calls"] and set(out) == {("note", 0), ("full", 0), ("note", 1)}
    assert all(r.method == "judge" for r in out["note", 1][0] if r.fact_id == "f_nolog")


def test_main_deterministic_and_dry_run(real_dir, capsys):
    run_dir, scenarios = real_dir
    out = run_dir / "scored"
    score_real.main(["--dir", str(run_dir), "--scenarios", str(scenarios), "--skip-endpoints"])
    assert (out / "summary.md").exists() and (out / "scores_note.jsonl").exists()
    assert (out / "analysis_full" / "survival_by_distance.csv").exists()
    score_real.main(["--dir", str(run_dir), "--scenarios", str(scenarios), "--judge", "gateway",
                     "--judge-model", "gpt-5.6-sol-tst", "--dry-run"])
    assert '"cost_usd": null' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="refusing to run the judge"):
        score_real.main(["--dir", str(run_dir), "--scenarios", str(scenarios), "--judge", "gateway",
                         "--judge-model", "gpt-5.6-sol-tst"])


def test_judge_prefetch_hits_cache_and_resumes(tmp_path):
    """Prefetched first attempts are exactly the requests the judge sends, and persist for resume."""
    import importlib.util
    from pathlib import Path as P
    from agent_telephone.backends.base import Completion
    spec = importlib.util.spec_from_file_location("sr", P(__file__).resolve().parent.parent / "scripts" / "score_real.py")
    sr = importlib.util.module_from_spec(spec); spec.loader.exec_module(sr)
    calls = []

    class B:
        def complete(self, system, messages, model, max_tokens):
            calls.append(messages[0]["content"]); return Completion(text=f"ans:{messages[0]['content']}")

    store = tmp_path / "c.jsonl"
    c = sr.OncePerPrompt(B(), store)
    c.prefetch(["p1", "p2", "p1"], "m", 10, workers=3)
    assert sorted(calls) == ["p1", "p2"]
    assert c.complete(sr.SYSTEM, [{"role": "user", "content": "p1"}], "m", 10).text == "ans:p1" and len(calls) == 2
    again = sr.OncePerPrompt(B(), store)
    again.prefetch(["p1", "p2"], "m", 10, workers=2)
    assert len(calls) == 2
