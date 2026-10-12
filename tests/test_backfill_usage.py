import importlib.util
import shutil
from pathlib import Path

import yaml

from agent_telephone import runner_cc
from agent_telephone.schema import MessageLog, Usage
from tests.test_runner_cc import HUB, SMOKE, _cond

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("backfill_usage", ROOT / "scripts" / "backfill_usage.py")
backfill_usage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill_usage)


def test_backfill_restores_usage_from_stream(tmp_path):
    run_dir, sc_dir = tmp_path / "run", tmp_path / "scenarios" / SMOKE.id
    sc_dir.mkdir(parents=True)
    (sc_dir / "scenario.yaml").write_text(yaml.safe_dump(SMOKE.model_dump(mode="json")), encoding="utf-8")
    log = run_dir / "messages.jsonl"
    events = runner_cc.read_stream(HUB)
    want = [m.usage for m in runner_cc.record(events, SMOKE, _cond(topology="hub_relay"), log, "r1", HUB).messages]
    shutil.copy(HUB, run_dir / "r1.stream.jsonl")
    rows = MessageLog(log).read()  # an old log: stub usage on every row, logged twice (a re-run)
    stale = [m.model_copy(update={"usage": Usage(output_tokens=3)}) for m in rows]
    log.write_text("".join(m.model_dump_json() + "\n" for m in stale * 2), encoding="utf-8")

    rep = backfill_usage.backfill(run_dir, tmp_path / "scenarios")
    assert rep["updated"] == ["r1"] and rep["skipped"] == [] and len(rep["stale_attempts"]) == 1
    got = MessageLog(log).read()
    assert [m.usage for m in got[len(rows):]] == want and sum(u.output_tokens for u in want) == 4531
    assert all(m.usage.output_tokens == 3 for m in got[: len(rows)])  # earlier attempt untouched
    assert [m.text for m in got] == [m.text for m in stale * 2]
    assert (run_dir / "messages.jsonl.bak").read_text(encoding="utf-8").count("\n") == 2 * len(rows)
