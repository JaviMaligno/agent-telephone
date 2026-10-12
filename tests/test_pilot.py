"""Endpoint evaluation and the end-to-end fake pilot (Phase 0)."""
import json
from pathlib import Path

import pytest
import yaml

from agent_telephone import endpoint, pilot
from agent_telephone.schema import Scenario

from .test_pilot_scenarios import ALL, EVENT_ALL, SOFTWARE_ALL

SOFTWARE, EVENT = pilot.PILOTS
MUTANTS = {d: yaml.safe_load((d / "fake_endpoint.yaml").read_text(encoding="utf-8"))["mutants"] for d in ALL}


def _evaluate(sdir: Path, text: str, tmp_path: Path, timeout: float = 120) -> endpoint.EndpointResult:
    sc = Scenario.load(sdir)
    art = tmp_path / ("artifact.py" if sc.domain == "software" else "artifact.json")
    art.write_text(text, encoding="utf-8")
    return endpoint.evaluate(sc, sdir, art, timeout)


@pytest.mark.parametrize("sdir", ALL, ids=lambda p: p.name)
def test_reference_artifact_passes_everything(sdir, tmp_path):
    res = _evaluate(sdir, pilot.fake_artifact(sdir, set()), tmp_path)
    assert res.passed == res.total and all(res.facts.values()) and res.error is None


@pytest.mark.parametrize("sdir", ALL, ids=lambda p: p.name)
def test_mutants_cover_every_endpoint_fact(sdir, tmp_path):
    checked = set(_evaluate(sdir, pilot.fake_artifact(sdir, set()), tmp_path).facts)
    assert set(MUTANTS[sdir]) == checked


@pytest.mark.parametrize("sdir, fact_id", [(d, f) for d in ALL for f in MUTANTS[d]],
                         ids=lambda x: x.name if isinstance(x, Path) else x)
def test_each_mutant_fails_its_fact(sdir, fact_id, tmp_path):
    text = pilot.fake_artifact(sdir, {fact_id})
    assert text != pilot.fake_artifact(sdir, set()), "mutation did not apply"
    res = _evaluate(sdir, text, tmp_path)
    assert res.facts[fact_id] is False and res.passed < res.total


def test_hidden_tests_time_out(tmp_path):
    res = _evaluate(SOFTWARE, "import time\ntime.sleep(60)\n", tmp_path, timeout=3)
    assert res.error == "timeout" and res.passed == 0 and not any(res.facts.values())


def test_bad_artifacts_fail_cleanly(tmp_path):
    assert _evaluate(SOFTWARE, "this is not python", tmp_path).passed == 0
    res = _evaluate(EVENT, "Here is the plan: {", tmp_path)
    assert res.passed == 0 and res.error.startswith("not JSON")
    fenced = "```json\n" + json.dumps(json.loads(pilot.fake_artifact(EVENT, set()))) + "\n```"
    assert _evaluate(EVENT, fenced, tmp_path).passed == res.total


def test_fake_pilot_end_to_end(tmp_path):
    pilot.main(["--backend", "fake", "--scenario", str(EVENT), "--out", str(tmp_path)])
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Phase 0 pilot" in summary and "## H7 plumbing" in summary
    ep = (tmp_path / "endpoints.csv").read_text(encoding="utf-8").splitlines()
    assert len(ep) == 1 + 2 * 3 * 4  # topologies x formats x (clean + 3 noise variants)
    assert (tmp_path / "analysis_note" / "survival_by_distance.csv").exists()


# software_pilot is left out: there, combined mutants can mask each other (a channel mutant leaves no
# SMS for the "time kept" test to inspect), so a broken fact may pass. The other scenarios hold.
@pytest.mark.parametrize("sdir", [d for d in ALL if d != SOFTWARE], ids=lambda p: p.name)
def test_fake_endpoint_reflects_ground_truth(sdir, tmp_path):
    """A fact that reached the endpoint author non-preserved fails its endpoint check."""
    sc = Scenario.load(sdir)
    grid = pilot.run_grid([sdir], tmp_path)
    for ep in grid["endpoints"]:
        final = grid["truth"][ep["run_id"]][1][-1]
        for fid, ok in ep["facts"].items():
            if final[fid] != "preserved":
                assert not ok, (ep["run_id"], fid)
    assert {e["noise"] for e in grid["endpoints"]} == {"clean", *(n.id for n in sc.noise)}


# --- endpoint sandbox ------------------------------------------------------------------

PROBE = '''import os, socket, subprocess, sys, time
from pathlib import Path
import pytest
facts = pytest.mark.facts


@facts("env")
def test_env_is_minimal():
    assert "AT_FAKE_SECRET" not in os.environ and os.environ["HOME"] == os.getcwd()


@facts("fs")
def test_cannot_read_credentials_or_write_outside():
    if sys.platform != "darwin":
        pytest.skip("sandbox-exec only on macOS")
    with pytest.raises(OSError):
        Path(SECRET_DIR, "token").read_text()
    with pytest.raises(OSError):
        Path(OUTSIDE, "written").write_text("x")
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", 9), timeout=2)
'''


def _probe_dir(tmp_path, body):
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "test_probe.py").write_text(body, encoding="utf-8")
    art = tmp_path / "artifact.py"
    art.write_text("", encoding="utf-8")
    return hidden, art


def test_hidden_tests_run_sandboxed(tmp_path, monkeypatch):
    secret = Path.home() / ".claude"  # a real credential dir the artifact must not read
    monkeypatch.setenv("AT_FAKE_SECRET", "sk-should-not-leak")
    outside = tmp_path / "outside"
    outside.mkdir()
    body = PROBE.replace("SECRET_DIR", repr(str(secret))).replace("OUTSIDE", repr(str(outside)))
    res = endpoint.run_hidden_tests(*_probe_dir(tmp_path, body))
    assert res.facts["env"] is True and res.error is None
    if __import__("sys").platform == "darwin":
        assert res.facts["fs"] is True and not (outside / "written").exists()


def test_timeout_kills_the_whole_process_group(tmp_path):
    import subprocess

    body = ("import subprocess, time\nimport pytest\nfacts = pytest.mark.facts\n\n\n@facts('x')\n"
            "def test_hang():\n    subprocess.Popen(['sleep', '987'])\n    time.sleep(60)\n")
    res = endpoint.run_hidden_tests(*_probe_dir(tmp_path, body), timeout=4)
    assert res.error == "timeout"
    left = subprocess.run(["pgrep", "-f", "sleep 987"], capture_output=True, text=True).stdout
    assert left.strip() == "", "grandchild survived the timeout"


# --- endpoint checks only see their own facts -------------------------------------------

def _tags(sdir):
    return endpoint.facts_by_test(sdir / "hidden_tests")


@pytest.mark.parametrize("sdir, fact_id", [(d, f) for d in SOFTWARE_ALL for f in MUTANTS[d]],
                         ids=lambda x: x.name if isinstance(x, Path) else x)
def test_hidden_tests_assert_only_on_tagged_facts(sdir, fact_id, tmp_path):
    """Breaking one fact fails only tests tagged with it (or everything, for a universal fact)."""
    from .test_pilot_scenarios import run_hidden

    art = tmp_path / "mutant.py"
    art.write_text(pilot.fake_artifact(sdir, {fact_id}), encoding="utf-8")
    failed = {t for t, ok in run_hidden(art, tmp_path, sdir / "hidden_tests").items() if not ok}
    assert failed, "mutant not caught"
    if fact_id not in Scenario.load(sdir).endpoint.universal:
        tags = _tags(sdir)
        assert {t for t in failed if fact_id not in tags[t]} == set()


@pytest.mark.parametrize("sdir, fact_id", [(d, f) for d in EVENT_ALL for f in MUTANTS[d]],
                         ids=lambda x: x.name if isinstance(x, Path) else x)
def test_plan_checks_assert_only_on_their_fact(sdir, fact_id, tmp_path):
    ep = Scenario.load(sdir).endpoint
    res = _evaluate(sdir, pilot.fake_artifact(sdir, {fact_id}), tmp_path)
    failed = {f for f, ok in res.facts.items() if not ok}
    assert failed - {fact_id} <= {g for g, deps in ep.depends.items() if fact_id in deps}


def test_negations_are_discriminating_or_declared_insensitive():
    sw, ev = Scenario.load(SOFTWARE), Scenario.load(EVENT)
    assert "Coachline" in ev.role("vendor").instructions  # a lost "do not book Coachline" now changes the plan
    assert "log one" in sw.role("implementer").instructions.lower()  # the email filter is exercised
    assert sw.endpoint.loss_insensitive == ["f_no_both"]
    sw2, sw3, ev2, ev3 = (Scenario.load(SOFTWARE.parent / n) for n in ("software_2", "software_3", "event_2", "event_3"))
    assert "credits included" in sw2.role("implementer").instructions  # a lost "credits never carry VAT" shows
    assert "half-up" in sw2.role("implementer").instructions  # ... and so does a lost "never round in our favour"
    assert "`email`" in sw3.role("implementer").instructions  # a lost "no email in the actions" shows
    assert "Lumière Fotografia" in ev2.role("vendor").instructions  # a lost "do not book Lumière" changes the plan
    assert all(k in ev3.role("vendor").instructions for k in ("livestreams every session", "flash", "back of the"))
    assert sw2.endpoint.loss_insensitive == ["f_low_order"] and not sw3.endpoint.loss_insensitive


@pytest.mark.parametrize("sdir", ALL, ids=lambda p: p.name)
def test_endpoint_declarations_are_tested_facts(sdir, tmp_path):
    """universal / depends / loss_insensitive only name facts the endpoint actually checks."""
    ep = Scenario.load(sdir).endpoint
    checked = set(MUTANTS[sdir])
    named = {*ep.universal, *ep.loss_insensitive, *ep.depends, *(f for fs in ep.depends.values() for f in fs)}
    assert named <= checked


def test_h7_eligibility():
    sw, ev = Scenario.load(SOFTWARE), Scenario.load(EVENT)
    ok = dict.fromkeys(MUTANTS[SOFTWARE], True)
    ep = {"handoff_format": "free", "facts": ok}
    assert "f_no_both" not in pilot.h7_eligible(sw, ep) and "f_signer" in pilot.h7_eligible(sw, ep)
    assert pilot.h7_eligible(sw, {**ep, "handoff_format": "forward"}) == set()
    assert pilot.h7_eligible(sw, {**ep, "facts": {**ok, "f_entry": False}}) == {"f_entry"}
    evf = dict.fromkeys(MUTANTS[EVENT], True) | {"e_ceo": False, "e_low_lefty": False}
    got = pilot.h7_eligible(ev, {"handoff_format": "structured", "facts": evf})
    assert "e_ceo" in got and "e_low_lefty" not in got


def test_strip_self_tests_removes_bundled_tests_and_self_import():
    from agent_telephone.endpoint import strip_self_tests
    code = (
        "import logging\nfrom datetime import datetime\n\ndef plan_reminders(a, now):\n    return []\n\n"
        "import pytest\nfrom reminders import plan_reminders\n\n@pytest.fixture\ndef appt():\n    return {}\n\n"
        "def test_empty():\n    assert plan_reminders([], None) == []\n\nclass TestX:\n    pass\n\n"
        "if __name__ == '__main__':\n    pytest.main()\n"
        "import unittest\n\nclass ReminderTests(unittest.TestCase):\n    def test_a(self):\n        pass\n"
    )
    out, stripped = strip_self_tests(code)
    assert stripped
    assert "def plan_reminders" in out and "import logging" in out and "from datetime import datetime" in out
    for gone in ("from reminders", "test_empty", "TestX", "ReminderTests", "__main__", "def appt"):
        assert gone not in out


def test_strip_self_tests_leaves_clean_code_untouched():
    from agent_telephone.endpoint import strip_self_tests
    code = "import json\n\ndef f():\n    return json.dumps({})  # comment kept\n"
    assert strip_self_tests(code) == (code, False)


def test_strip_self_tests_output_imports_cleanly(tmp_path):
    import subprocess, sys
    from agent_telephone.endpoint import strip_self_tests
    code = ("import unittest\nimport pytest\nfrom brennan_reminders.core import plan\n\ndef plan_reminders(a, now):\n"
            "    return []\n\nclass ReminderTests(unittest.TestCase):\n    def test_a(self):\n        pass\n")
    out, _ = strip_self_tests(code)
    (tmp_path / "m.py").write_text(out)
    assert subprocess.run([sys.executable, "-c", "import m"], cwd=tmp_path).returncode == 0
