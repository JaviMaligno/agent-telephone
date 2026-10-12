"""Endpoint evaluation: did the final artifact satisfy the original brief?

- `hidden_tests` (software): the hidden tests are copied with the artifact into a fresh temp
  dir and run in a sandboxed subprocess (see `_sandboxed_run`), results read from junit XML. A fact passes when every
  test tagged with it via `@pytest.mark.facts(...)` passes.
- `plan_checks` (event): the artifact is parsed as JSON, schema-validated and run through the
  scenario's `CHECKS` (one per fact); untagged `CONSISTENCY` checks are reported apart, unscored.
  Before running, `strip_self_tests` removes tests the implementer bundled into the artifact
  (policy decided 2026-10-12 after the pilot, see DESIGN changelog).
Facts listed in the endpoint's `JUDGE_ONLY` have no endpoint check and are absent from `facts`.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

from pydantic import BaseModel

from .schema import Scenario

FACTS_MARK_RE = re.compile(r"@facts\(([^)]*)\)\s*\ndef (test_\w+)")


class EndpointResult(BaseModel):
    passed: int
    total: int
    facts: dict[str, bool]  # fact id -> endpoint check passed
    error: str | None = None
    self_tests_stripped: bool = False
    # untagged plan consistency checks (e.g. total within cap); not part of the score
    consistency: dict[str, bool] = {}

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


@cache
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def facts_by_test(hidden_dir: Path) -> dict[str, set[str]]:
    """{test name: fact ids} from the `@facts(...)` markers of every hidden test file."""
    out = {}
    for f in sorted(hidden_dir.glob("test_*.py")):
        for ids, name in FACTS_MARK_RE.findall(f.read_text(encoding="utf-8")):
            out[name] = set(re.findall(r"""["'](\w+)["']""", ids))
    return out


def _strip_fences(text: str) -> str:
    m = re.search(r"```[\w-]*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE")
SENSITIVE = (".ssh", ".claude", ".aws", ".azure", ".config", ".gnupg", ".kube", ".docker", ".netrc",
             ".local/state/agent-telephone", "Library/Keychains")


def _sandbox_profile(workdir: Path) -> str:
    home = Path.home()
    reads = " ".join(f'(subpath "{home / p}")' for p in SENSITIVE)
    return ("(version 1)(allow default)(deny network*)(deny file-write*)"
            f'(allow file-write* (subpath "{workdir}") (literal "/dev/null") (regex #"^/dev/tty") (regex #"^/dev/fd/"))'
            f"(deny file-read* {reads})")


def _limits(cpu_s: int) -> None:  # runs in the child before exec
    for lim, val in ((resource.RLIMIT_CPU, cpu_s), (resource.RLIMIT_FSIZE, 64 << 20), (resource.RLIMIT_AS, 4 << 30)):
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):  # e.g. RLIMIT_AS is not settable on macOS
            pass


def _sandboxed_run(cmd: list[str], workdir: Path, extra_env: dict[str, str], timeout: float) -> bool:
    """Run model-written code with a minimal env (no secrets, HOME/TMPDIR = workdir), rlimits,
    its own process group (killed whole on exit or timeout) and, on macOS, sandbox-exec
    (no network, writes only in workdir, no reads of credential dirs). Returns False on timeout."""
    env = {k: os.environ[k] for k in ENV_ALLOW if k in os.environ} | {
        "HOME": str(workdir), "TMPDIR": str(workdir), "PYTHONDONTWRITEBYTECODE": "1", **extra_env}
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        cmd = ["sandbox-exec", "-p", _sandbox_profile(workdir), *cmd]
    proc = subprocess.Popen(cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True,
                            preexec_fn=lambda: _limits(int(timeout) + 5))
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # also any children the artifact left behind
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


def _resolvable(mod: str | None, level: int) -> bool:
    if level or not mod:
        return False
    try:
        return importlib.util.find_spec(mod.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


def _is_self_test(node: ast.stmt) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name.startswith("test_") or any("pytest" in ast.unparse(d) for d in node.decorator_list)
    if isinstance(node, ast.ClassDef):
        return node.name.startswith("Test") or any("TestCase" in ast.unparse(b) for b in node.bases)
    if isinstance(node, ast.If):
        return "__name__" in ast.unparse(node.test)
    # test-framework imports stay (installed, harmless, and remaining code may reference them);
    # only imports that cannot resolve go: the artifact importing itself under an imagined name
    if isinstance(node, ast.Import):
        return any(not _resolvable(a.name, 0) for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return not _resolvable(node.module, node.level)
    return False


def strip_self_tests(code: str) -> tuple[str, bool]:
    """Drop top-level self-tests the implementer bundled with the module: test_* functions,
    Test* classes and TestCase subclasses, pytest-decorated helpers, `__main__` blocks and imports
    that cannot resolve (the artifact importing itself under the file name it imagined). The
    hidden tests measure the brief, not whether the agent's own tests import. Unparseable code is
    returned unchanged."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False
    keep = [n for n in tree.body if not _is_self_test(n)]
    if len(keep) == len(tree.body):
        return code, False
    tree.body = keep
    return ast.unparse(tree) + "\n", True


def run_hidden_tests(hidden_dir: Path, artifact: Path, timeout: float = 120) -> EndpointResult:
    mapping = facts_by_test(hidden_dir)
    facts = sorted(set().union(*mapping.values()))
    with tempfile.TemporaryDirectory(prefix="agent-telephone-endpoint-") as tmp:
        tmp = Path(tmp).resolve()
        shutil.copytree(hidden_dir, tmp / "hidden_tests", ignore=shutil.ignore_patterns("__pycache__"))
        module = tmp / "artifact.py"
        code, stripped = strip_self_tests(_strip_fences(artifact.read_text(encoding="utf-8")))
        module.write_text(code, encoding="utf-8")
        xml = tmp / "junit.xml"
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", "hidden_tests"]
        if not _sandboxed_run(cmd, tmp, {"AGENT_TELEPHONE_ARTIFACT": str(module)}, timeout):
            return EndpointResult(passed=0, total=len(mapping), facts=dict.fromkeys(facts, False), error="timeout",
                                  self_tests_stripped=stripped)
        if not xml.exists():
            return EndpointResult(passed=0, total=len(mapping), facts=dict.fromkeys(facts, False), error="no junit xml",
                                  self_tests_stripped=stripped)
        results = {tc.get("name"): len(tc) == 0 for tc in ET.parse(xml).iter("testcase")}
    ok = {t: results.get(t, False) for t in mapping}  # a test that never ran counts as failed
    per_fact = {f: all(ok[t] for t, fs in mapping.items() if f in fs) for f in facts}
    return EndpointResult(passed=sum(ok.values()), total=len(mapping), facts=per_fact, self_tests_stripped=stripped)


def run_plan_checks(checks_dir: Path, artifact: Path) -> EndpointResult:
    checks = _load(checks_dir / "checks.py", f"plan_checks_{abs(hash(checks_dir))}")
    total = len(checks.CHECKS)
    try:
        plan = json.loads(_strip_fences(artifact.read_text(encoding="utf-8")))
    except json.JSONDecodeError as e:
        return EndpointResult(passed=0, total=total, facts=dict.fromkeys(checks.CHECKS, False), error=f"not JSON: {e}")
    results = checks.run_checks(plan)
    errors = checks.validate(plan)
    consistency = checks.run_consistency(plan) if hasattr(checks, "run_consistency") else {}
    return EndpointResult(passed=sum(results.values()), total=total, facts=results, consistency=consistency,
                          error=f"{len(errors)} schema error(s): {errors[0]}" if errors else None)


def evaluate(scenario: Scenario, scenario_dir: str | Path, artifact: str | Path, timeout: float = 120) -> EndpointResult:
    where = Path(scenario_dir) / scenario.endpoint.path
    if scenario.endpoint.kind == "hidden_tests":
        return run_hidden_tests(where, Path(artifact), timeout)
    return run_plan_checks(where, Path(artifact))
