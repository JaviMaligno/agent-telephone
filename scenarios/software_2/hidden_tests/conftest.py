"""Hidden tests for software_2: run against the implementer's final module.

The module under test is loaded from $AGENT_TELEPHONE_ARTIFACT (a .py file); without it,
the reference solution is used. Each test carries @pytest.mark.facts(...) with the fact
ids of the original brief it checks; JUDGE_ONLY lists facts no test can check.
"""
import importlib.util
import os
from pathlib import Path

import pytest

REFERENCE = Path(__file__).resolve().parent.parent / "reference_solution" / "billing.py"
JUDGE_ONLY = {"f_rat_seats", "f_rat_round", "f_priority_credit", "f_low_legacy"}


def pytest_configure(config):
    config.addinivalue_line("markers", "facts(*ids): fact ids of the brief this test checks")


@pytest.fixture(scope="session")
def mod():
    path = Path(os.environ.get("AGENT_TELEPHONE_ARTIFACT") or REFERENCE)
    spec = importlib.util.spec_from_file_location("artifact_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
