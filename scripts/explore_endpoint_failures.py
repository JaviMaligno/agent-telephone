"""Exploratory (not pre-registered, 2026-10-14): why software endpoints fail, per model.

1. Failure class per software artifact: no_code (the implementer returned prose, e.g. a spec),
   syntax_error, no_entry_point, runs_but_all_fail, partial, all_pass.
2. Did the input contract reach the implementer? The brief lists the input dict fields; a field
   unique to each scenario is used as a marker in the last downward message to the implementer
   (agent-written part only for non-forward formats). Endpoint mean and zero rate by marker presence.

    uv run python scripts/explore_endpoint_failures.py runs/cc_grid runs/luna_grid
"""
from __future__ import annotations

import ast
import collections
import csv
import re
import sys
from pathlib import Path

from agent_telephone.endpoint import strip_self_tests
from agent_telephone.formats import strip_attachment
from agent_telephone.schema import MessageLog, latest_attempts

ENTRY = {"software_pilot": "plan_reminders", "software_2": "prorate_seats", "software_3": "plan_purge"}
SCHEMA_MARKER = {"software_pilot": "customer_name", "software_2": "plan_name", "software_3": "closed_by"}


def failure_class(artifact: str, scenario: str, score: float) -> str:
    if not re.search(r"^\s*(def|import|from|class)\b", artifact, re.M):
        return "no_code"
    code, _ = strip_self_tests(artifact)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "syntax_error"
    if ENTRY[scenario] not in {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}:
        return "no_entry_point"
    return "all_pass" if score == 1 else "runs_but_all_fail" if score == 0 else "partial"


def main(dirs: list[str]) -> None:
    for d in map(Path, dirs):
        eps = {r["run_id"]: r for r in csv.DictReader((d / "scored_judge" / "endpoints.csv").open())
               if r["scenario"] in ENTRY}
        classes = collections.Counter()
        for rid, r in eps.items():
            art = (d / f"{rid}.artifact.py").read_text(encoding="utf-8")
            classes[r["model"], failure_class(art, r["scenario"], float(r["score"]))] += 1
        print(f"== {d}: failure classes")
        for (model, cls), n in sorted(classes.items()):
            print(f"  {model:12} {cls:18} {n}")

        last = {}
        for m in latest_attempts(MessageLog(d / "messages.jsonl").read()):
            if m.run_id in eps and m.receiver == "implementer" and m.direction == "down":
                if m.run_id not in last or m.distance > last[m.run_id].distance:
                    last[m.run_id] = m
        agg = collections.defaultdict(lambda: [0, 0, 0.0])
        for rid, m in last.items():
            fmt = m.condition.handoff_format
            text = m.text if fmt == "forward" else strip_attachment(m.text)
            score = float(eps[rid]["score"])
            a = agg[eps[rid]["model"], fmt, SCHEMA_MARKER[m.scenario] in text]
            a[0] += 1
            a[1] += score == 0
            a[2] += score
        print(f"== {d}: endpoint by input contract at the implementer")
        for (model, fmt, has), (n, zeros, total) in sorted(agg.items()):
            print(f"  {model:12} {fmt:10} {'passed' if has else 'dropped':8} n={n:3} zeros={zeros / n:.2f} mean={total / n:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/cc_grid", "runs/luna_grid"])
