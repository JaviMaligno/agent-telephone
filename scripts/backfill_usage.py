"""Re-derive per-message usage of logged Claude Code runs from their saved streams (no re-run).

    uv run python scripts/backfill_usage.py --dir runs/cc_pilot [--dry-run]

Every run's rows in `messages.jsonl` get the usage `runner_cc.parse_stream` now attributes (see
`runner_cc.turn_usage`); only `usage` changes, and only when the re-parsed hops line up with the
logged rows edge by edge (and text by text, the endpoint artifact aside). A run logged several
times (run ids are deterministic, re-runs append) has one stream, that of its latest attempt: only
that block is updated, earlier blocks are left as they were and reported. The original log is kept
once as `messages.jsonl.bak` (never overwritten by later backfills).
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from agent_telephone import runner_cc
from agent_telephone.schema import Message, MessageLog, Scenario, Usage, attempt_blocks

ROOT = Path(__file__).resolve().parent.parent


def reparsed_usage(stream: Path, scenario: Scenario, first: Message) -> list[tuple[tuple, str, Usage]]:
    """(edge key, text, usage) per message as `runner_cc.record` would log it, endpoint row included."""
    roles = [r.id for r in scenario.roles]
    hops = runner_cc.parse_stream(runner_cc.read_stream(stream), scenario.source_brief(first.condition.noise), roles[0])
    out = [((h.sender, h.receiver, h.distance, h.direction), h.text, h.usage) for h in hops]
    ret = next((h for h in reversed(hops) if h.kind == "return" and h.sender == roles[-1]), None)
    if ret is not None:
        out.append(((roles[-1], runner_cc.ENDPOINT, ret.distance, "down"), None, Usage()))
    return out


def backfill(run_dir: Path, scenarios_dir: Path, dry_run: bool = False) -> dict[str, list[str]]:
    log_path = run_dir / "messages.jsonl"
    rows = MessageLog(log_path).read()
    scenarios: dict[str, Scenario] = {}
    report = {"updated": [], "skipped": [], "stale_attempts": []}
    for run_id, blocks in attempt_blocks(rows).items():
        if len(blocks) > 1:
            report["stale_attempts"].append(f"{run_id}: {len(blocks) - 1} earlier attempt(s) left unchanged")
        block = blocks[-1]
        first = rows[block[0]]
        stream = run_dir / f"{run_id}.stream.jsonl"
        if not stream.exists():
            report["skipped"].append(f"{run_id}: no stream")
            continue
        sc = scenarios.setdefault(first.scenario, Scenario.load(scenarios_dir / first.scenario))
        new = reparsed_usage(stream, sc, first)
        old = [rows[i] for i in block]
        keys_ok = [k for k, _, _ in new] == [(m.sender, m.receiver, m.distance, m.direction or "down") for m in old]
        texts_ok = keys_ok and all(t is None or t == m.text for (_, t, _), m in zip(new, old))
        if not texts_ok:
            report["skipped"].append(f"{run_id}: re-parsed hops do not match the logged rows")
            continue
        for i, (_, _, u) in zip(block, new):
            rows[i] = rows[i].model_copy(update={"usage": u})
        report["updated"].append(run_id)
    if not dry_run:
        bak = log_path.with_name(log_path.name + ".bak")
        if not bak.exists():
            shutil.copy2(log_path, bak)
        fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=".messages-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(m.model_dump_json() + "\n" for m in rows)
        os.chmod(tmp, log_path.stat().st_mode & 0o777)
        os.replace(tmp, log_path)
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=ROOT / "runs" / "cc_pilot", help="run dir with messages.jsonl + streams")
    ap.add_argument("--scenarios", type=Path, default=ROOT / "scenarios")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not rewrite the log")
    args = ap.parse_args(argv)
    rep = backfill(args.dir, args.scenarios, args.dry_run)
    print(f"{len(rep['updated'])} runs updated{' (dry run)' if args.dry_run else ''}, {len(rep['skipped'])} skipped")
    for line in rep["skipped"] + rep["stale_attempts"]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
