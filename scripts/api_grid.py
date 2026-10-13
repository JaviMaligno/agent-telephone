"""Grid on the API harness (runner_api) for models outside Claude Code: gpt-6-luna through the Codex
CLI on the ChatGPT subscription (no API spend). Same grid, run ids and runs.csv as cc_pilot.py, so
score_real.py reads both.

    uv run python scripts/api_grid.py --out runs/luna_grid --scenarios software_pilot,... --all-noise --seeds 3
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent_telephone import runner_api
from agent_telephone.backends.codex_cli import CodexBackend
from agent_telephone.schema import Condition, Scenario

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("cc_pilot_script", ROOT / "scripts" / "cc_pilot.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

FIELDS = ["run_id", "seed", "scenario", "topology", "format", "noise", "model", "valid", "problems", "messages",
          "duration_s", "input_tokens", "output_tokens", "cache_read", "handoff_out_tokens", "handoff_chars"]


def one(backend, scenario_dir: str, noise: str, topology: str, fmt: str, model: str, seed: int, out: Path) -> dict:
    sc = Scenario.load(ROOT / scenario_dir)
    run_id = cc.run_name(scenario_dir, noise, topology, fmt, model, seed)
    row = {"run_id": run_id, "scenario": sc.id, "topology": topology, "format": fmt, "noise": noise, "model": model,
           "seed": seed}
    cond = Condition(topology=topology, model=model, handoff_format=fmt, noise=noise, seed=seed)
    t0 = time.monotonic()
    try:
        r = runner_api.run(sc, cond, backend, out / "messages.jsonl", run_id=run_id)
    except Exception as e:  # keep the grid going; the row records the failure (resume retries it)
        return {**row, "valid": False, "problems": f"{type(e).__name__}: {str(e)[:300]}"}
    problems = [f"empty message {m.sender}->{m.receiver}" for m in r.messages if not m.text.strip()]
    agent = [m for m in r.messages if m.model != "human"]
    row.update(valid=not problems, problems="; ".join(problems), messages=len(r.messages),
               duration_s=round(time.monotonic() - t0, 1),
               input_tokens=sum(m.usage.input_tokens for m in agent),
               output_tokens=sum(m.usage.output_tokens for m in agent),
               cache_read=sum(m.usage.cache_read_tokens for m in agent),
               handoff_out_tokens=sum(m.usage.output_tokens for m in agent if m.receiver != runner_api.ENDPOINT),
               handoff_chars=sum(len(m.text) for m in r.messages if m.receiver != runner_api.ENDPOINT))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/luna_grid")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenarios", default=",".join(cc.DEFAULT_SCENARIOS))
    ap.add_argument("--all-noise", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--reasoning-effort", default="medium")
    args = ap.parse_args()
    scenarios = cc.scenario_noises([s.strip() for s in args.scenarios.split(",") if s.strip()], args.all_noise)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = [(s, n, t, f, args.model, k) for k in range(args.seeds) for s, ns in scenarios.items()
            for n, t, f in itertools.product(ns, cc.TOPOLOGIES, cc.FORMATS)]
    if args.limit:
        grid = grid[: args.limit]
    csv_path = out / "runs.csv"
    done = {r["run_id"] for r in csv.DictReader(csv_path.open()) if r.get("valid") == "True"} if csv_path.exists() else set()
    backend = CodexBackend(reasoning_effort=args.reasoning_effort)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, backend, *g, out) for g in grid if cc.run_name(*g) not in done]
        for fut in as_completed(futs):
            row = fut.result()
            print(f"{row['run_id']}: valid={row.get('valid')} t={row.get('duration_s')}s {row.get('problems', '')[:120]}",
                  flush=True)
            new = not csv_path.exists()
            with csv_path.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)


if __name__ == "__main__":
    main()
