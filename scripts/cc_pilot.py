"""Subscription pilot on Claude Code: run a grid of conditions and record list-price cost per run.

Claude Code reports, in every stream's final `result` event, what the run would have cost on the API
at list price (`modelUsage[*].costUSD`, `costBasis: list`), even when billed to the subscription.
That is what we use to estimate the API arm.

    uv run python scripts/cc_pilot.py --out runs/cc_pilot --workers 4
    uv run python scripts/cc_pilot.py --out runs/cc_grid --scenarios software_2,event_3

Each scenario runs clean plus its ASR-name noise variant(s), as in the first pilot.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent_telephone import runner_cc
from agent_telephone.schema import Condition, Scenario

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIOS = ["software_pilot", "event_pilot"]
TOPOLOGIES = ["hier_chain", "hub_relay"]
FORMATS = ["free", "structured", "forward"]
MODELS = ["haiku", "sonnet"]
FIELDS = ["run_id", "seed", "scenario", "topology", "format", "noise", "model", "valid", "problems", "messages", "duration_s",
          "num_turns", "cost_usd_list", "cost_basis", "models_used", "input_tokens", "output_tokens", "cache_read",
          "cache_write", "handoff_out_tokens", "handoff_chars"]


def final_result(stream: Path) -> dict:
    res = {}
    for line in stream.read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev.get("type") == "result":
            res = ev
    return res


def scenario_noises(names: list[str], all_noise: bool = False) -> dict[str, list[str]]:
    """{scenario dir: noise levels} for scenario names (dirs under scenarios/) or paths: clean + the
    asr_name variant(s), or every noise variant with all_noise (the full grid)."""
    out = {}
    for name in names:
        sdir = name if "/" in name else f"scenarios/{name}"
        sc = Scenario.load(ROOT / sdir)  # relative paths are relative to the repo
        if sc.id != Path(sdir).name:  # run ids and the resume check use the dir name
            raise SystemExit(f"{sdir}: scenario id {sc.id!r} differs from its directory name")
        out[sdir] = ["clean", *(n.id for n in sc.noise if all_noise or n.kind == "asr_name")]
    return out


SCENARIOS = scenario_noises(DEFAULT_SCENARIOS)


def run_name(scenario_dir: str, noise: str, topology: str, fmt: str, model: str, seed: int) -> str:
    """Replicate 0 keeps the pilot's id; later replicates get an -s<k> suffix."""
    return f"{Path(scenario_dir).name}-{topology}-{fmt}-{noise}-{model}" + (f"-s{seed}" if seed else "")


def one(scenario_dir: str, noise: str, topology: str, fmt: str, model: str, seed: int, out: Path) -> dict:
    sc = Scenario.load(scenario_dir)
    cond = Condition(topology=topology, model=model, handoff_format=fmt, noise=noise)
    run_id = run_name(scenario_dir, noise, topology, fmt, model, seed)
    row = {"run_id": run_id, "scenario": sc.id, "topology": topology, "format": fmt, "noise": noise, "model": model,
           "seed": seed}
    try:
        r = runner_cc.run(sc, cond, out / "messages.jsonl", run_id=run_id, timeout=1800)
    except Exception as e:  # keep the grid going; the row records the failure
        return {**row, "valid": False, "problems": f"{type(e).__name__}: {e}"}
    res = final_result(r.stream_path)
    mu = res.get("modelUsage") or {}
    row.update(valid=r.valid, problems="; ".join(r.problems), messages=len(r.messages),
               duration_s=round((res.get("duration_ms") or 0) / 1000, 1), num_turns=res.get("num_turns"),
               cost_usd_list=res.get("total_cost_usd"),
               cost_basis=",".join(sorted({str(v.get("costBasis")) for v in mu.values()})),
               models_used=",".join(sorted(mu)),
               input_tokens=sum(v.get("inputTokens", 0) for v in mu.values()),
               output_tokens=sum(v.get("outputTokens", 0) for v in mu.values()),
               cache_read=sum(v.get("cacheReadInputTokens", 0) for v in mu.values()),
               cache_write=sum(v.get("cacheCreationInputTokens", 0) for v in mu.values()),
               handoff_out_tokens=sum(m.usage.output_tokens for m in r.messages),
               handoff_chars=sum(len(m.text) for m in r.messages))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/cc_pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N conditions (smoke test)")
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS),
                    help="comma-separated scenario names (dirs under scenarios/) or paths; default: the two pilots")
    ap.add_argument("--all-noise", action="store_true", help="every noise variant (homophone, punct too), not just ASR")
    ap.add_argument("--seeds", type=int, default=1, help="replicates per condition")
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    scenarios = scenario_noises([s.strip() for s in args.scenarios.split(",") if s.strip()], args.all_noise)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = [(s, n, t, f, m, k) for k in range(args.seeds) for s, ns in scenarios.items()
            for n, t, f, m in itertools.product(ns, TOPOLOGIES, FORMATS, models)]
    if args.limit:
        grid = grid[: args.limit]
    done = set()
    csv_path = out / "runs.csv"
    if csv_path.exists():  # resumable: skip runs already recorded as valid
        done = {r["run_id"] for r in csv.DictReader(csv_path.open()) if r.get("valid") == "True"}
    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, *g, out): g for g in grid if run_name(*g) not in done}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            print(f"{row['run_id']}: valid={row.get('valid')} cost=${row.get('cost_usd_list')} "
                  f"t={row.get('duration_s')}s {row.get('problems', '')[:120]}", flush=True)
            new = not csv_path.exists()
            with csv_path.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)


if __name__ == "__main__":
    main()
