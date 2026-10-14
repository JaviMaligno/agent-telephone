"""Score real (Claude Code) runs: deterministic scoring, endpoint evaluation, optional judge.

    uv run python scripts/score_real.py                                    # no network, no cost
    uv run python scripts/score_real.py --judge gateway --judge-model gpt-5.6-sol-tst --dry-run

Inputs (`--dir`, default runs/cc_pilot): `runs.csv` (the latest row per run id decides validity),
`messages.jsonl` (latest attempt per run id), `<run_id>.artifact.*`. Only valid runs are scored.

- Deterministic scoring of every (fact, message) in both views (`note`, `full`); judge-only facts
  stay unscored without `--judge`.
- Endpoints: hidden tests / plan checks via `agent_telephone.endpoint` (sandboxed subprocess).
- Judge (`--judge gateway`): cross-family (Claude agents need a non-Claude judge), through
  `GatewayBackend` + `BudgetGuard`; passes: note + full view, plus a second note pass for κ
  (`--judge-passes 2`). Identical prompts within a pass are sent once (outside `forward` runs the
  full view is the note). `--dry-run` prints calls, tokens per call and cost from
  config/prices.yaml without sending anything; a real run refuses null prices and needs
  `--max-usd` >= the worst-case estimate.

Outputs (`--out`, default <dir>/scored): scores_<view>.jsonl, invented_<view>.jsonl,
endpoints.csv, endpoint_facts.csv, analysis_<view>/ (agent_telephone.analysis), summary.md,
judge_plan.json (with --judge).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import statistics
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from agent_telephone import analysis, endpoint
from agent_telephone.backends.base import Backend, Completion
from agent_telephone.budget import DEFAULT_PRICES, MILLION, estimate_input_tokens, load_prices
from agent_telephone.pilot import endpoint_table, endpoint_vs_last_handoff, h7_by_fact
from agent_telephone.schema import Message, MessageLog, Scenario, latest_attempts
from agent_telephone.scoring import InventedClaim, ScoreRow, message_kind, save_jsonl, score_messages, view_text
from agent_telephone.scoring.agreement import kappa_between
from agent_telephone.scoring.deterministic import needs_judge
from agent_telephone.scoring.judge import SYSTEM, CrossFamilyError, Judge, build_prompt

ROOT = Path(__file__).resolve().parent.parent
VIEWS = ("note", "full")
# Visible judge JSON per fact (id, label, evidence span) plus invented spans; generous on purpose.
OUT_TOKENS_PER_FACT, OUT_TOKENS_BASE = 60, 300


# --------------------------------------------------------------------------- loading

def load_runs(run_dir: Path) -> tuple[dict[str, dict], list[Message]]:
    """Valid runs (latest runs.csv row per run id) and their latest-attempt messages."""
    latest = {}
    with (run_dir / "runs.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            latest[row["run_id"]] = row
    valid = {k: r for k, r in latest.items() if r.get("valid") == "True"}
    messages = [m for m in latest_attempts(MessageLog(run_dir / "messages.jsonl").read()) if m.run_id in valid]
    if missing := set(valid) - {m.run_id for m in messages}:
        raise SystemExit(f"valid runs without messages: {sorted(missing)}")
    return valid, messages


def load_scenarios(messages: list[Message], scenarios_dir: Path) -> dict[str, Scenario]:
    return {sid: Scenario.load(scenarios_dir / sid) for sid in sorted({m.scenario for m in messages})}


# --------------------------------------------------------------------------- endpoints

def evaluate_endpoints(run_dir: Path, scenarios_dir: Path, scenarios: dict[str, Scenario], messages: list[Message],
                       timeout: float = 120) -> list[dict]:
    first = {}
    for m in messages:
        first.setdefault(m.run_id, m)
    out = []
    for run_id, m in sorted(first.items()):
        arts = sorted(run_dir.glob(f"{run_id}.artifact.*"))
        c = m.condition
        row = {"run_id": run_id, "scenario": m.scenario, "topology": c.topology, "handoff_format": c.handoff_format,
               "noise": c.noise, "model": c.model}
        if len(arts) != 1:
            out.append({**row, "passed": 0, "total": 0, "score": 0.0, "error": f"{len(arts)} artifacts", "consistency": "",
                        "facts": {}})
            continue
        ep = endpoint.evaluate(scenarios[m.scenario], scenarios_dir / m.scenario, arts[0], timeout)
        out.append({**row, "passed": ep.passed, "total": ep.total, "score": ep.score, "error": ep.error or "",
                    "consistency": ";".join(f"{k}={v}" for k, v in sorted(ep.consistency.items())),
                    "facts": ep.facts})
    return out


def endpoint_rates(endpoints: list[dict], by: tuple[str, ...]) -> list[dict]:
    """Checks passed / checks run (pooled) next to the per-run mean of `endpoint_table`."""
    pooled = defaultdict(lambda: [0, 0])
    for e in endpoints:
        p = pooled[tuple(e[k] for k in by)]
        p[0] += e["passed"]
        p[1] += e["total"]
    return [{**r, "checks_passed": pooled[k][0], "checks": pooled[k][1],
             "pass_rate": pooled[k][0] / pooled[k][1] if pooled[k][1] else None}
            for r, k in ((r, tuple(r[b] for b in by)) for r in endpoint_table(endpoints, by))]


# --------------------------------------------------------------------------- judge

class OncePerPrompt:
    """Backend wrapper: an identical request within one pass is sent once (full view == note view
    outside `forward` runs), so a second pass gets a fresh instance and genuinely resamples.
    With `store`, answers are appended to a JSONL file and reloaded, so a long run resumes."""

    def __init__(self, inner: Backend, store: Path | None = None):
        self.inner, self.cache, self.sent, self.store = inner, {}, 0, store
        self.lock = threading.Lock()
        if store and store.exists():
            for line in store.open(encoding="utf-8"):
                row = json.loads(line)
                self.cache[row["key"]] = Completion.model_validate(row["completion"])

    @staticmethod
    def key(system: str, messages: list[dict], model: str, max_tokens: int) -> str:
        return json.dumps([system, messages, model, max_tokens], ensure_ascii=False)

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> Completion:
        key = self.key(system, messages, model, max_tokens)
        if key not in self.cache:
            c = self.inner.complete(system, messages, model, max_tokens)
            with self.lock:
                self.cache[key] = c
                self.sent += 1
                if self.store:
                    with self.store.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"key": key, "completion": c.model_dump()}, ensure_ascii=False) + "\n")
        return self.cache[key]

    def prefetch(self, prompts: list[str], model: str, max_tokens: int, workers: int) -> None:
        """Send the first-attempt requests concurrently; scoring then reads them from the cache (only
        retries after invalid output are sent serially)."""
        todo = [p for p in dict.fromkeys(prompts)
                if self.key(SYSTEM, [{"role": "user", "content": p}], model, max_tokens) not in self.cache]
        def one(p):
            try:
                self.complete(SYSTEM, [{"role": "user", "content": p}], model, max_tokens)
            except Exception as e:  # left uncached: the serial scoring pass retries it
                print(f"prefetch failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            for i, _ in enumerate(ex.map(one, todo), 1):
                if i % 100 == 0 or i == len(todo):
                    print(f"prefetch {i}/{len(todo)}", flush=True)


class NoCalls:
    def complete(self, *a, **kw) -> Completion:
        raise RuntimeError("dry run: no judge calls")


def retest_sample(messages: list[Message], fraction: float) -> list[Message]:
    """Messages of a stable pseudo-random subset of runs (sha256 of the run id), for retest passes."""
    if fraction >= 1:
        return messages
    keep = lambda rid: int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction
    return [m for m in messages if keep(m.run_id)]


def judge_passes(n: int) -> list[tuple[str, int]]:
    """(view, pass) in run order; pass 0 of both views shares one prompt cache."""
    return [("note", 0), ("full", 0)] + [("note", p) for p in range(1, n)]


def judge_plan(scenarios: dict[str, Scenario], messages: list[Message], judge_model: str, n_passes: int,
               max_tokens: int, prices_path: Path = DEFAULT_PRICES, retest_fraction: float = 1.0) -> dict:
    """Exact call count and conservative tokens / cost of a judge run, without sending anything.
    Input tokens use budget.estimate_input_tokens (2 chars/token) on the exact prompts; visible
    output is estimated per fact; the worst case is max_tokens per call (reasoning included) and
    every call retried once (the judge retries invalid output once)."""
    try:
        Judge(NoCalls(), judge_model, {m.model for m in messages})
    except CrossFamilyError as e:
        raise SystemExit(f"refusing the judge: {e} (Claude agent runs need a non-Claude judge)") from None
    per_pass, calls = [], []
    seen: set[str] = set()
    for view, p in judge_passes(n_passes):
        if p > 0:
            seen = set()
        n = 0
        for m in (messages if p == 0 else retest_sample(messages, retest_fraction)):
            facts = [f for f in scenarios[m.scenario].facts if needs_judge(f)]
            if not facts:
                continue
            prompt = build_prompt(scenarios[m.scenario].brief, facts, view_text(m, view))
            if prompt in seen:
                continue
            seen.add(prompt)
            n += 1
            calls.append({"in": estimate_input_tokens(SYSTEM, [{"role": "user", "content": prompt}]),
                          "out": min(max_tokens, OUT_TOKENS_BASE + OUT_TOKENS_PER_FACT * len(facts)),
                          "kind": message_kind(m)})
        per_pass.append({"view": view, "pass": p, "calls": n})
    ins, outs = [c["in"] for c in calls], [c["out"] for c in calls]
    plan = {"judge_model": judge_model, "max_tokens": max_tokens, "passes": per_pass, "calls": len(calls),
            "calls_worst_with_retries": 2 * len(calls),
            "input_tokens_per_call": {"min": min(ins, default=0), "median": statistics.median(ins) if ins else 0,
                                      "max": max(ins, default=0), "total": sum(ins)},
            "output_tokens_per_call": {"expected_visible": statistics.median(outs) if outs else 0,
                                       "worst": max_tokens, "total_expected": sum(outs),
                                       "total_worst": max_tokens * len(calls)},
            # a retry resends the prompt plus the invalid reply and the error
            "worst_input_tokens_with_retries": 2 * sum(ins) + len(calls) * (max_tokens + 100),
            "worst_output_tokens_with_retries": 2 * max_tokens * len(calls)}
    price = load_prices(prices_path).get(judge_model)
    if not price or price.get("input") is None or price.get("output") is None:
        plan["cost_usd"] = None
        plan["cost_note"] = f"no price for {judge_model!r} in {prices_path}: fill it before any real run"
    else:
        pin, pout = price["input"], price["output"]

        def usd(i: int, o: int) -> float:
            return float((Decimal(i) * pin + Decimal(o) * pout) / MILLION)

        plan["cost_usd"] = {"expected": usd(sum(ins), sum(outs)),
                            "worst_no_retries": usd(sum(ins), max_tokens * len(calls)),
                            "worst_with_retries": usd(plan["worst_input_tokens_with_retries"],
                                                      plan["worst_output_tokens_with_retries"])}
    return plan


def judge_prompts(scenarios: dict[str, Scenario], messages: list[Message], view: str) -> list[str]:
    out = []
    for m in messages:
        facts = [f for f in scenarios[m.scenario].facts if needs_judge(f)]
        if facts:
            out.append(build_prompt(scenarios[m.scenario].brief, facts, view_text(m, view)))
    return out


def run_judge(scenarios: dict[str, Scenario], messages: list[Message], backend: Backend, judge_model: str,
              n_passes: int, max_tokens: int, workers: int = 1, cache_dir: Path | None = None,
              retest_fraction: float = 1.0) -> dict[tuple[str, int], tuple[list[ScoreRow], list[InventedClaim]]]:
    def store(p):
        return cache_dir / f"judge_cache_pass{p}.jsonl" if cache_dir else None
    out, cached = {}, OncePerPrompt(backend, store(0))
    for view, p in judge_passes(n_passes):
        if p > 0:
            cached = OncePerPrompt(backend, store(p))
        msgs = messages if p == 0 else retest_sample(messages, retest_fraction)
        if workers > 1:
            cached.prefetch(judge_prompts(scenarios, msgs, view), judge_model, max_tokens, workers)
        judge = Judge(cached, judge_model, {m.model for m in messages}, max_tokens=max_tokens)
        out[view, p] = score_messages(scenarios, msgs, judge, view=view, judge_pass=p)
    return out


def gateway_backend(args) -> Backend:
    from agent_telephone.backends.gateway import GatewayBackend
    from agent_telephone.budget import DEFAULT_LEDGER, BudgetGuard

    if not args.key_file:
        raise SystemExit("--key-file is required for --judge gateway")
    guard = BudgetGuard(args.budget_limit, args.ledger or DEFAULT_LEDGER, args.prices, init_ledger=args.init_ledger)
    return GatewayBackend(args.gateway_url, args.key_file, guard, reasoning_effort=args.reasoning_effort)


# --------------------------------------------------------------------------- summary

def endpoint_text_vs_check(scores: list[ScoreRow], endpoints: list[dict]) -> list[dict]:
    """Per fact: the text label of the endpoint artifact vs its functional check. Where they
    disagree the text instrument is reading format (code/JSON), not retention."""
    checks = {(e["run_id"], f): ok for e in endpoints for f, ok in e["facts"].items()}
    agg = defaultdict(lambda: [0, 0])
    for r in scores:
        if r.kind == "endpoint" and r.label != "unscored" and (r.run_id, r.fact_id) in checks:
            a = agg[(r.fact_id, r.label)]
            a[0] += 1
            a[1] += checks[(r.run_id, r.fact_id)]
    return [{"fact_id": f, "text_label": lab, "n": n, "check_passed": p} for (f, lab), (n, p) in sorted(agg.items())]


def usage_table(messages: list[Message]) -> list[dict]:
    groups = defaultdict(list)
    for m in messages:
        if message_kind(m) == "handoff":
            groups[(m.condition.model, m.direction or "down")].append(m.usage)
    return [{"model": k[0], "direction": k[1], "messages": len(us),
             "median_output_tokens": statistics.median(u.output_tokens for u in us),
             "total_output_tokens": sum(u.output_tokens for u in us),
             "estimated_share": sum(u.estimated for u in us) / len(us)} for k, us in sorted(groups.items())]


def write_summary(out: Path, runs: dict, messages: list[Message], scenarios: dict[str, Scenario],
                  scores: dict[str, list[ScoreRow]], invented: dict[str, list[InventedClaim]],
                  endpoints: list[dict], judge_note: str) -> Path:
    md, note = analysis.markdown_table, scores["note"]
    rate = ["scored", "survival", "lost_rate", "distorted_rate", "inverted_rate"]
    main = analysis.primary(note)
    handoffs = [r for r in main if r.kind == "handoff"]
    noise_rows = analysis.noise_classes(scenarios, messages, note)
    lines = [
        "# agent-telephone — Claude Code pilot, scored", "",
        f"- {len(runs)} valid runs, {len(messages)} messages, {len(note)} scored units (note view), "
        f"{len(endpoints)} endpoint evaluations.",
        f"- Judge: {judge_note}", "",
        "## Survival by distance (note view; source, downward non-forward hand-offs, endpoint)", "",
        md(analysis.survival(main), ["kind", "distance", *rate, "unscored"]),
        "## Survival by information distance (hand-offs, note view)", "",
        md(analysis.survival(handoffs, distance_key="info_distance"), ["kind", "info_distance", *rate]),
        "## Hand-off survival by fact type (note view; downward, non-forward)", "",
        md(analysis.survival(handoffs, ("fact_type",)), ["fact_type", "distance", *rate]),
        "## Hand-off survival by model (note view; downward, non-forward)", "",
        md(analysis.survival(handoffs, ("model",)), ["model", "distance", *rate]),
        "## Hand-off survival by format (note view; free vs structured)", "",
        md(analysis.survival(handoffs, ("handoff_format",)), ["handoff_format", "distance", *rate]),
        "## Upward hand-backs (note view)", "",
        md(analysis.survival(analysis.upward(note), ("topology",)), ["topology", "distance", *rate]),
        "## Forward notes: contradiction of the attached original (note view)", "",
        md(analysis.forward_contradiction(note, ("model",))),
        "## Noise propagation (asr variants, note view)", "",
        md(analysis.noise_summary(noise_rows)),
        "By model (all distances):", "",
        md(_noise_by(noise_rows, "model")),
        "## Endpoint by model and format", "",
        md(endpoint_rates(endpoints, ("scenario", "model", "handoff_format"))),
        "## Endpoint by model", "",
        md(endpoint_rates(endpoints, ("scenario", "model"))),
        "## Endpoint by topology and noise", "",
        md(endpoint_rates(endpoints, ("scenario", "topology", "noise"))),
        "Endpoint rows above are text labels of code/JSON artifacts; the functional checks below are the "
        "endpoint measure (see the text-vs-check table).", "",
        "## Endpoint artifact: text label vs functional check, per fact (note view)", "",
        md(endpoint_text_vs_check(note, endpoints)),
        "## Plan consistency checks (untagged, not in the score)", "",
        md([{"run_id": e["run_id"], "consistency": e["consistency"]} for e in endpoints if e.get("consistency")]),
        "## H7: fact label at the last downward hand-off vs its endpoint check (note view)", "",
        "Forward runs, loss-insensitive facts, facts failed only through a dependency and the noise-affected "
        "facts of noisy runs are excluded (their wrong reading propagates by design: see the per-fact table).", "",
        md(endpoint_vs_last_handoff(note, endpoints, scenarios)),
        "Per fact (noise-affected facts of noisy runs kept and flagged):", "",
        md(h7_by_fact(note, endpoints, scenarios)),
        "## Hand-off output tokens (per message, from the stream; see runner_cc.turn_usage)", "",
        md(usage_table(messages)),
    ]
    if invented.get("note"):
        lines += ["## Invented claims by distance (judge, note view)", "",
                  md(analysis.invented_by_distance(invented["note"], messages))]
    path = out / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _noise_by(rows: list[dict], key: str) -> list[dict]:
    groups = defaultdict(lambda: dict.fromkeys(analysis.NOISE_CLASSES, 0))
    for r in rows:
        if r["class"] in analysis.NOISE_CLASSES:
            groups[(r["noise"], r[key], r["kind"])][r["class"]] += 1
    return [{"noise": n, key: v, "kind": k, **c} for (n, v, k), c in sorted(groups.items())]


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=ROOT / "runs" / "cc_pilot")
    ap.add_argument("--scenarios", type=Path, default=ROOT / "scenarios")
    ap.add_argument("--out", type=Path, help="default: <dir>/scored")
    ap.add_argument("--timeout", type=float, default=120, help="hidden-test timeout per artifact (s)")
    ap.add_argument("--skip-endpoints", action="store_true")
    ap.add_argument("--judge", choices=["none", "gateway", "codex", "claude"], default="none",
                    help="codex = GPT on the ChatGPT subscription via codex exec; claude = Claude on the Claude "
                         "subscription via claude -p (for non-Claude agent runs); neither spends")
    ap.add_argument("--retest-fraction", type=float, default=1.0,
                    help="share of runs (stable hash of run id) judged again in retest passes (kappa)")
    ap.add_argument("--judge-workers", type=int, default=1, help="concurrent first-attempt judge calls")
    ap.add_argument("--judge-model", default="gpt-5.6-sol-tst")
    ap.add_argument("--judge-passes", type=int, default=2, help="note-view passes (2 = second pass for kappa)")
    ap.add_argument("--judge-max-tokens", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true", help="with --judge: print the plan and cost, send nothing")
    ap.add_argument("--max-usd", type=float, help="required for a real judge run: refuse if the worst case exceeds it")
    ap.add_argument("--gateway-url", default=os.environ.get("AGENT_TELEPHONE_GATEWAY_URL", ""))
    ap.add_argument("--key-file", help="file holding the gateway key (never printed)")
    ap.add_argument("--reasoning-effort")
    ap.add_argument("--budget-limit", default="400")
    ap.add_argument("--ledger")
    ap.add_argument("--init-ledger", action="store_true")
    ap.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    args = ap.parse_args(argv)
    out = args.out or args.dir / "scored"

    runs, messages = load_runs(args.dir)
    scenarios = load_scenarios(messages, args.scenarios)

    plan = None
    if args.judge != "none":
        plan = judge_plan(scenarios, messages, args.judge_model, args.judge_passes, args.judge_max_tokens, args.prices,
                          args.retest_fraction)
        out.mkdir(parents=True, exist_ok=True)
        (out / "judge_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
        print(json.dumps(plan, indent=1))
        if args.dry_run:
            return
        if args.judge in ("codex", "claude"):
            pass  # subscription: no spend to cap
        elif plan["cost_usd"] is None:
            raise SystemExit(f"refusing to run the judge: {plan['cost_note']}")
        elif args.max_usd is None or plan["cost_usd"]["worst_with_retries"] > args.max_usd:
            worst = plan["cost_usd"]["worst_with_retries"]
            raise SystemExit(f"refusing to run the judge: worst case ${worst:.2f} needs --max-usd >= that")

    scores, invented = {}, {}
    if plan is not None:
        if args.judge == "codex":
            from agent_telephone.backends.codex_cli import CodexBackend
            backend = CodexBackend(reasoning_effort=args.reasoning_effort or "medium")
        elif args.judge == "claude":
            from agent_telephone.backends.claude_cli import ClaudeCliBackend
            backend = ClaudeCliBackend(effort=args.reasoning_effort)
        else:
            backend = gateway_backend(args)
        out.mkdir(parents=True, exist_ok=True)
        judged = run_judge(scenarios, messages, backend, args.judge_model, args.judge_passes,
                           args.judge_max_tokens, args.judge_workers, out, args.retest_fraction)
        for (view, p), (rows, inv) in judged.items():
            name = view if p == 0 else f"{view}_pass{p}"
            scores[name], invented[name] = rows, inv
        judge_note = f"{args.judge_model} via {args.judge}, passes {[f'{v}/{p}' for v, p in judge_passes(args.judge_passes)]}"
        if "note_pass1" in scores:
            k = kappa_between([r for r in scores["note"] if r.method == "judge"],
                              [r for r in scores["note_pass1"] if r.method == "judge"])
            judge_note += f"; pass 0 vs 1 κ = {k.kappa:.3f} (n = {k.n})"
    else:
        for view in VIEWS:
            scores[view], invented[view] = score_messages(scenarios, messages, None, view=view)
        n_judge = {sid: sum(f.check.kind == "judge" for f in s.facts) for sid, s in scenarios.items()}
        judge_note = (f"not run (deterministic only): judge-only facts are not scored ({n_judge}); "
                      "`also_judge` facts keep their deterministic label.")

    for name, rows in scores.items():
        save_jsonl(rows, out / f"scores_{name}.jsonl")
        save_jsonl(invented[name], out / f"invented_{name}.jsonl")
    for view in VIEWS:
        analysis.analyse(scenarios, messages, scores[view], invented[view], out / f"analysis_{view}")

    endpoints = [] if args.skip_endpoints else evaluate_endpoints(args.dir, args.scenarios, scenarios, messages,
                                                                  args.timeout)
    analysis.write_csv([{k: v for k, v in e.items() if k != "facts"} for e in endpoints], out / "endpoints.csv")
    analysis.write_csv([{"run_id": e["run_id"], "fact_id": f, "passed": ok} for e in endpoints
                        for f, ok in e["facts"].items()], out / "endpoint_facts.csv")
    path = write_summary(out, runs, messages, scenarios, scores, invented, endpoints, judge_note)
    print(f"{len(runs)} runs, {len(messages)} messages -> {path}")


if __name__ == "__main__":
    sys.exit(main())
