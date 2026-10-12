"""End-to-end pilot grid: run -> log -> score -> endpoint -> summary.

    uv run python -m agent_telephone.pilot --backend fake

Phase 0 (`--backend fake`) runs both pilot scenarios x 3 hand-off formats x (clean + every
noise variant) x 2 topologies through the API runner with a seeded `FakeTeam`, scores every
message (deterministic + fake judge, both views), evaluates every endpoint artifact (hidden
tests in a temp dir / plan checks) and writes `<out>/summary.md` plus CSVs. The numbers are
synthetic: they check the pipeline, not any hypothesis.

`FakeTeam` is the FakeBackend with seeded random distortions (drop / accent drop / negation
flip / invent) and an endpoint author that starts from the scenario's reference artifact and
applies the mutation (`fake_endpoint.yaml`) of every fact that reached it non-preserved.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from . import analysis, endpoint, runner_api
from .backends.base import Completion
from .backends.fake import Distortion, FakeBackend, FakeConfig, expected_labels
from .backends.fake_judge import FakeJudgeBackend
from .schema import Condition, Message, Scenario
from .scoring import ScoreRow, save_jsonl, score_messages
from .scoring.agreement import kappa_between
from .scoring.deterministic import fold
from .scoring.judge import Judge

ROOT = Path(__file__).resolve().parent.parent
PILOTS = [ROOT / "scenarios" / "software_pilot", ROOT / "scenarios" / "event_pilot"]
FORMATS = ("free", "structured", "forward")
TOPOLOGIES = ("hier_chain", "hub_relay")
INVENT = "The client also wants a weekly summary sent to the whole team."
AGENT_MODEL, JUDGE_MODEL = "fake", "fake-judge"


# --- fake team ---------------------------------------------------------------------------

def fake_distortions(scenario: Scenario) -> list[Distortion]:
    """A plausible lossy team: asides and reasons dropped most, accents and polarity sometimes."""
    out = [Distortion(op="invent", text=INVENT, p=0.15)]
    for f in scenario.facts:
        if f.type in ("low_salience", "rationale"):
            out.append(Distortion(op="drop_fact", fact_id=f.id, p=0.2 if f.type == "low_salience" else 0.15))
        elif f.type in ("negation", "conditional"):
            out.append(Distortion(op="flip_negation", fact_id=f.id, p=0.05))
        elif f.type == "priority":
            out.append(Distortion(op="drop_fact", fact_id=f.id, p=0.08))
        elif f.type == "name" and f.check.kind in ("exact", "fuzzy") and f.check.pattern \
                and (new := fold(f.check.pattern)) != f.check.pattern.casefold():
            out.append(Distortion(op="swap_name", fact_id=f.id, new=new, p=0.08))
    return out


def _set_path(obj, path: str, value) -> None:
    *keys, last = path.split(".")
    for k in keys:
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    obj[int(last) if isinstance(obj, list) else last] = value


def fake_artifact(scenario_dir: Path, broken: set[str]) -> str:
    """The reference artifact with the mutation of every broken fact applied."""
    spec = yaml.safe_load((scenario_dir / "fake_endpoint.yaml").read_text(encoding="utf-8"))
    ref = (scenario_dir / spec["reference"]).read_text(encoding="utf-8")
    muts = [m for fid, m in spec["mutants"].items() if fid in broken]
    if spec["reference"].endswith(".json"):
        plan = copy.deepcopy(json.loads(ref))
        for m in muts:
            _set_path(plan, *m["set"])
        return json.dumps(plan, ensure_ascii=False, indent=2)
    for m in muts:
        ref = ref.replace(*m["replace"])
    return ref


class FakeTeam(FakeBackend):
    def __init__(self, scenario: Scenario, scenario_dir: Path, noise: str, seed: int):
        super().__init__(scenario, FakeConfig(distortions=fake_distortions(scenario)), seed=seed)
        self.scenario_dir, self.noise = scenario_dir, noise
        self.endpoint_instruction = runner_api.ENDPOINT_INSTRUCTIONS[scenario.domain][0]

    def complete(self, system: str, messages: list[dict], model: str, max_tokens: int) -> Completion:
        c = super().complete(system, messages, model, max_tokens)
        if self.endpoint_instruction not in messages[-1]["content"]:
            return c
        final = expected_labels(self.scenario, self.applied, self.noise)[0][-1]
        broken = {fid for fid, label in final.items() if label != "preserved"}
        return Completion(text=fake_artifact(self.scenario_dir, broken), usage=c.usage)


# --- grid --------------------------------------------------------------------------------

def run_grid(scenario_dirs: list[Path], out: Path, timeout: float = 120) -> dict:
    log_dir = out / "log"
    shutil.rmtree(log_dir, ignore_errors=True)
    scenarios, messages, truth, endpoints = {}, [], {}, []
    for sdir in scenario_dirs:
        sc = Scenario.load(sdir)
        scenarios[sc.id] = sc
        cells = itertools.product(TOPOLOGIES, FORMATS, ["clean", *(n.id for n in sc.noise)])
        for seed, (topo, fmt, noise) in enumerate(cells):
            cond = Condition(topology=topo, model=AGENT_MODEL, handoff_format=fmt, noise=noise, seed=seed)
            team = FakeTeam(sc, sdir, noise, seed)
            res = runner_api.run(sc, cond, team, log_dir / "messages.jsonl", run_id=f"{sc.id}-{topo}-{fmt}-{noise}")
            messages += res.messages
            truth[res.run_id] = (res.messages, expected_labels(sc, team.applied, noise)[0])
            ep = endpoint.evaluate(sc, sdir, res.artifact_path, timeout)
            endpoints.append({"run_id": res.run_id, "scenario": sc.id, "topology": topo, "handoff_format": fmt,
                              "noise": noise, "passed": ep.passed, "total": ep.total, "score": ep.score,
                              "error": ep.error or "", "facts": ep.facts})
    return {"scenarios": scenarios, "messages": messages, "truth": truth, "endpoints": endpoints}


def score_grid(scenarios: dict[str, Scenario], messages: list[Message]) -> dict:
    out = {}
    for view in ("note", "full"):
        judge = Judge(FakeJudgeBackend(inventions=[INVENT]), JUDGE_MODEL, [AGENT_MODEL])
        out[view] = score_messages(scenarios, messages, judge, view=view)
    second = Judge(FakeJudgeBackend(inventions=[INVENT], noise_p=0.05, seed=1), JUDGE_MODEL, [AGENT_MODEL])
    out["note_pass1"] = score_messages(scenarios, messages, second, view="note", judge_pass=1)
    return out


# --- summary -----------------------------------------------------------------------------

def truth_agreement(scores: list[ScoreRow], truth: dict) -> list[dict]:
    """Scorer label vs injected ground truth, per method, on source + hand-off messages (note view)."""
    gt = {}
    for run_id, (msgs, labels) in truth.items():
        for m, lab in zip(msgs, labels):
            gt.update({(run_id, m.distance, m.sender, m.receiver, fid): v for fid, v in lab.items()})
    agg = defaultdict(lambda: [0, 0, 0])
    for r in scores:
        if r.kind == "endpoint":
            continue
        a = agg[r.method]
        a[0] += 1
        if r.label == "unscored":
            a[2] += 1
        elif r.label == gt[r.key]:
            a[1] += 1
    return [{"method": k, "units": n, "agree": ok, "unscored": u, "accuracy": ok / (n - u) if n > u else None}
            for k, (n, ok, u) in sorted(agg.items())]


def noise_affected(scenario: Scenario, noise: str) -> set[str]:
    return {f for n in scenario.noise if n.id == noise for f in n.affected_fact_ids}


def h7_eligible(scenario: Scenario, ep: dict, with_noise: bool = False) -> set[str]:
    """Facts of one endpoint evaluation that can show H7: not loss-insensitive, and not failed
    only because a fact they depend on (universal / `depends`) failed. Forward runs: none (the
    endpoint author read the attached original, so a note-view `lost` is not a loss). Facts the
    run's noise variant corrupts in the source are excluded unless `with_noise` (their wrong
    reading propagates by design; `h7_by_fact` reports them flagged)."""
    if ep["handoff_format"] == "forward":
        return set()
    spec, facts = scenario.endpoint, ep["facts"]
    noisy = set() if with_noise else noise_affected(scenario, ep.get("noise", "clean"))
    if any(not facts.get(u, True) for u in spec.universal):
        return {u for u in spec.universal if u in facts} - noisy
    return {f for f in facts if f not in spec.loss_insensitive and f not in noisy
            and all(facts.get(d, True) for d in spec.depends.get(f, []))}


def _last_downward(scores: list[ScoreRow]) -> dict[tuple[str, str], tuple[int, str]]:
    last = {}
    for r in scores:
        if r.kind == "handoff" and r.direction == "down" and r.distance >= last.get((r.run_id, r.fact_id), (-1, None))[0]:
            last[(r.run_id, r.fact_id)] = (r.distance, r.label)
    return last


def endpoint_vs_last_handoff(scores: list[ScoreRow], endpoints: list[dict], scenarios: dict[str, Scenario]) -> list[dict]:
    """H7: label of a fact in the last downward hand-off before the endpoint vs its endpoint check
    (noise-affected facts of noisy runs excluded, see `h7_eligible`)."""
    last = _last_downward(scores)
    agg = defaultdict(lambda: [0, 0])
    for ep in endpoints:
        for fid in h7_eligible(scenarios[ep["scenario"]], ep):
            if (ep["run_id"], fid) not in last:  # not scored at the hand-off (judge fact, no judge)
                continue
            label = last[(ep["run_id"], fid)][1]
            agg[label][0] += 1
            agg[label][1] += ep["facts"][fid]
    return [{"label_at_last_handoff": k, "facts": n, "endpoint_pass_rate": p / n} for k, (n, p) in sorted(agg.items())]


def h7_by_fact(scores: list[ScoreRow], endpoints: list[dict], scenarios: dict[str, Scenario]) -> list[dict]:
    """H7 per fact x label with n; noise-affected facts of noisy runs kept, flagged `noise_affected`."""
    last = _last_downward(scores)
    agg = defaultdict(lambda: [0, 0])
    for ep in endpoints:
        sc = scenarios[ep["scenario"]]
        noisy = noise_affected(sc, ep.get("noise", "clean"))
        for fid in h7_eligible(sc, ep, with_noise=True):
            if (ep["run_id"], fid) in last:
                a = agg[(fid, last[(ep["run_id"], fid)][1], fid in noisy)]
                a[0] += 1
                a[1] += ep["facts"][fid]
    return [{"fact_id": f, "label_at_last_handoff": lab, "noise_affected": na, "facts": n, "passed": p,
             "endpoint_pass_rate": p / n} for (f, lab, na), (n, p) in sorted(agg.items())]


def endpoint_table(endpoints: list[dict], by: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for e in endpoints:
        groups[tuple(e[k] for k in by)].append(e)
    return [{**dict(zip(by, k)), "runs": len(v), "mean_score": sum(e["score"] for e in v) / len(v),
             "all_passed": sum(e["passed"] == e["total"] for e in v), "errors": sum(bool(e["error"]) for e in v)}
            for k, v in sorted(groups.items())]


def write_summary(grid: dict, scored: dict, out: Path) -> Path:
    scenarios, messages, endpoints = grid["scenarios"], grid["messages"], grid["endpoints"]
    note, note_inv = scored["note"]
    md = analysis.markdown_table
    rate_cols = ["scored", "survival", "lost_rate", "distorted_rate", "inverted_rate"]
    main = analysis.primary(note)
    by_format = analysis.survival([s for s in main if s.kind == "handoff"], ("handoff_format",))
    by_type = analysis.survival([s for s in main if s.kind == "handoff"], ("fact_type",))
    judge = [r for r in note if r.method == "judge"]
    kappa = kappa_between(judge, [r for r in scored["note_pass1"][0] if r.method == "judge"])
    lines = [
        "# agent-telephone — Phase 0 pilot (fake backend)", "",
        "**Synthetic data.** Agents are a seeded `FakeTeam` and the judge is `FakeJudgeBackend`: these tables "
        "check that runner, log, scoring, endpoint evaluation and analysis fit together. They say nothing about "
        "the hypotheses (the fake has no format or topology effect by construction).", "",
        f"- {len(grid['truth'])} runs ({', '.join(scenarios)} x {len(TOPOLOGIES)} topologies x {len(FORMATS)} formats "
        f"x clean + noise variants), {len(messages)} messages, {len(note)} scored units per view, "
        f"{len(endpoints)} endpoint evaluations.",
        "- Views: `note` = what the agent wrote; `full` = what the receiver saw (incl. a forwarded original).", "",
        "## Instrument check: scorer vs injected ground truth (note view, source + hand-offs)", "",
        md(truth_agreement(note, grid["truth"])),
        f"Judge pass 0 vs pass 1 (noise_p=0.05): κ = {kappa.kappa:.3f} (observed {kappa.observed:.3f}, n = {kappa.n}).", "",
        "Expected judge gaps: `FakeJudgeBackend` cannot emit `distorted`, so noise-affected judge facts come out "
        "`lost`; an accent drop inside a name also breaks the literal match of judge facts that quote it.", "",
        "## Survival by distance (note view; downward, non-forward hand-offs)", "",
        md(analysis.survival(main), ["kind", "distance", *rate_cols]),
        "## Survival by information distance (note view; downward, non-forward hand-offs)", "",
        md(analysis.survival([s for s in main if s.kind == "handoff"], distance_key="info_distance"),
           ["kind", "info_distance", *rate_cols]),
        "## Upward hand-backs (note view; reported separately, not loss)", "",
        md(analysis.survival(analysis.upward(note), ("topology",)), ["topology", "distance", *rate_cols]),
        "## Hand-off survival by format (note view; free vs structured only)", "",
        md(by_format, ["handoff_format", "distance", *rate_cols]),
        "## Forward notes: contradiction of the attached original (note view)", "",
        md(analysis.forward_contradiction(note)),
        "## Hand-off survival by fact type (note view; downward, non-forward)", "",
        md(by_type, ["fact_type", "distance", *rate_cols]),
        "## Invented claims by distance (note view)", "",
        md(analysis.invented_by_distance(note_inv, messages)),
        "## Noise propagation classes (note view)", "",
        md(analysis.noise_summary(analysis.noise_classes(scenarios, messages, note))),
        "## Endpoint by scenario and noise", "",
        md(endpoint_table(endpoints, ("scenario", "noise"))),
        "## Endpoint by scenario and topology", "",
        md(endpoint_table(endpoints, ("scenario", "topology"))),
        "## H4: endpoint by scenario and hand-off format", "",
        md(endpoint_table(endpoints, ("scenario", "handoff_format"))),
        "## H7 plumbing: fact label at the last hand-off vs its endpoint check (note view)", "",
        md(endpoint_vs_last_handoff(note, endpoints, scenarios)),
        "Excluded: forward runs, loss-insensitive facts, and facts whose check failed only because a fact it "
        "depends on failed. A preserved fact can still fail: hidden tests tagged with several facts fail for all "
        "of them, and the endpoint author can drop a fact itself.", "",
    ]
    path = out / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agent_telephone.pilot", description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=["fake"], default="fake", help="only the Phase 0 fake backend for now")
    ap.add_argument("--scenario", action="append", type=Path, help="scenario dir (repeatable; default: both pilots)")
    ap.add_argument("--out", type=Path, help="output dir (default: runs/<backend>)")
    ap.add_argument("--timeout", type=float, default=120, help="hidden-test timeout per artifact (s)")
    args = ap.parse_args(argv)
    out = args.out or Path("runs") / args.backend
    out.mkdir(parents=True, exist_ok=True)

    grid = run_grid(args.scenario or PILOTS, out, args.timeout)
    scored = score_grid(grid["scenarios"], grid["messages"])
    for view in ("note", "full"):
        rows, inv = scored[view]
        save_jsonl(rows, out / f"scores_{view}.jsonl")
        save_jsonl(inv, out / f"invented_{view}.jsonl")
        analysis.analyse(grid["scenarios"], grid["messages"], rows, inv, out / f"analysis_{view}")
    analysis.write_csv([{k: v for k, v in e.items() if k != "facts"} for e in grid["endpoints"]], out / "endpoints.csv")
    analysis.write_csv([{"run_id": e["run_id"], "fact_id": f, "passed": ok} for e in grid["endpoints"]
                        for f, ok in e["facts"].items()], out / "endpoint_facts.csv")
    print(f"{len(grid['truth'])} runs -> {write_summary(grid, scored, out)}")


if __name__ == "__main__":
    main()
