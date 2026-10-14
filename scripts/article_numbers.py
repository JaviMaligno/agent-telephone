"""Recompute every number quoted in the article "Nobody tells the implementer why" from the published
data (runs/cc_grid, runs/luna_grid). Prints claim -> value, so each figure in the text can be checked.

    uv run python scripts/article_numbers.py
"""
from __future__ import annotations

import collections
import csv
import gzip
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GRIDS = {"claude": ROOT / "runs/cc_grid", "luna": ROOT / "runs/luna_grid"}
ATTACH = "----- ORIGINAL BRIEF (verbatim) -----"


def scores(g: str, pass_: str = "scores_note"):
    with gzip.open(GRIDS[g] / "scored_judge" / f"{pass_}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def pct(a: int, n: int) -> str:
    return f"{100 * a / n:.1f}% ({a}/{n})" if n else "n=0"


def section(title: str) -> None:
    print(f"\n## {title}")


def runs_and_attempts():
    section("Runs")
    for g, d in GRIDS.items():
        rows = list(csv.DictReader((d / "runs.csv").open()))
        latest = {}
        for r in rows:
            latest[r["run_id"]] = r
        print(f"{g}: valid runs {sum(r['valid'] == 'True' for r in latest.values())} of {len(latest)} conditions")
    rows = list(csv.DictReader((GRIDS["claude"] / "runs.csv").open()))
    first = {}
    for r in rows:
        first.setdefault(r["run_id"], r)
    sonnet = [r for r in first.values() if r["model"] == "sonnet"]
    bad = [r for r in sonnet if r["valid"] == "False"]
    print("Sonnet first attempts with a double delegation:", pct(len(bad), len(sonnet)),
          "| of which event scenarios:", sum(r["scenario"].startswith("event") for r in bad))


def survival_by_type():
    section("Hand-off survival at hop 4 by fact type (downward, prose+list, note view)")
    for g in GRIDS:
        agg = collections.defaultdict(lambda: [0, 0])
        for r in scores(g):
            c = r["condition"]
            if (r.get("judge_pass", 0) or r.get("direction") != "down" or c["handoff_format"] == "forward"
                    or r["label"] == "unscored" or r["receiver"] == "endpoint" or r["sender"] == "user" or r["distance"] != 4):
                continue
            a = agg[r["fact_type"]]
            a[0] += 1
            a[1] += r["label"] == "preserved"
        print(g, {t: pct(p, n) for t, (n, p) in sorted(agg.items())})


def rationale_by_edge():
    section("Rationale kept, per message (downward, prose+list)")
    for g in GRIDS:
        agg = collections.defaultdict(lambda: [0, 0])
        for r in scores(g):
            c = r["condition"]
            if (r.get("judge_pass", 0) or r["fact_type"] != "rationale" or r.get("direction") != "down"
                    or c["handoff_format"] == "forward" or r["label"] == "unscored" or r["receiver"] == "endpoint"
                    or r["sender"] == "user"):
                continue
            a = agg[(c["topology"], f"{r['sender']}->{r['receiver']}", c["model"])]
            a[0] += 1
            a[1] += r["label"] == "preserved"
        for k, (n, p) in sorted(agg.items()):
            print(g, k, pct(p, n))


def rationale_effect():
    section("Rule obeyed in the artifact when its rationale reached the last hand-off vs not (exploratory)")
    pairs = {}
    for f in (ROOT / "scenarios").glob("*/scenario.yaml"):
        facts = yaml.safe_load(f.read_text())["facts"]
        for prev, cur in zip(facts, facts[1:]):
            if cur["type"] == "rationale":
                pairs[cur["id"]] = prev["id"]
    for g in GRIDS:
        ep = {(r["run_id"], r["fact_id"]): r["passed"] in ("True", "1", "true")
              for r in csv.DictReader((GRIDS[g] / "scored_judge/endpoint_facts.csv").open())}
        last = {}
        for r in scores(g):
            c = r["condition"]
            if (r.get("judge_pass", 0) or r["fact_id"] not in pairs or r.get("direction") != "down"
                    or r["receiver"] == "endpoint" or c["handoff_format"] == "forward"):
                continue
            k = (r["run_id"], r["fact_id"])
            if k not in last or r["distance"] > last[k][0]:
                last[k] = (r["distance"], r["label"], c["handoff_format"], c["model"])
        strata = collections.defaultdict(lambda: ([], []))
        for (rid, rat), (_, lab, fmt, model) in last.items():
            if (rid, pairs[rat]) in ep:
                strata[(pairs[rat], fmt, model)][0 if lab == "preserved" else 1].append(ep[(rid, pairs[rat])])
        strata = {k: v for k, v in strata.items() if v[0] and v[1]}

        def est(s):
            num = den = 0.0
            for kept, dropped in s.values():
                w = len(kept) * len(dropped) / (len(kept) + len(dropped))
                num += w * (sum(kept) / len(kept) - sum(dropped) / len(dropped))
                den += w
            return num / den
        rng = random.Random(0)
        boot = sorted(est({k: ([rng.choice(a) for _ in a], [rng.choice(b) for _ in b]) for k, (a, b) in strata.items()})
                      for _ in range(2000))
        print(f"{g}: diff {est(strata):+.3f}, 95% CI [{boot[50]:+.3f}, {boot[1949]:+.3f}], strata {len(strata)}")
        if g == "claude":
            seen = strata.get(("f_seen_field", "free", "haiku"), ([], []))
            both = [x for fmt in ("free", "structured") for x in [strata.get(("f_seen_field", fmt, "haiku"))] if x]
            kept = [v for k, _ in both for v in k]
            dropped = [v for _, d in both for v in d]
            print("  Haiku, f_seen_field (last_seen_at): kept", pct(sum(kept), len(kept)), "dropped", pct(sum(dropped), len(dropped)))


def h7():
    section("H7: fact label at the last hand-off vs endpoint check (summary.md)")
    for g in GRIDS:
        text = (GRIDS[g] / "scored_judge/summary.md").read_text()
        block = text.split("## H7")[1].split("Per fact")[0]
        print(g, [l for l in block.splitlines() if l.startswith("| ") and "label_at" not in l])


def noise():
    section("Noise classes, hand-offs and endpoints (summary.md)")
    for g in GRIDS:
        text = (GRIDS[g] / "scored_judge/summary.md").read_text()
        block = text.split("## Noise propagation")[1].split("\n## ")[0]
        tot = collections.defaultdict(lambda: collections.Counter())
        for line in block.splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 9 and cells[0] in ("asr_name", "homophone", "punct"):
                kind, where = cells[0], cells[1]
                for name, v in zip(["flagged", "freeze", "amplify", "repair", "lost", "drift"], cells[3:]):
                    tot[(kind, where)][name] += int(v)
        for (kind, where), c in sorted(tot.items()):
            n = sum(c.values())
            carried = c["flagged"] + c["freeze"] + c["amplify"]
            extra = f" flagged/carried {pct(c['flagged'], carried)}" if kind == "asr_name" else ""
            print(g, kind, where, dict(c), f"repair {pct(c['repair'], n)} freeze {pct(c['freeze'], n)}{extra}")


def formats_and_contract():
    section("Hop-4 survival, all non-rationale-specific facts, prose vs list")
    for g in GRIDS:
        agg = collections.defaultdict(lambda: [0, 0])
        for r in scores(g):
            c = r["condition"]
            if (r.get("judge_pass", 0) or r.get("direction") != "down" or c["handoff_format"] == "forward"
                    or r["label"] == "unscored" or r["receiver"] == "endpoint" or r["distance"] != 4):
                continue
            a = agg[c["handoff_format"]]
            a[0] += 1
            a[1] += r["label"] == "preserved"
        print(g, {k: pct(p, n) for k, (n, p) in agg.items()})
    section("Software endpoint mean by model and format; input contract at the implementer")
    marker = {"software_pilot": "customer_name", "software_2": "plan_name", "software_3": "closed_by"}
    for g, d in GRIDS.items():
        eps = {r["run_id"]: r for r in csv.DictReader((d / "scored_judge/endpoints.csv").open()) if r["scenario"] in marker}
        agg = collections.defaultdict(list)
        for r in eps.values():
            agg[(r["model"], r["handoff_format"])].append(float(r["score"]))
        print(g, {k: f"{sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(agg.items())})
        last = {}
        for line in (d / "messages.jsonl").open():
            m = json.loads(line)
            if m["run_id"] in eps and m["receiver"] == "implementer" and m.get("direction") == "down":
                if m["run_id"] not in last or m["distance"] > last[m["run_id"]]["distance"]:
                    last[m["run_id"]] = m
        has = collections.defaultdict(lambda: [0, 0])
        for rid, m in last.items():
            fmt = m["condition"]["handoff_format"]
            text = m["text"] if fmt == "forward" else m["text"].split(ATTACH)[0]
            a = has[(eps[rid]["model"], fmt)]
            a[0] += 1
            a[1] += marker[m["scenario"]] in text
        print(g, "contract reached implementer:", {k: pct(p, n) for k, (n, p) in sorted(has.items())})


def no_code():
    section("Software artifacts with no code at all")
    import re
    for g, d in GRIDS.items():
        c = collections.Counter()
        n = collections.Counter()
        for r in csv.DictReader((d / "scored_judge/endpoints.csv").open()):
            if not r["scenario"].startswith("software"):
                continue
            art = (d / f"{r['run_id']}.artifact.py").read_text(encoding="utf-8")
            n[r["model"]] += 1
            c[r["model"]] += not re.search(r"^\s*(def|import|from|class)\b", art, re.M)
        print(g, {m: f"{c[m]}/{n[m]}" for m in n})


def judges():
    section("Judge retest agreement and excluded-vendor bookings")
    for g, d in GRIDS.items():
        print(g, (d / "scored_judge/summary.md").read_text().splitlines()[3])
        for r in csv.DictReader((d / "scored_judge/endpoint_facts.csv").open()):
            pass
    import importlib.util
    for scen, field, name in [("event_pilot", ("transport", "provider"), "coachline"), ("event_2", ("photographer",), "lumiere")]:
        spec = importlib.util.spec_from_file_location(scen, ROOT / "scenarios" / scen / "plan_checks/checks.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        booked = 0
        for d in GRIDS.values():
            for f in d.glob(f"{scen}-*.artifact.json"):
                try:
                    v = json.loads(f.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                for k in field:
                    v = v.get(k) if isinstance(v, dict) else None
                booked += mod.books(str(v), name)
        print(f"{scen}: plans booking the excluded vendor {booked}")


if __name__ == "__main__":
    runs_and_attempts()
    survival_by_type()
    rationale_by_edge()
    rationale_effect()
    h7()
    noise()
    formats_and_contract()
    no_code()
    judges()
