"""Blind manual-labelling sheet: a seeded, stratified sample of (fact, message) units from valid pilot runs.

Strata = model x scenario x fact type x distance bin (2, 3, 4, 5+). Units are drawn round-robin across
strata (one per stratum per round, strata in seeded order) so every stratum is covered before any gets
a second item. Only agent-written hand-offs are sampled (no source brief, no endpoint artifact), in the
note view; `forward` hand-offs are skipped by default (their notes are told not to restate the brief).
The sheet carries no deterministic or judge label. Fill `label`, then:

    uv run python scripts/make_label_sheet.py --runs runs/cc_pilot --n 150 --seed 0
    uv run python -m agent_telephone.scoring.agreement --sheet runs/cc_pilot/labels/sheet.csv \
        --log runs/cc_pilot/messages.jsonl [--scores judge_scores.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from agent_telephone.schema import Fact, Message, MessageLog, Scenario
from agent_telephone.scoring import message_kind, view_text
from agent_telephone.scoring.deterministic import canon, fold

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["item_id", "run_id", "distance", "sender", "receiver", "fact_id", "fact_type", "fact_text",
           "message_excerpt", "message_file", "label", "notes"]
STOP = set("the and for with that this from are was were but not only also then than into its it's have has "
           "will should must can our your their they them you all any each per one".split())
UNIT = re.compile(r"\n+|(?<=[.!?;])\s+")
EXCERPT_CHARS = 900


def valid_runs(runs_csv: Path) -> set[str]:
    with runs_csv.open(newline="", encoding="utf-8") as f:
        return {r["run_id"] for r in csv.DictReader(f) if r.get("valid") == "True"}


def distance_bin(d: int) -> str:
    return str(d) if d < 5 else "5+"


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_:-]+", fold(canon(s))) if (len(w) > 2 or w.isdigit()) and w not in STOP}


def excerpt(text: str, fact: Fact, k: int = 2) -> str:
    """Lexical hint: the k lines/sentences sharing most words with the fact text, +-1 unit of context.

    Located by word overlap only (never by the scorer's patterns), so it says nothing about the label."""
    units = [u.strip() for u in UNIT.split(text) if u.strip()]
    want = _tokens(fact.text)
    scored = sorted(((len(want & _tokens(u)), -i) for i, u in enumerate(units)), reverse=True)
    hits = [-i for s, i in scored[:k] if s > 0]
    if not hits:
        return "[no lexical match: read the full message]"
    keep = sorted({j for i in hits for j in range(max(0, i - 1), min(len(units), i + 2))})
    parts, prev = [], None
    for j in keep:
        if prev is not None and j != prev + 1:
            parts.append("[...]")
        parts.append(units[j])
        prev = j
    out = " ".join(parts)
    return out if len(out) <= EXCERPT_CHARS else out[:EXCERPT_CHARS] + " [...]"


def candidates(messages: list[Message], scenarios: dict[str, Scenario], formats: set[str]) -> list[tuple]:
    seen, out = set(), []
    for m in messages:
        key = (m.run_id, m.distance, m.sender, m.receiver)
        if key in seen:
            raise ValueError(f"duplicate message key {key}: log mixes several attempts of one run")
        seen.add(key)
        if message_kind(m) != "handoff" or m.condition.handoff_format not in formats:
            continue
        out += [(m, f) for f in scenarios[m.scenario].facts]
    return out


def stratified(units: list[tuple], n: int, seed: int) -> list[tuple]:
    rng = random.Random(seed)
    strata: dict[tuple, list] = defaultdict(list)
    for m, f in units:
        strata[(m.condition.model, m.scenario, f.type, distance_bin(m.distance))].append((m, f))
    order = sorted(strata)
    rng.shuffle(order)
    for k in order:
        rng.shuffle(strata[k])
    picked: list[tuple] = []
    while len(picked) < n and any(strata.values()):
        for k in order:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    rng.shuffle(picked)  # sheet order carries no stratum information
    return picked


def write_sheet(picked: list[tuple], out: Path) -> Path:
    msg_dir = out / "messages"
    msg_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (m, f) in enumerate(picked, 1):
        item, note = f"L{i:03d}", view_text(m, "note")
        path = msg_dir / f"{item}.txt"
        path.write_text(f"item: {item}\nrun: {m.run_id}\nmessage: {m.sender} -> {m.receiver} (distance {m.distance})\n"
                        f"fact to check ({f.id}, {f.type}): {f.text}\n{'=' * 72}\n{note}\n", encoding="utf-8")
        rows.append({"item_id": item, "run_id": m.run_id, "distance": m.distance, "sender": m.sender,
                     "receiver": m.receiver, "fact_id": f.id, "fact_type": f.type, "fact_text": f.text,
                     "message_excerpt": excerpt(note, f), "message_file": str(path.relative_to(out)),
                     "label": "", "notes": ""})
    sheet = out / "sheet.csv"
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return sheet


def build(runs: Path, scenarios_dir: Path, n: int, seed: int, formats: set[str], out: Path | None = None) -> Path:
    valid = valid_runs(runs / "runs.csv")
    msgs = [m for m in MessageLog(runs / "messages.jsonl").read() if m.run_id in valid]
    scenarios = {sid: Scenario.load(scenarios_dir / sid) for sid in {m.scenario for m in msgs}}
    out = out or runs / "labels"
    if (out / "sheet.csv").exists():
        raise FileExistsError(f"{out / 'sheet.csv'} exists; refusing to overwrite labels (use another --out)")
    return write_sheet(stratified(candidates(msgs, scenarios, formats), n, seed), out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "runs" / "cc_pilot")
    ap.add_argument("--scenarios", type=Path, default=ROOT / "scenarios")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--formats", default="free,structured", help="hand-off formats to sample (comma-separated)")
    ap.add_argument("--out", type=Path, default=None, help="default: <runs>/labels")
    args = ap.parse_args(argv)
    sheet = build(args.runs, args.scenarios, args.n, args.seed, set(args.formats.split(",")), args.out)
    print(f"wrote {sheet}")


if __name__ == "__main__":
    main()
