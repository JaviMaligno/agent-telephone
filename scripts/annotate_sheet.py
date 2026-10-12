"""Model-anchored labels for the blind sheet (no human labels, see DESIGN changelog 2026-10-12).

1. Annotator B (default Claude Opus 5.5 on the subscription) labels every sheet row blind, with the
   same instructions a human would get (labels/README.md) and the full message.
2. Where B disagrees with the instrument (judge label if the unit was judged, else deterministic),
   an adjudicator (default Claude Fable 5.1) sees both labels with their evidence and decides.
3. Writes sheet_B.csv and sheet_anchor.csv (fillable-sheet format, readable by scoring.agreement).

    uv run python scripts/annotate_sheet.py --scores runs/cc_pilot/scored_judge/scores_note.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_telephone.backends.claude_cli import ClaudeCliBackend
from agent_telephone.scoring import load_scores

LABELS = ["preserved", "lost", "distorted", "inverted"]
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["label", "evidence"],
          "properties": {"label": {"type": "string", "enum": LABELS},
                         "evidence": {"type": "string", "description": "quoted span relied on, or why absent"}}}


def instructions(readme: Path) -> str:
    text = readme.read_text(encoding="utf-8")
    start, end = text.index("## How to label one row"), text.index("## Agreement")
    return ("You are a careful annotator. You label how one client fact survives in one message passed "
            "between agents. Follow these instructions exactly.\n\n" + text[start:end].strip())


def key(r: dict) -> tuple:
    return (r["run_id"], int(r["distance"]), r["sender"], r["receiver"], r["fact_id"])


def instrument_labels(scores: Path) -> dict[tuple, tuple[str, str, str]]:
    out = {}
    for r in load_scores(scores):
        if r.view != "note" or r.judge_pass != 0 or r.label == "unscored":
            continue
        if r.key not in out or r.method == "judge":  # judge overrides deterministic where both exist
            out[r.key] = (r.label, r.method, r.evidence or "")
    return out


def label_one(backend, model, system, row, message) -> dict:
    prompt = (f"FACT (as the client stated it):\n{row['fact_text']}\n\n"
              f"MESSAGE ({row['sender']} -> {row['receiver']}):\n<<<\n{message}\n>>>\n\nLabel the fact.")
    r = backend.complete(system, [{"role": "user", "content": prompt}], model, 2000, schema=SCHEMA)
    return json.loads(r.text)


def adjudicate(backend, model, system, row, message, a: tuple, b: dict) -> dict:
    prompt = (f"FACT (as the client stated it):\n{row['fact_text']}\n\nMESSAGE ({row['sender']} -> {row['receiver']}):"
              f"\n<<<\n{message}\n>>>\n\nTwo annotators disagree.\n- Annotator 1: {a[0]} (evidence: {a[2] or 'none given'})"
              f"\n- Annotator 2: {b['label']} (evidence: {b['evidence']})\n\nRead the message yourself and give the "
              "correct label under the instructions. You may pick a label neither annotator chose.")
    r = backend.complete(system, [{"role": "user", "content": prompt}], model, 2000, schema=SCHEMA)
    return json.loads(r.text)


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", type=Path, default=Path("runs/cc_pilot/labels"))
    ap.add_argument("--scores", type=Path, required=True, help="instrument score rows (judge run, note view)")
    ap.add_argument("--annotator", default="claude-opus-5-5")
    ap.add_argument("--adjudicator", default="claude-fable-5-1")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    d = args.labels_dir
    rows = list(csv.DictReader((d / "sheet.csv").open(encoding="utf-8")))
    fields = list(rows[0].keys()) + ["instrument_label", "instrument_method", "b_label", "decided_by"]
    system, backend = instructions(d / "README.md"), ClaudeCliBackend()
    msg = {r["item_id"]: (d / "messages" / f"{r['item_id']}.txt").read_text(encoding="utf-8") for r in rows}
    inst = instrument_labels(args.scores)

    b_path = d / "sheet_B.jsonl"  # resumable cache of annotator B
    done = {json.loads(l)["item_id"]: json.loads(l) for l in b_path.open()} if b_path.exists() else {}
    todo = [r for r in rows if r["item_id"] not in done]
    with ThreadPoolExecutor(args.workers) as ex:
        for r, out in zip(todo, ex.map(lambda r: label_one(backend, args.annotator, system, r, msg[r["item_id"]]), todo)):
            done[r["item_id"]] = {"item_id": r["item_id"], **out}
            with b_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(done[r["item_id"]]) + "\n")

    sheet_b, anchor, disagree = [], [], []
    for r in rows:
        b, a = done[r["item_id"]], inst.get(key(r))
        base = {**r, "instrument_label": a[0] if a else "", "instrument_method": a[1] if a else "", "b_label": b["label"]}
        sheet_b.append({**base, "label": b["label"], "notes": b["evidence"][:500]})
        if a is None or a[0] == b["label"]:
            anchor.append({**base, "label": b["label"], "decided_by": "agreement" if a else "annotator_b_only"})
        else:
            disagree.append((r, a, b))
    a_path = d / "adjudications.jsonl"
    adj = {json.loads(l)["item_id"]: json.loads(l) for l in a_path.open()} if a_path.exists() else {}
    pending = [x for x in disagree if x[0]["item_id"] not in adj]
    with ThreadPoolExecutor(args.workers) as ex:
        res = ex.map(lambda x: adjudicate(backend, args.adjudicator, system, x[0], msg[x[0]["item_id"]], x[1], x[2]), pending)
        for (r, _, _), out in zip(pending, res):
            adj[r["item_id"]] = {"item_id": r["item_id"], **out}
            with a_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(adj[r["item_id"]]) + "\n")
    for r, a, b in disagree:
        out = adj[r["item_id"]]
        anchor.append({**r, "instrument_label": a[0], "instrument_method": a[1], "b_label": b["label"],
                       "label": out["label"], "notes": out["evidence"][:500], "decided_by": "adjudicator"})
    write(d / "sheet_B.csv", sheet_b, fields)
    write(d / "sheet_anchor.csv", sorted(anchor, key=lambda x: x["item_id"]), fields)
    n_inst = sum(1 for r in rows if inst.get(key(r)))
    print(f"{len(rows)} items; instrument label for {n_inst}; B agrees with instrument on "
          f"{n_inst - len(disagree)}/{n_inst}; adjudicated {len(disagree)} "
          f"(adjudicator sided with instrument {sum(adj[r['item_id']]['label'] == a[0] for r, a, _ in disagree)}, "
          f"with B {sum(adj[r['item_id']]['label'] == b['label'] for r, _, b in disagree)})")


if __name__ == "__main__":
    main()
