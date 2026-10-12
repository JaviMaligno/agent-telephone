"""Agreement and accuracy: Cohen's kappa (pass vs pass, judge vs manual) and confusion matrices."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .rows import LABELS, ScoreRow

Key = tuple  # (run_id, distance, sender, receiver, fact_id)


@dataclass
class Agreement:
    kappa: float
    observed: float
    n: int


def cohen_kappa(a: list[str], b: list[str]) -> Agreement:
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of equal length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / n**2
    return Agreement(1.0 if pe == 1 else (po - pe) / (1 - pe), po, n)


def _labels(rows: list[ScoreRow]) -> dict[Key, str]:
    return {r.key: r.label for r in rows if r.label != "unscored"}


def _paired(a: dict[Key, str], b: dict[Key, str]) -> Agreement:
    keys = sorted(a.keys() & b.keys())
    return cohen_kappa([a[k] for k in keys], [b[k] for k in keys])


def kappa_between(pass_a: list[ScoreRow], pass_b: list[ScoreRow]) -> Agreement:
    """Agreement between two scoring passes on the units both scored (unscored excluded)."""
    return _paired(_labels(pass_a), _labels(pass_b))


MANUAL_COLUMNS = ("run_id", "distance", "sender", "receiver", "fact_id", "label")


def load_manual(path: str | Path) -> dict[Key, str]:
    """CSV with columns run_id,distance,sender,receiver,fact_id,label."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r["label"] not in LABELS:
            raise ValueError(f"bad manual label {r['label']!r}")
        out[(r["run_id"], int(r["distance"]), r["sender"], r["receiver"], r["fact_id"])] = r["label"]
    return out


def load_sheet(path: str | Path) -> dict[Key, str]:
    """Filled labelling sheet (scripts/make_label_sheet.py). Blank labels are skipped (not yet labelled)."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        label = (r.get("label") or "").strip().lower()
        if not label:
            continue
        if label not in LABELS:
            raise ValueError(f"bad label {r['label']!r} in item {r.get('item_id')}")
        out[(r["run_id"], int(r["distance"]), r["sender"], r["receiver"], r["fact_id"])] = label
    return out


def against_manual_by_method(rows: list[ScoreRow], manual: dict[Key, str], view: str = "note",
                             judge_pass: int = 0) -> dict[str, tuple[Agreement, list[str], list[str]]]:
    """Per scoring method: (kappa vs manual, manual labels, method labels) on the shared scored units.

    Only rows of one view and one judge pass are kept, so every key has a single label per method."""
    by: dict[str, dict[Key, str]] = defaultdict(dict)
    for r in rows:
        if r.view == view and r.judge_pass == judge_pass and r.label != "unscored":
            by[r.method][r.key] = r.label
    out = {}
    for method, labels in sorted(by.items()):
        keys = sorted(labels.keys() & manual.keys())
        if keys:
            truth, pred = [manual[k] for k in keys], [labels[k] for k in keys]
            out[method] = (cohen_kappa(truth, pred), truth, pred)
    return out


def kappa_against_manual(rows: list[ScoreRow], manual: dict[Key, str]) -> Agreement:
    return _paired(_labels(rows), manual)


def confusion(truth: list[str], pred: list[str]) -> Counter:
    return Counter(zip(truth, pred))


def precision_recall(truth: list[str], pred: list[str]) -> dict[str, tuple[float, float, int]]:
    """label -> (precision, recall, support); NaN where undefined."""
    cm, out = confusion(truth, pred), {}
    for lab in sorted(set(truth) | set(pred)):
        tp = cm[(lab, lab)]
        npred = sum(v for (_, p), v in cm.items() if p == lab)
        ntrue = sum(v for (t, _), v in cm.items() if t == lab)
        out[lab] = (tp / npred if npred else float("nan"), tp / ntrue if ntrue else float("nan"), ntrue)
    return out


def format_confusion(truth: list[str], pred: list[str]) -> str:
    cm, labels = confusion(truth, pred), sorted(set(truth) | set(pred))
    head = "truth \\ pred | " + " | ".join(labels)
    lines = [head] + [f"{t} | " + " | ".join(str(cm[(t, p)]) for p in labels) for t in labels]
    pr = precision_recall(truth, pred)
    lines += [f"{lab}: precision={p:.3f} recall={r:.3f} n={n}" for lab, (p, r, n) in pr.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Manual sheet vs deterministic (recomputed from the log, free) and vs any judge score files."""
    from ..schema import MessageLog, Scenario
    from . import load_scores, score_messages

    ap = argparse.ArgumentParser(description="Agreement of scorers with a filled manual labelling sheet.")
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--log", type=Path, help="messages.jsonl: deterministic labels are recomputed from it")
    ap.add_argument("--scenarios", type=Path, default=Path(__file__).resolve().parents[2] / "scenarios")
    ap.add_argument("--scores", type=Path, nargs="*", default=[], help="score JSONL files (e.g. judge passes)")
    ap.add_argument("--view", default="note", choices=["note", "full"])
    ap.add_argument("--judge-pass", type=int, default=0)
    args = ap.parse_args(argv)
    manual = load_sheet(args.sheet)
    print(f"manual labels: {len(manual)}")
    rows: list[ScoreRow] = [r for p in args.scores for r in load_scores(p)]
    if args.log:
        keys = {k[:4] for k in manual}
        msgs = [m for m in MessageLog(args.log).read() if (m.run_id, m.distance, m.sender, m.receiver) in keys]
        scenarios = {sid: Scenario.load(args.scenarios / sid) for sid in {m.scenario for m in msgs}}
        rows = [r for r in rows if r.method != "deterministic"]
        rows += score_messages(scenarios, msgs, None, view=args.view)[0]
    for method, (ag, truth, pred) in against_manual_by_method(rows, manual, args.view, args.judge_pass).items():
        print(f"\n## manual vs {method}: kappa={ag.kappa:.3f} observed={ag.observed:.3f} n={ag.n}")
        print(format_confusion(truth, pred))


if __name__ == "__main__":
    main()
