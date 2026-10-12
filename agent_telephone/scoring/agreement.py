"""Agreement and accuracy: Cohen's kappa (pass vs pass, judge vs manual) and confusion matrices."""
from __future__ import annotations

import csv
from collections import Counter
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
