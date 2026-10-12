"""Analysis: survival curves and noise propagation classes from message logs + scores.

Survival = preserved / scored (unscored excluded) for each group x distance, also broken down
into lost / distorted / inverted rates. Rows are split by message `kind` (source / handoff /
endpoint) so the endpoint artifact is never pooled with hand-offs at the same distance.

Pooling rules (pre-registered, DESIGN §7):
- H1/H5 survival-by-distance tables use only **downward, non-forward** hand-offs (`primary`):
  upward hand-backs (pm -> lead in hub_relay) do not remove anything from the system and are
  reported separately (`survival_upward`); `forward` notes are told not to restate the brief,
  so "lost" is meaningless for them.
- `survival_by_info_distance` uses the writer's information distance (hub_relay's lead -> r2 is
  nominal distance 4 but information distance 2).
- `forward` hand-offs are scored as **contradiction**: did the note contradict the attached
  original (distorted / inverted)? A silent note is not a loss, and repeating the attachment's own
  noise reading is not a contradiction (counted apart as `noise_reading`) (`forward_contradiction`).

Noise classes, per (noisy run, affected fact, non-source message), always on the agent-written
part of the message (note view: deterministic labels recomputed on the note, judge labels only
from note-view score rows, else `unscored`). The wrong reading is the variant's machine-checkable
`wrong_pattern`, searched on number-normalised text:
- flagged: the wrong reading is present and its own sentence doubts the *name/reading* ("typo",
  "transcription", "please confirm the spelling", "name unclear", or a "?" right after it); a
  "confirm" about something else in the sentence (the venue, transport) is not doubt;
- amplify: the wrong reading appears in a prose sentence outside its legitimate context
  (`wrong_context`): it spread to another fact (repeating it in context is not amplification).
  Code/JSON (fenced blocks, `x = ...` / `{` lines, endpoint artifacts, hand-backs relaying code)
  has no sentences to spread to: a reading found only there is freeze;
- freeze:  the wrong reading is present, in context, unquestioned;
- repair:  the wrong reading is absent and the fact is preserved (only possible when the
  variant is `repairable`; an ASR-mangled name gives no way back);
- lost:    neither reading present;
- drift:   wrong reading absent but the fact is distorted some other way.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from .formats import strip_attachment
from .schema import Message, MessageLog, Scenario
from .scoring import DeterministicScorer, InventedClaim, ScoreRow, load_scores, message_kind, save_jsonl, score_messages
from .scoring.deterministic import canon
from .topologies import USER

CONDITION_KEYS = ("topology", "model", "handoff_format", "noise")
# H1/H5 tables exclude upward hand-backs and forward notes; see `primary`.
NOISE_CLASSES = ("flagged", "freeze", "amplify", "repair", "lost", "drift")
# Doubt about the name/reading itself (not "confirm the venue"): searched in the hit's own sentence.
DOUBT = re.compile(
    r"\b(?:typo|transcri\w*|mis-?heard|mis-?spel\w*|phonetic\w*|garbled|sounds? like|as heard)\b"
    r"|\b(?:confirm\w*|verify|check\w*|double[- ]check\w*|clarify)\b[^.\n]{0,40}?\b(?:spell\w*|names?)\b"
    r"|\b(?:spell\w*|names?)\b[^.\n]{0,40}?\b(?:unclear|uncertain|unsure|not sure|unverified|to be confirmed|tbc"
    r"|needs? (?:confirm\w*|check\w*|verif\w*)|confirm\w*|verify|check)\b", re.I)
QUESTION_AFTER = re.compile(r"[\s\"'*)]{0,3}\?")
SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+")
CODE_LINE = re.compile(
    r"^\s*(?:[\w.\[\]'\"]+\s*(?:=|:=)\s*\S|[{}\[\]]|[\"'][^\"']*[\"']\s*[:,]|def |class |return |import |from \w+ import"
    r"|#|//|assert\b|self\.|else\b|elif\b|(?:if|for|while|with)\b.*:\s*$)"
    r"|^ {4,}(?![-*+] |\d+[.)] )\S"  # indented code (not a nested list item)
    r"|[\"'][)\]},;:]*\s*$")  # a line that ends inside a string literal / call
FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)


def prose_part(text: str) -> str:
    """The prose of a note: fenced blocks and code/JSON-shaped lines removed."""
    return "\n".join(line for line in FENCE.sub("\n", text).splitlines() if not CODE_LINE.match(line))


def _get(r: ScoreRow, key: str):
    return getattr(r.condition, key) if key in CONDITION_KEYS else getattr(r, key)


def survival(scores: list[ScoreRow], by: tuple[str, ...] = (), distance_key: str = "distance") -> list[dict]:
    groups: dict[tuple, list[ScoreRow]] = defaultdict(list)
    keys = (*by, "kind", distance_key)
    for r in scores:
        groups[tuple(_get(r, k) for k in keys)].append(r)
    out = []
    for g in sorted(groups, key=lambda t: tuple(map(str, t[:-1])) + (t[-1],)):
        rs = groups[g]
        counts = {lab: sum(r.label == lab for r in rs) for lab in ("preserved", "lost", "distorted", "inverted", "unscored")}
        scored = len(rs) - counts["unscored"]
        rates = {f"{lab}_rate": (counts[lab] / scored if scored else None)
                 for lab in ("preserved", "lost", "distorted", "inverted")}
        out.append({**dict(zip(keys, g)), "n": len(rs), "scored": scored, **counts,
                    "survival": rates.pop("preserved_rate"), **rates})
    return out


def primary(scores: list[ScoreRow]) -> list[ScoreRow]:
    """Rows eligible for H1/H5 survival: source, endpoint and downward non-forward hand-offs."""
    return [r for r in scores if r.kind != "handoff" or (r.direction == "down" and r.condition.handoff_format != "forward")]


def upward(scores: list[ScoreRow]) -> list[ScoreRow]:
    return [r for r in scores if r.kind == "handoff" and r.direction == "up"]


def is_noise_reading(r: ScoreRow) -> bool:
    """Distorted only because the note carries the run's noise reading (deterministic, see FactMatcher)."""
    return r.label == "distorted" and r.method == "deterministic" and (r.evidence or "").startswith("noise reading")


def forward_contradiction(scores: list[ScoreRow], by: tuple[str, ...] = ()) -> list[dict]:
    """Forward hand-offs: per distance, how often the note contradicts the attachment."""
    groups: dict[tuple, list[ScoreRow]] = defaultdict(list)
    for r in scores:
        if r.kind == "handoff" and r.condition.handoff_format == "forward":
            groups[(*(_get(r, k) for k in by), r.direction, r.distance)].append(r)
    out = []
    for g, rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        scored = [r for r in rs if r.label != "unscored"]
        echo = sum(is_noise_reading(r) for r in scored)  # repeats the attached (noisy) source: no contradiction
        bad = sum(r.label in ("distorted", "inverted") and not is_noise_reading(r) for r in scored)
        out.append({**dict(zip((*by, "direction", "distance"), g)), "n": len(rs), "scored": len(scored),
                    "contradicted": bad, "contradiction_rate": bad / len(scored) if scored else None,
                    "noise_reading": echo,
                    "silent_rate": sum(r.label == "lost" for r in scored) / len(scored) if scored else None})
    return out


def invented_by_distance(invented: list[InventedClaim], messages: list[Message]) -> list[dict]:
    per_msg = defaultdict(int)
    for c in invented:
        per_msg[(c.run_id, c.distance, c.sender, c.receiver)] += 1
    groups = defaultdict(list)
    for m in messages:
        groups[(message_kind(m), m.distance)].append(per_msg[(m.run_id, m.distance, m.sender, m.receiver)])
    return [{"kind": k, "distance": d, "messages": len(v), "invented": sum(v), "per_message": sum(v) / len(v)}
            for (k, d), v in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0]))]


def wrong_reading(note: str, pattern: str, context: str) -> tuple[int, bool, bool]:
    """(occurrences of the wrong reading, any of them doubted in its own sentence, any outside its
    context). Doubt and spread are read on the prose part only (see `prose_part`)."""
    text = canon(note)
    count = len(re.findall(pattern, text, re.I))
    flagged = spread = False
    for sent in SENTENCE.split(prose_part(text)):
        hits = list(re.finditer(pattern, sent, re.I))
        if not hits:
            continue
        flagged |= bool(DOUBT.search(sent)) or any(QUESTION_AFTER.match(sent, m.end()) for m in hits)
        spread |= not re.search(context, sent, re.I)
    return count, flagged, spread


def noise_classes(scenarios: dict[str, Scenario], messages: list[Message], scores: list[ScoreRow]) -> list[dict]:
    labels = {r.key: r.label for r in scores if r.view == "note"}
    det = {sid: DeterministicScorer(s) for sid, s in scenarios.items()}
    by_run: dict[str, list[Message]] = defaultdict(list)
    for m in messages:
        by_run[m.run_id].append(m)
    out = []
    for run_id, msgs in by_run.items():
        cond, scen = msgs[0].condition, scenarios[msgs[0].scenario]
        if cond.noise == "clean":
            continue
        v = next(n for n in scen.noise if n.id == cond.noise)
        src_count = wrong_reading(next(m.text for m in msgs if m.sender == USER), v.wrong_pattern, v.wrong_context)[0]
        for m in msgs:
            if m.sender == USER:
                continue
            note = strip_attachment(m.text)
            det_labels = det[scen.id].label_text(note, cond.noise)
            count, flagged, spread = wrong_reading(note, v.wrong_pattern, v.wrong_context)
            for fid in v.affected_fact_ids:
                label = det_labels[fid][0] if fid in det_labels else labels.get(
                    (run_id, m.distance, m.sender, m.receiver, fid), "unscored")
                if count:
                    prose = message_kind(m) != "endpoint"  # the artifact is code/JSON: no sentences to spread to
                    cls = "flagged" if flagged and prose else "amplify" if spread and prose else "freeze"
                else:
                    cls = {"preserved": "repair", "lost": "lost", "unscored": "unscored"}.get(label, "drift")
                out.append({"run_id": run_id, "scenario": scen.id, "noise": cond.noise, "noise_kind": v.kind,
                            "repairable": v.repairable,
                            **{k: getattr(cond, k) for k in ("topology", "model", "handoff_format")},
                            "fact_id": fid, "kind": message_kind(m), "direction": m.direction or "down",
                            "distance": m.distance, "wrong_count": count, "source_count": src_count,
                            "label": label, "class": cls})
    return out


def noise_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(lambda: dict.fromkeys(NOISE_CLASSES, 0))
    for r in rows:
        if r["class"] in NOISE_CLASSES:
            groups[(r["noise_kind"], r["kind"], r["distance"])][r["class"]] += 1
    return [{"noise_kind": nk, "kind": k, "distance": d, **c} for (nk, k, d), c in sorted(groups.items())]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _cell(v) -> str:
    return "" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def markdown_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_no data_\n"
    cols = cols or list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_cell(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


RATE_COLS = ["scored", "survival", "lost_rate", "distorted_rate", "inverted_rate", "unscored"]


def analyse(scenarios: dict[str, Scenario], messages: list[Message], scores: list[ScoreRow],
            invented: list[InventedClaim], out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    main = primary(scores)
    tables = {
        "survival_by_distance": (survival(main), ["kind", "distance", *RATE_COLS]),
        "survival_by_info_distance": (survival([r for r in main if r.kind == "handoff"], distance_key="info_distance"),
                                      ["kind", "info_distance", *RATE_COLS]),
        "survival_upward": (survival(upward(scores), ("topology",)), ["topology", "kind", "distance", *RATE_COLS]),
        "survival_by_type": (survival(main, ("fact_type",)), ["fact_type", "kind", "distance", *RATE_COLS]),
        "survival_by_condition": (survival(scores, (*CONDITION_KEYS, "direction")),
                                  [*CONDITION_KEYS, "direction", "kind", "distance", *RATE_COLS]),
        "forward_contradiction": (forward_contradiction(scores, ("topology",)), None),
        "invented_by_distance": (invented_by_distance(invented, messages), None),
        "noise_classes": (noise_classes(scenarios, messages, scores), None),
    }
    tables["noise_summary"] = (noise_summary(tables["noise_classes"][0]), None)
    paths = {}
    md = ["# agent-telephone analysis", "",
          f"{len({m.run_id for m in messages})} runs, {len(messages)} messages, {len(scores)} scored units.", ""]
    for name, (rows, cols) in tables.items():
        paths[name] = out_dir / f"{name}.csv"
        write_csv(rows, paths[name])
        if name != "noise_classes":  # per-message rows: CSV only
            md += [f"## {name.replace('_', ' ')}", "", markdown_table(rows, cols)]
    paths["summary"] = out_dir / "summary.md"
    paths["summary"].write_text("\n".join(md), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agent_telephone.analysis",
                                 description="Survival and noise analysis. Without --scores, scores deterministically.")
    ap.add_argument("--scenario", action="append", required=True, help="scenario dir or yaml (repeatable)")
    ap.add_argument("--log", action="append", required=True, help="message log JSONL (repeatable)")
    ap.add_argument("--scores", help="score rows JSONL (default: deterministic scoring, written to OUT/scores.jsonl)")
    ap.add_argument("--invented", help="invented claims JSONL")
    ap.add_argument("--view", choices=["full", "note"], default="full")
    ap.add_argument("--out", default="runs/analysis")
    args = ap.parse_args(argv)

    scenarios = {s.id: s for s in map(Scenario.load, args.scenario)}
    messages = [m for p in args.log for m in MessageLog(p).read()]
    if args.scores:
        scores = load_scores(args.scores)
    else:
        scores, _ = score_messages(scenarios, messages, view=args.view)
        save_jsonl(scores, Path(args.out) / "scores.jsonl")
    invented = []
    if args.invented:
        lines = Path(args.invented).read_text(encoding="utf-8").splitlines()
        invented = [InventedClaim.model_validate_json(x) for x in lines if x.strip()]
    paths = analyse(scenarios, messages, scores, invented, args.out)
    print(f"wrote {', '.join(str(p) for p in paths.values())}")


if __name__ == "__main__":
    main()
