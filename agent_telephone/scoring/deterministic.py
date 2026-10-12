"""Deterministic scoring for facts whose check is exact / regex / fuzzy (names, numbers).

Decisions (see DESIGN §6):
- Text is compared after NFC + casefold + whitespace collapsing. Case and spacing never
  change a label.
- `number` facts are matched on **canonical** text (`canon`): spelled-out numbers become digits
  ("a hundred and forty-two" -> 142, "two" -> 2; a lone "one" is left alone), times become 24h
  HH:MM (6:30pm / 18.30 / 18h30 -> 18:30, 8:00 -> 08:00), dates with a year become ISO
  (19 November 2026, November 19th, 2026, 19/11/2026 -> 2026-11-19), and money shorthands are
  expanded (€18.5k -> €18500, EUR 18 500 -> EUR 18500). Patterns and slots are written against
  canonical text.
- **An accent drop is `distorted`, not `preserved`**: "Siobhan Ni Dhuill" for "Siobhán Ní Dhúill"
  is a wrong name. This holds even when the accent-folded form is listed in `aliases`; aliases
  that differ by more than diacritics (e.g. a short form) count as preserved.
- Slots: every slot regex is searched for **all** its matches (values listed in `ignore` are
  skipped). A slot is preserved if any match carries the expected value, distorted if it matched
  but never with the expected value, missing otherwise. With several slots (independent parts,
  any order/line): all preserved -> preserved; any distorted or some missing -> distorted;
  all missing -> the fallback below. The expected value is the slot's (unique) value on the
  clean brief.
- Fallback without a slot match: exact pattern/alias anywhere -> preserved; accent-folded match ->
  distorted; for `name` facts with a literal target, a rapidfuzz near match (>= threshold)
  -> distorted; otherwise lost.
- Numbers compare by value (1,500 == 1500 == 1500.0; times like 17:45 compare literally);
  a unit written next to the number in the slot must match too.
- Deterministic scoring never emits `inverted`: polarity is the judge's job (`also_judge`).
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from ..schema import Fact, Message, Scenario
from .rows import ScoreRow, View, row, view_text

DETERMINISTIC_KINDS = ("exact", "regex", "fuzzy")
NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*")

# --- canonical text for numbers, times, dates and money ---------------------------------------
_UNITS = "one two three four five six seven eight nine".split()
_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
WORD_VALUE = {w: i + 1 for i, w in enumerate(_UNITS)} | {w: i + 10 for i, w in enumerate(_TEENS)} | {
    w: (i + 2) * 10 for i, w in enumerate(_TENS)}
_U, _T, _TY = "|".join(_UNITS), "|".join(_TEENS), "|".join(_TENS)
_B100 = rf"(?:(?:{_TY})(?:[\s-]+(?:{_U}))?|{_T}|{_U})"
_B1000 = rf"(?:(?:a|{_U})\s+hundred(?:\s+(?:and\s+)?{_B100})?|{_B100})"
NUMBER_WORDS = re.compile(rf"\b(?:{_B1000}\s+thousand(?:\s+(?:and\s+)?{_B1000})?|{_B1000})\b", re.I)
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MON = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


def words_to_int(s: str) -> int:
    total, cur = 0, 0
    for w in re.findall(r"[a-z]+", s.lower()):
        if w == "hundred":
            cur = max(cur, 1) * 100
        elif w == "thousand":
            total, cur = total + max(cur, 1) * 1000, 0
        else:
            cur += WORD_VALUE.get(w, 0)
    return total + cur


def _date(y: str, m: int, d: str) -> str | None:
    return f"{int(y):04d}-{m:02d}-{int(d):02d}" if 1 <= m <= 12 and 1 <= int(d) <= 31 else None


def _month(name: str) -> int:
    return MONTHS.index(name[:3].lower()) + 1


def _dmy(m: re.Match) -> str:
    a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
    day, mon = (b, a) if b > 12 else (a, b)  # D/M/Y unless that is impossible
    return _date(y, mon, str(day)) or m.group(0)


def _ampm(h: str, mi: str | None, ap: str) -> str:
    h = int(h) % 12 + (12 if ap.lower() == "p" else 0)
    return f"{h:02d}:{mi or '00'}"


def canon(text: str) -> str:
    """Number-normalised text (see module docstring). Idempotent."""
    t = unicodedata.normalize("NFC", text)
    t = NUMBER_WORDS.sub(lambda m: m.group(0) if m.group(0).lower() == "one" else str(words_to_int(m.group(0))), t)
    t = re.sub(r"\b(\d{1,2})[:.h](\d{2})\s*([ap])\.?m\.?(?!\w)", lambda m: _ampm(m[1], m[2], m[3]), t, flags=re.I)
    t = re.sub(r"\b(\d{1,2})\s*([ap])\.?m\.?(?!\w)", lambda m: _ampm(m[1], None, m[2]), t, flags=re.I)
    t = re.sub(r"\b([01]\d|2[0-3])[.h]([0-5]\d)\b(?![.,]\d)", r"\1:\2", t)
    t = re.sub(r"\b(\d):([0-5]\d)\b", r"0\1:\2", t)
    t = re.sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+({_MON})\.?,?\s+(\d{{4}})\b",
               lambda m: _date(m[3], _month(m[2]), m[1]) or m[0], t, flags=re.I)
    t = re.sub(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
               lambda m: _date(m[3], _month(m[1]), m[2]) or m[0], t, flags=re.I)
    t = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", _dmy, t)
    t = re.sub(r"(€|\beur\b|\beuros?\b)(\s?)(\d{1,3})[ \u00a0\u202f](\d{3})\b", r"\1\2\3\4", t, flags=re.I)
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s?k\b", lambda m: str(round(float(m[1]) * 1000)), t, flags=re.I)
    return t


# --- matching --------------------------------------------------------------------------------

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip().casefold()


def fold(s: str) -> str:
    """norm() without diacritics."""
    return "".join(c for c in unicodedata.normalize("NFKD", norm(s)) if not unicodedata.combining(c))


def _contains(needle: str, hay: str) -> re.Match | None:
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay) if needle else None


def _numbers(s: str) -> list:
    out = []
    for tok in NUM_RE.findall(s):
        if ":" in tok or re.fullmatch(r"\d{4}-\d{2}-\d{2}", tok):
            out.append(tok)
            continue
        t = tok.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", tok) else tok.replace(",", ".")
        try:
            out.append(float(t))
        except ValueError:
            out.append(tok)
    return out


def _unit(s: str) -> str:
    return fold(NUM_RE.sub(" ", s)).strip(" .,;:")


def _value(m: re.Match) -> str | None:
    vals = [v for k, v in m.groupdict().items() if k.startswith("value") and v is not None]
    return vals[0].strip() if vals else None


def is_deterministic(fact: Fact) -> bool:
    return fact.check.kind in DETERMINISTIC_KINDS


def needs_judge(fact: Fact) -> bool:
    return not is_deterministic(fact) or fact.check.also_judge


class FactMatcher:
    def __init__(self, fact: Fact, brief: str, fuzzy_threshold: float = 85.0):
        self.fact, self.threshold = fact, fuzzy_threshold
        c = fact.check
        self.numeric = fact.type == "number"
        self.literal = None if c.kind == "regex" else (c.pattern or fact.text)
        self.aliases = [a for a in c.aliases if self.literal is None or fold(a) != fold(self.literal)]
        slots = [c.slot] if isinstance(c.slot, str) else (c.slot or [])
        self.slots = [re.compile(s, re.I) for s in slots]
        self.expected = []
        for i, slot in enumerate(self.slots):
            values = {v for v in self._slot_values(slot, self._text(brief))}
            if not values:
                raise ValueError(f"slot {i} of fact {fact.id} does not match the clean brief")
            if len({self._key(v) for v in values}) > 1:
                raise ValueError(f"slot {i} of fact {fact.id} is ambiguous on the clean brief: {sorted(values)}")
            self.expected.append(next(iter(values)))

    def _text(self, text: str) -> str:
        return canon(text) if self.numeric else unicodedata.normalize("NFC", text)

    def _key(self, value: str):
        return tuple(_numbers(value)) if self.numeric and _numbers(value) else fold(value)

    def _ignored(self, value: str) -> bool:
        return any(self._key(value) == self._key(self._text(i)) or _contains(norm(i), norm(value))
                   for i in self.fact.check.ignore)

    def _slot_values(self, slot: re.Pattern, text: str) -> list[str]:
        return [v for m in slot.finditer(text) if (v := _value(m)) is not None and not self._ignored(v)]

    def _same(self, expected: str, value: str) -> bool:
        if self.numeric and _numbers(expected):
            exp, got = _numbers(expected), _numbers(value)
            return bool(exp) and got[: len(exp)] == exp and (_unit(expected) in ("", _unit(value)))
        return any(_contains(norm(t), norm(value)) for t in [expected, *self.aliases])

    def _exact(self, text: str) -> str | None:
        if self.literal is None:
            m = re.search(self.fact.check.pattern, text, re.I)
            return m.group(0) if m else None
        hay = norm(text)
        for t in [self.literal, *self.aliases]:
            if m := _contains(norm(t), hay):
                return m.group(0)
        return None

    def _slots_label(self, text: str) -> tuple[str, str] | None:
        states, ev = [], []
        for slot, exp in zip(self.slots, self.expected):
            values = self._slot_values(slot, text)
            if any(self._same(exp, v) for v in values):
                states.append("preserved")
            elif values:
                states.append("distorted")
                ev.append(f"slot value {values[0]!r} (expected {exp!r})")
            else:
                states.append("missing")
                ev.append(f"slot for {exp!r} missing")
        if all(s == "missing" for s in states):
            return None
        if all(s == "preserved" for s in states):
            return "preserved", f"slot value(s) {self.expected!r}"
        return "distorted", "; ".join(ev)

    def label(self, text: str) -> tuple[str, str | None]:
        text = self._text(text)
        if self.slots and (lab := self._slots_label(text)):
            return lab
        if span := self._exact(text):
            return "preserved", span
        if self.literal is not None:
            if m := _contains(fold(self.literal), fold(text)):
                return "distorted", f"diacritics differ: {m.group(0)!r}"
            if self.fact.type == "name":
                al = fuzz.partial_ratio_alignment(fold(self.literal), fold(text), score_cutoff=self.threshold)
                if al is not None:
                    return "distorted", f"fuzzy {al.score:.0f}: {fold(text)[al.dest_start:al.dest_end]!r}"
        return "lost", None


class DeterministicScorer:
    """Scores the deterministic facts of one scenario. Validates every check on the clean brief."""

    def __init__(self, scenario: Scenario, fuzzy_threshold: float = 85.0):
        self.scenario = scenario
        self.matchers = {f.id: FactMatcher(f, scenario.brief, fuzzy_threshold)
                         for f in scenario.facts if is_deterministic(f)}
        for fid, m in self.matchers.items():
            if m.label(scenario.brief)[0] != "preserved":
                raise ValueError(f"check of fact {fid} does not score the clean brief as preserved")

    def label_text(self, text: str) -> dict[str, tuple[str, str | None]]:
        return {fid: m.label(text) for fid, m in self.matchers.items()}

    def score(self, msg: Message, view: View = "full") -> list[ScoreRow]:
        text = view_text(msg, view)
        return [row(msg, self.scenario.fact(fid), lab, "deterministic", ev, view=view)
                for fid, (lab, ev) in self.label_text(text).items()]
