"""GDPR clean-up of Tidewater Climbing member accounts (reference solution for the hidden tests)."""
from __future__ import annotations

from datetime import date, timedelta

INACTIVE_DAYS = 730
CLOSED_DAYS = 30
WARN_DAYS = 14
MAX_DELETIONS = 200
STAFF_DOMAIN = "@tidewater-klatring.no"
DPO = "Þórdís Sveinbjörnsdóttir"


def _due(acc: dict) -> date:
    due = acc["last_seen_at"] + timedelta(days=INACTIVE_DAYS)
    if acc.get("closed_at"):
        due = min(due, acc["closed_at"] + timedelta(days=CLOSED_DAYS))
    return due


def _action(acc: dict, kind: str, due: date) -> dict:
    return {"account_id": acc["id"], "action": kind, "due": due.isoformat(),
            "approved_by": None if kind == "warn" else DPO}


def plan_purge(accounts: list[dict], today: date) -> list[dict]:
    actions, deletes = [], []
    for acc in accounts:
        if acc.get("litigation_hold"):
            continue
        if acc["email"].lower().endswith(STAFF_DOMAIN):
            continue
        due = _due(acc)
        if due <= today:
            if acc.get("unpaid_invoices"):  # keep the invoices, drop the personal data
                actions.append(_action(acc, "anonymise", due))
            else:
                deletes.append((acc, due))
        elif due <= today + timedelta(days=WARN_DAYS) and acc.get("closed_by") != "member":
            actions.append(_action(acc, "warn", due))
    deletes.sort(key=lambda d: d[0].get("closed_at") is None)  # closed accounts first (stable)
    actions += [_action(acc, "delete", due) for acc, due in deletes[:MAX_DELETIONS]]
    return sorted(actions, key=lambda a: a["account_id"])
