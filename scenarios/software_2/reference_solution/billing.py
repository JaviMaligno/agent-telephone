"""Seat-upgrade proration for Fernhollow Studios (reference solution for the hidden tests)."""
from __future__ import annotations

from datetime import datetime

PERIOD_DAYS = 30
MIN_CENTS = 50
VAT_RATE = 21
DESC_MAX = 80
BIG_CREDIT_CENTS = 50_000
APPROVER = "Nguyễn Thị Phương Thảo"
APPROVER_BIG = "Ramón Etxeberria-Goñi"


def _half_up(n: int, d: int) -> int:
    return (2 * n + d) // (2 * d)


def _description(kind: str, plan: str, end: datetime) -> str:
    def build(name: str) -> str:
        return f"{kind.capitalize()}: {name} seats until {end:%Y-%m-%d}"

    text = build(plan)
    if len(text) > DESC_MAX:  # shorten the plan name, never the date
        text = build(plan[: max(DESC_MAX - len(build("")), 0)])
    return text


def prorate_seats(subscription: dict, new_plan: dict, changed_at: datetime) -> list[dict]:
    if subscription.get("status") == "trialing":
        return []
    seats = subscription["billable_seats"]
    old, new = subscription["seat_price_cents"] * seats, new_plan["seat_price_cents"] * seats
    if new < old:  # a downgrade applies at renewal
        return []
    end = subscription["period_end"]
    days = min(max((end.date() - changed_at.date()).days, 0), PERIOD_DAYS)
    credit, charge = -(-old * days // PERIOD_DAYS), new * days // PERIOD_DAYS  # credits up, charges down
    sid = subscription["id"]
    lines = [
        {"subscription_id": sid, "kind": "credit", "amount_cents": int(credit), "vat_cents": 0,
         "description": _description("credit", subscription["plan_name"], end),
         "approved_by": APPROVER_BIG if credit > BIG_CREDIT_CENTS else APPROVER},
        {"subscription_id": sid, "kind": "charge", "amount_cents": charge,
         "vat_cents": 0 if subscription.get("vat_id") else _half_up(charge * VAT_RATE, 100),
         "description": _description("charge", new_plan["name"], end), "approved_by": None},
    ]
    return [line for line in lines if line["amount_cents"] >= MIN_CENTS]
