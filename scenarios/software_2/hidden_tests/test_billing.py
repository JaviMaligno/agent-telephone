"""Derived only from the original brief. A default change on 16 March leaves 15 of the period's
days (period_end 31 March); prices are per seat and month, in cents.

Each test asserts only on the facts in its @facts tag: properties of other facts (period length,
rounding, VAT, approvers, order) are not assumed, so losing one fact fails only the tests tagged
with it (checked by tests/test_pilot.py against every mutant)."""
from datetime import datetime, timedelta, timezone

import pytest

facts = pytest.mark.facts
NOW = datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc)
END = datetime(2026, 3, 31, 9, 0, tzinfo=timezone.utc)  # same time of day as NOW: whole days under any convention
APPROVER, APPROVER_BIG = "Nguyễn Thị Phương Thảo", "Ramón Etxeberria-Goñi"


def sub(i=1, **kw):
    base = {"id": f"S{i}", "status": "active", "plan_name": "Studio", "seat_price_cents": 1000, "seats": 4,
            "billable_seats": 4, "vat_id": None, "period_end": END, "legacy_discount": 0.25}
    return base | kw


def new_plan(name="Studio Plus", price=1600):
    return {"name": name, "seat_price_cents": price}


def lines(mod, s=None, p=None, at=NOW):
    return mod.prorate_seats(s or sub(), p or new_plan(), at)


def kind(ls, k):
    return [line for line in ls if line["kind"] == k]


@facts("f_entry")
def test_entry_point_and_line_shape(mod):
    ls = lines(mod)
    assert ls
    for line in ls:
        assert {"subscription_id", "kind", "amount_cents", "vat_cents", "description", "approved_by"} <= set(line)
        assert line["subscription_id"] == "S1" and line["kind"] in {"credit", "charge"}


@facts("f_period", "f_seat_field")
def test_prorated_over_30_days_on_billable_seats(mod):
    ls = lines(mod, sub(seats=10))  # 4 billable of 10; 15 days left of a 30-day period
    assert [c["amount_cents"] for c in kind(ls, "credit")] == [2000]
    assert [c["amount_cents"] for c in kind(ls, "charge")] == [3200]


@facts("f_no_round_us", "f_period")
def test_rounding_in_the_customers_favour(mod):
    ls = lines(mod, sub(seats=1, billable_seats=1), new_plan(price=2000), at=NOW + timedelta(days=5))  # 10 days
    assert [c["amount_cents"] for c in kind(ls, "credit")] == [334]  # 333.33 up
    assert [c["amount_cents"] for c in kind(ls, "charge")] == [666]  # 666.67 down


@facts("f_min")
def test_lines_under_50_cents_are_skipped(mod):
    ls = lines(mod, sub(seats=1, billable_seats=1, seat_price_cents=100), new_plan(price=3000),
               at=NOW + timedelta(days=12))  # 3 days left: a ~10-cent credit and a ~300-cent charge
    assert ls and all(line["amount_cents"] >= 50 for line in ls)


@facts("f_cond_downgrade")
def test_downgrade_returns_no_lines(mod):
    assert lines(mod, sub(seat_price_cents=1600), new_plan(price=1000)) == []


@facts("f_no_trial")
def test_no_lines_for_trialing(mod):
    assert lines(mod, sub(status="trialing")) == [] and lines(mod, sub())


@facts("f_vat_rate")
def test_charges_carry_21_percent_vat(mod):
    charges = kind(lines(mod), "charge")  # whatever the amount, VAT is 21% of it (to the cent)
    assert charges and all(abs(c["vat_cents"] - c["amount_cents"] * 0.21) <= 1 for c in charges)


@facts("f_cond_vat")
def test_reverse_charge_when_vat_id(mod):
    with_id, without = kind(lines(mod, sub(vat_id="ATU12345678")), "charge"), kind(lines(mod), "charge")
    assert with_id and all(c["vat_cents"] == 0 for c in with_id)
    assert without and all(c["vat_cents"] > 0 for c in without)


@facts("f_no_vat_credit")
def test_credits_carry_no_vat(mod):
    credits = kind(lines(mod), "credit")
    assert credits and all(c["vat_cents"] == 0 for c in credits)


LONG = "Enterprise Collaboration Suite with Advanced Analytics and Priority Support Bundle"


@facts("f_desc_len")
def test_descriptions_at_most_80_chars(mod):
    ls = lines(mod, sub(plan_name=LONG), new_plan(name=LONG + " Plus"))
    assert ls and all(len(line["description"]) <= 80 for line in ls)


@facts("f_priority_date", "f_desc_len")
def test_long_plan_name_is_shortened_but_date_kept(mod):
    ls = lines(mod, sub(plan_name="Ultra-" * 30), new_plan(name="Mega-" * 30))
    assert ls
    for line in ls:
        assert len(line["description"]) <= 80 and "2026-03-31" in line["description"]


@facts("f_approver")
def test_credits_approved_by_thao(mod):
    credits = kind(lines(mod), "credit")  # a 20 EUR credit
    assert credits and all(c["approved_by"] == APPROVER for c in credits)


@facts("f_approver_big", "f_cond_big")
def test_big_credits_approved_by_ramon(mod):
    big = kind(lines(mod, sub(seats=20, billable_seats=20, seat_price_cents=6000), new_plan(price=8000)), "credit")
    small = kind(lines(mod), "credit")
    assert big and all(c["approved_by"] == APPROVER_BIG for c in big)  # a ~600 EUR credit
    assert small and all(c["approved_by"] != APPROVER_BIG for c in small)  # the default approver is f_approver's test


@facts("f_low_order")
def test_credit_before_charge(mod):
    assert [line["kind"] for line in lines(mod)] == ["credit", "charge"]


@facts("f_low_int")
def test_amounts_are_integer_cents(mod):
    ls = lines(mod, sub(seats=1, billable_seats=1), new_plan(price=2000), at=NOW + timedelta(days=5))
    assert ls and all(type(line["amount_cents"]) is int and type(line["vat_cents"]) is int for line in ls)
