"""Derived only from the original brief. Accounts are described by how many days ago they were
last seen; with the brief's rules, last seen 731 days ago is due, 725 days ago is due in 5 days.

Each test asserts only on the facts in its @facts tag: properties of other facts (thresholds,
approver, ordering, date format) are not assumed, so losing one fact fails only the tests tagged
with it (checked by tests/test_pilot.py against every mutant)."""
from datetime import date, timedelta

import pytest

facts = pytest.mark.facts
TODAY = date(2026, 6, 15)
DPO = "Þórdís Sveinbjörnsdóttir"


def ago(days):
    return TODAY - timedelta(days=days)


def acc(i=1, seen=100, **kw):
    base = {"id": f"M{i:04d}", "email": f"member{i}@example.no", "status": "active", "last_seen_at": ago(seen),
            "last_login": ago(seen), "closed_at": None, "closed_by": None, "unpaid_invoices": 0,
            "litigation_hold": False, "tenant": "oslo-2"}
    return base | kw


def purge(mod, *accounts):
    return mod.plan_purge(list(accounts), TODAY)


def ids(actions, kind=None):
    return {a["account_id"] for a in actions if kind is None or a["action"] == kind}


@facts("f_entry")
def test_entry_point_and_action_shape(mod):
    acts = purge(mod, acc(1, seen=800))
    assert acts
    for a in acts:
        assert {"account_id", "action", "due", "approved_by"} <= set(a)
        assert a["account_id"] == "M0001" and a["action"] in {"delete", "anonymise", "warn"}


@facts("f_inactive")
def test_inactive_accounts_deleted_after_730_days(mod):
    acts = purge(mod, acc(1, seen=731), acc(2, seen=700), acc(3, seen=400), acc(4, seen=725))
    assert ids(acts, "delete") == {"M0001"}


@facts("f_seen_field")
def test_inactivity_uses_last_seen_at(mod):
    acts = purge(mod, acc(1, seen=10, last_login=ago(1000)), acc(2, seen=800, last_login=ago(5)))
    assert ids(acts, "delete") == {"M0002"}


@facts("f_closed")
def test_closed_accounts_deleted_30_days_after_closing(mod):
    acts = purge(mod, acc(1, seen=50, closed_at=ago(31), closed_by="member"),
                 acc(2, seen=50, closed_at=ago(20), closed_by="staff"))
    assert ids(acts, "delete") == {"M0001"}


@facts("f_warn", "f_inactive")
def test_warning_for_accounts_due_within_14_days(mod):
    acts = purge(mod, acc(1, seen=720), acc(2, seen=712), acc(3, seen=726))  # due in 10, 18 and 4 days
    assert ids(acts, "warn") == {"M0001", "M0003"}


@facts("f_cond_warn", "f_closed")
def test_no_warning_when_the_member_closed_it(mod):
    acts = purge(mod, acc(1, seen=50, closed_at=ago(25), closed_by="member"),  # both due in 5 days
                 acc(2, seen=50, closed_at=ago(25), closed_by="staff"))
    assert ids(acts, "warn") == {"M0002"}


@facts("f_cond_invoice")
def test_unpaid_invoices_anonymised_not_deleted(mod):
    acts = purge(mod, acc(1, seen=800, unpaid_invoices=2), acc(2, seen=800))
    assert {(a["account_id"], a["action"]) for a in acts} == {("M0001", "anonymise"), ("M0002", "delete")}


@facts("f_no_hold")
def test_litigation_hold_never_touched(mod):
    acts = purge(mod, acc(1, seen=800, litigation_hold=True), acc(2, seen=725, litigation_hold=True),
                 acc(3, seen=800, unpaid_invoices=1, litigation_hold=True), acc(4, seen=800))
    assert ids(acts) == {"M0004"}


@facts("f_no_staff")
def test_no_actions_for_staff_accounts(mod):
    acts = purge(mod, acc(1, seen=800, email="ragnhild@tidewater-klatring.no"), acc(2, seen=800))
    assert ids(acts) == {"M0002"}


@facts("f_no_email")
def test_email_never_in_the_actions(mod):
    acts = purge(mod, acc(1, seen=800), acc(2, seen=800, unpaid_invoices=1), acc(3, seen=725))
    assert acts and not any(f"member{i}@example.no" in str(acts) for i in (1, 2, 3))


@facts("f_batch")
def test_at_most_200_deletions_per_run(mod):
    acts = purge(mod, *(acc(i, seen=800) for i in range(1, 231)))
    assert 0 < len([a for a in acts if a["action"] == "delete"]) <= 200


@facts("f_priority_closed")
def test_closed_accounts_deleted_first(mod):
    inactive = [acc(i, seen=800) for i in range(1, 196)]
    closed = [acc(i, seen=800, closed_at=ago(100), closed_by="member") for i in range(196, 206)]
    acts = purge(mod, *inactive, *closed)  # 205 due, closed ones last in the input
    assert {c["id"] for c in closed} <= ids(acts, "delete")


@facts("f_dpo")
def test_deletions_approved_by_the_dpo(mod):
    acts = [a for a in purge(mod, acc(1, seen=800), acc(2, seen=800, unpaid_invoices=1)) if a["action"] != "warn"]
    assert len(acts) == 2 and all(a["approved_by"] == DPO for a in acts)


@facts("f_low_sorted")
def test_sorted_by_account_id(mod):
    acts = purge(mod, acc(3, seen=800), acc(1, seen=800), acc(2, seen=725))
    assert len(acts) == 3 and [a["account_id"] for a in acts] == sorted(a["account_id"] for a in acts)


@facts("f_low_iso")
def test_due_dates_are_iso_strings(mod):
    acts = purge(mod, acc(1, seen=800), acc(2, seen=725))
    assert acts and all(isinstance(a["due"], str) and date.fromisoformat(a["due"]) for a in acts)
