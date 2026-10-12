"""Derived only from the original brief. Inputs are UTC (the server's zone); the brief's
rules (quiet hours, HH:MM in the text) are in Europe/Dublin, which is UTC+1 in July.

Each test asserts only on the facts in its @facts tag: properties of other facts (offsets,
signature, channel, late-booking rule) are not assumed, so losing one fact fails only the
tests tagged with it (checked by tests/test_pilot.py against every mutant)."""
import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

facts = pytest.mark.facts
NOW = datetime(2026, 7, 10, 11, 0, tzinfo=timezone.utc)  # 12:00 in Dublin
SIGNER, SIGNER_PL = "Siobhán Ní Dhuibhir", "Grzegorz Szczepański"
DUBLIN = ZoneInfo("Europe/Dublin")


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def appt(i=1, starts_at=utc(2026, 7, 20, 13, 30), **kw):  # 14:30 Dublin
    base = {"id": f"A{i}", "customer_name": "Maeve", "email": f"maeve{i}@example.ie",
            "phone": "+353861234567", "country": "IE", "status": "booked",
            "starts_at": starts_at, "optout_sms_v2": False, "opt_out": False}
    return base | kw


def plan(mod, *appts, now=NOW):
    return mod.plan_reminders(list(appts), now)


def times(rems):
    """Distinct send times (a duplicate on another channel is f_no_both's test)."""
    return sorted({r["send_at"] for r in rems})


@facts("f_entry")
def test_entry_point_and_reminder_shape(mod):
    rems = plan(mod, appt())
    assert rems
    for r in rems:
        assert {"appointment_id", "channel", "send_at", "text"} <= set(r)
        assert r["appointment_id"] == "A1" and r["channel"] in {"sms", "email"}


@facts("f_first", "f_second")
def test_reminders_48h_and_2h_before(mod):
    start = utc(2026, 7, 20, 13, 30)
    assert times(plan(mod, appt(starts_at=start))) == [start - timedelta(hours=48), start - timedelta(hours=2)]


@facts("f_cond_late", "f_first")
def test_late_booking_sends_first_reminder_now(mod):
    start = NOW + timedelta(hours=30)  # 18:00 Dublin next day; the 48h point has passed
    ts = times(plan(mod, appt(starts_at=start)))
    assert ts and ts[0] == NOW  # whatever the second reminder does


@facts("f_no_past", "f_cond_late")
def test_never_in_the_past(mod):
    rems = plan(mod, appt(1, starts_at=NOW + timedelta(hours=1)), appt(2, starts_at=NOW + timedelta(hours=5)),
                appt(3), appt(4, starts_at=NOW - timedelta(hours=3)))
    assert all(r["send_at"] >= NOW for r in rems)
    assert times(r for r in rems if r["appointment_id"] == "A1") == [NOW]


@facts("f_cond_channel", "f_optout_field")
def test_sms_when_phone_and_not_opted_out(mod):
    rems = plan(mod, appt(opt_out=True))  # the stale old field must be ignored
    assert rems and all(any(r["channel"] == "sms" and r["send_at"] == t for r in rems) for t in times(rems))


@facts("f_cond_channel", "f_optout_field")
def test_email_when_opted_out(mod):
    rems = plan(mod, appt(optout_sms_v2=True))
    assert rems and {r["channel"] for r in rems} == {"email"}


@facts("f_cond_channel")
def test_email_when_no_phone(mod):
    rems = plan(mod, appt(phone=None))
    assert rems and {r["channel"] for r in rems} == {"email"}


@facts("f_no_both")
def test_never_both_channels_for_the_same_reminder(mod):
    rems = plan(mod, appt(1), appt(2, optout_sms_v2=True), appt(3, phone=None))
    per_reminder = Counter((r["appointment_id"], r["send_at"]) for r in rems)
    assert rems and max(per_reminder.values()) == 1


@facts("f_sms_len")
def test_sms_at_most_160_chars(mod):
    rems = plan(mod, appt(1), appt(2, customer_name="Saoirse Bernadette Ní Chonchobhair-Wiśniewska", country="PL"))
    assert rems and all(len(r["text"]) <= 160 for r in rems if r["channel"] == "sms")


@facts("f_priority_time", "f_sms_len")
def test_long_name_is_shortened_but_time_kept(mod):
    rems = plan(mod, appt(customer_name="Bartholomew-" * 20))
    sms = [r for r in rems if r["channel"] == "sms"]  # channel choice is another fact's test
    assert rems
    for r in sms:
        assert len(r["text"]) <= 160
        assert "14:30" in r["text"] or "13:30" in r["text"]  # the time survives, in either zone


def _all_moved_to_20h(rems, start):
    """Every reminder is at 20:00 Dublin, before the appointment, within a day of it or the 48h
    point (any lead time between 1 and 3 h / 24 and 48 h lands in the quiet window here)."""
    assert rems
    for r in rems:
        local = r["send_at"].astimezone(DUBLIN)
        assert (local.hour, local.minute) == (20, 0) and r["send_at"] < start


@facts("f_quiet", "f_low_tz")
def test_early_morning_reminders_move_to_previous_evening(mod):
    start = utc(2026, 7, 20, 6, 30)  # 07:30 Dublin: both reminders fall inside 21:00-08:00
    _all_moved_to_20h(plan(mod, appt(starts_at=start)), start)


@facts("f_quiet", "f_low_tz")
def test_late_evening_reminders_move_back_to_20h(mod):
    start = utc(2026, 7, 20, 22, 30)  # 23:30 Dublin: both reminders fall inside 21:00-08:00
    _all_moved_to_20h(plan(mod, appt(starts_at=start)), start)


@facts("f_low_tz")
def test_text_shows_dublin_time(mod):
    rems = plan(mod, appt())  # 13:30 UTC = 14:30 Dublin; whether a time is shown at all is f_priority_time's
    shown = [t for r in rems for t in re.findall(r"\b\d{2}:\d{2}\b", r["text"])]
    assert rems and "13:30" not in shown and all(t == "14:30" for t in shown)


@facts("f_no_cancelled")
def test_no_reminders_for_cancelled(mod):
    rems = plan(mod, appt(1, status="cancelled"), appt(2))
    assert {r["appointment_id"] for r in rems} == {"A2"}


@facts("f_no_email_log")
def test_email_address_never_logged_or_printed(mod, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    plan(mod, appt(1), appt(2, optout_sms_v2=True), appt(3, phone=None))
    out = capsys.readouterr()
    for i in (1, 2, 3):
        assert f"maeve{i}@example.ie" not in caplog.text + out.out + out.err


@facts("f_signer")
def test_signed_by_siobhan(mod):
    rems = plan(mod, appt(), appt(2, optout_sms_v2=True))
    assert rems and all(r["text"].rstrip().endswith(SIGNER) for r in rems)


@facts("f_signer_pl", "f_cond_pl")
def test_polish_customers_signed_by_grzegorz(mod):
    rems = plan(mod, appt(1, country="PL"), appt(2))
    pl = [r for r in rems if r["appointment_id"] == "A1"]
    ie = [r for r in rems if r["appointment_id"] == "A2"]
    assert pl and all(r["text"].rstrip().endswith(SIGNER_PL) and SIGNER not in r["text"] for r in pl)
    assert ie and all(SIGNER_PL not in r["text"] for r in ie)  # the default signer is f_signer's test


@facts("f_low_sorted")
def test_sorted_by_send_at(mod):
    rems = plan(mod, appt(1, starts_at=utc(2026, 7, 25, 13, 0)), appt(2, starts_at=NOW + timedelta(hours=30)),
                appt(3, starts_at=utc(2026, 7, 14, 10, 0)))
    assert rems and [r["send_at"] for r in rems] == sorted(r["send_at"] for r in rems)
