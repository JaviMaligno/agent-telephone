"""Appointment reminders for Brennan Opticians (reference solution for the hidden tests)."""
from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

TZ = ZoneInfo("Europe/Dublin")
SMS_MAX = 160
QUIET_FROM, QUIET_UNTIL, MOVE_TO = time(21), time(8), time(20)
SIGNER, SIGNER_PL = "Siobhán Ní Dhuibhir", "Grzegorz Szczepański"


def _out_of_quiet_hours(t: datetime) -> datetime:
    """Move a send time inside 21:00-08:00 (Dublin) back to the 20:00 just before it."""
    local = t.astimezone(TZ)
    if local.time() >= QUIET_FROM:
        day = local.date()
    elif local.time() < QUIET_UNTIL:
        day = local.date() - timedelta(days=1)
    else:
        return t
    return datetime.combine(day, MOVE_TO, TZ)


def _text(appt: dict, channel: str) -> str:
    start = appt["starts_at"].astimezone(TZ)
    signer = SIGNER_PL if appt.get("country") == "PL" else SIGNER

    def build(name: str) -> str:
        return (f"Hi {name}, a reminder of your Brennan Opticians appointment on "
                f"{start:%a %d %b} at {start:%H:%M}. — {signer}")

    text = build(appt["customer_name"])
    if channel == "sms" and len(text) > SMS_MAX:  # shorten the name, never the date/time
        text = build(appt["customer_name"][: max(SMS_MAX - len(build("")), 0)])
    return text


def plan_reminders(appointments: list[dict], now: datetime) -> list[dict]:
    reminders = []
    for appt in appointments:
        start = appt["starts_at"]
        if appt.get("status") == "cancelled" or start <= now:
            continue
        first = _out_of_quiet_hours(start - timedelta(hours=48))
        second = _out_of_quiet_hours(start - timedelta(hours=2))
        times = {first if first > now else now}
        if second > now:
            times.add(second)
        channel = "sms" if appt.get("phone") and not appt.get("optout_sms_v2") else "email"
        text = _text(appt, channel)
        reminders += [{"appointment_id": appt["id"], "channel": channel, "send_at": t, "text": text}
                      for t in times]
        log.info("planned %d reminder(s) by %s for appointment %s", len(times), channel, appt["id"])
    return sorted(reminders, key=lambda r: r["send_at"])
