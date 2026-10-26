"""Guardrails between the planner and the world.

Every proposed tool call passes validate() before it runs; a blocked call is
recorded in the trace with the reason. After the planner finishes, the safety
floor compares what ran against the rule planner's mandatory actions and runs
anything essential that was skipped (e.g. a caregiver alert for a possible
double dose). A model can add judgement and better wording; it cannot remove
a safety action or invent a dose.
"""
from __future__ import annotations

import re
from datetime import timedelta

from .. import domain
from .context import AgentContext
from .tools import TOOL_NAMES

MAX_MESSAGE_CHARS = 500
_DOSING_ADVICE = [
    re.compile(r"\b(take|have|swallow)\b[^.!?]{0,30}\b(another|extra|double|second|more|additional|two)\b", re.I),
    re.compile(r"\b(skip|stop|increase|decrease|reduce|halve)\b[^.!?]{0,25}\b(dose|doses|tablet|tablets|pill|pills|medicine|medication)\b", re.I),
    re.compile(r"\b\d+\s?(mg|ml|mcg)\b.*\b(instead|more|less)\b", re.I),
]


_PHOTO_PHRASE = re.compile(r"\b(take|send|click)\b[^.!?]{0,25}\b(photos?|pictures?|pics?|shots?)\b", re.I)


def gives_dosing_advice(text: str) -> bool:
    # "Please take one more photo" is a capture instruction, not a dose.
    text = _PHOTO_PHRASE.sub(" ", text)
    return any(p.search(text) for p in _DOSING_ADVICE)


def validate(ctx: AgentContext, name: str, args: dict) -> tuple[bool, str]:
    if name not in TOOL_NAMES:
        return False, "unknown tool"

    if name == "record_dose":
        mid = args.get("medication_id")
        if not ctx.trusted:
            return False, "photo is not trusted for a dose record"
        rec = ctx.reconciliation.get(mid)
        if rec is None:
            return False, "no reconciliation for this medication (it is not the strip in the photo)"
        if not rec["assigned"] and not rec["extra_pills"]:
            return False, "no newly removed tablets to record"
        if mid in ctx.recorded:
            return False, "already recorded in this run"
        return True, ""

    if name in ("send_patient_message", "request_retake", "alert_caregiver", "request_human_review"):
        text = (args.get("message") or args.get("question") or "").strip()
        if not text:
            return False, "empty message"
        if len(text) > MAX_MESSAGE_CHARS:
            return False, "message too long"
        # Instructions to the patient must never read as dosing advice. Caregiver
        # alerts are reports ("possible double dose") and are exempt.
        if name in ("send_patient_message", "request_retake") and gives_dosing_advice(text):
            return False, "message reads as dosing advice"
        # No repeated message within one run.
        if any(a["type"] == name and (a["payload"].get("message") or a["payload"].get("question")) == text
               for a in ctx.actions):
            return False, "same message already sent in this run"

    if name == "alert_caregiver" and ctx.check is not None:
        # Photo-triggered alerts: one per reason per hour (a patient re-sending
        # photos should not page the caregiver repeatedly). Schedule alerts are
        # de-duplicated per dose slot instead.
        since = domain.iso(ctx.now - timedelta(minutes=60))
        recent = [a for a in ctx.store.find("actions", ctx.patient["id"], type="alert_caregiver")
                  if a["created_at"] >= since and a["payload"].get("reason") == args.get("reason")
                  and a["source"].get("event") == "photo_check"]
        if recent and args.get("reason") != "extra_removed":
            return False, "caregiver already alerted for this in the last hour"

    if name == "request_human_review":
        if args.get("kind") not in ("confirm_dose", "new_strip", "other"):
            return False, "invalid review kind"

    if name == "draft_refill_request":
        mid = args.get("medication_id")
        med = ctx.medications.get(mid)
        if not med:
            return False, "unknown medication"
        sup = domain.supply(ctx.store, ctx.patient, med)
        if sup["days_left"] is None or sup["days_left"] > 2 * ctx_threshold():
            return False, "supply is not low"
        if ctx.store.find("actions", ctx.patient["id"], type="draft_refill_request", status="pending_approval", medication_id=mid):
            return False, "a refill request is already pending"
    return True, ""


def ctx_threshold() -> int:
    from ..config import settings
    return settings.agent.refill_days_threshold


def missing_mandatory(ctx: AgentContext, mandatory: list[dict]) -> list[dict]:
    """Mandatory rule actions whose effect the planner did not achieve."""
    done_types = {(a["type"], a["payload"].get("reason")) for a in ctx.actions}
    out = []
    for m in mandatory:
        key = (m["tool"], m["args"].get("reason"))
        if m["tool"] == "record_dose":
            if m["args"]["medication_id"] not in ctx.recorded:
                out.append(m)
        elif m["tool"] in ("request_retake", "draft_refill_request"):
            if not any(a["type"] == m["tool"] for a in ctx.actions):
                out.append(m)
        elif m["tool"] == "request_human_review":
            if not any(a["type"] == m["tool"] and a["payload"].get("kind") == m["args"]["kind"] for a in ctx.actions):
                out.append(m)
        elif key not in done_types:
            out.append(m)
    return out
