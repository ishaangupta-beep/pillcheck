"""The actions the agent can take. Each one is a small, auditable side effect.

Tools never accept numbers the model could get wrong: record_dose takes only a
medication id and applies the reconciliation computed from the photo; the
refill draft reads the pill count from the last trusted check.
"""
from __future__ import annotations

from .. import domain, notify
from ..storage import new_id
from .context import AgentContext
from .messages import caregiver_text, fmt_time

TOOL_SPECS = [
    {
        "name": "record_dose",
        "description": "Record the tablets the photo shows were newly removed, for one medication. "
                       "The pill-to-slot assignment is already computed (see reconciliation); only call this when the photo is trusted.",
        "input_schema": {"type": "object", "properties": {"medication_id": {"type": "string"}}, "required": ["medication_id"]},
    },
    {
        "name": "send_patient_message",
        "description": "Send a short, kind message to the patient in their language: a confirmation, a reminder, or information. Never give dosing advice.",
        "input_schema": {"type": "object", "properties": {
            "message": {"type": "string"},
            "kind": {"type": "string", "enum": ["confirmation", "reminder", "information"]}},
            "required": ["message", "kind"]},
    },
    {
        "name": "request_retake",
        "description": "Ask the patient for a new photo, with concrete capture instructions (from capture_guidance).",
        "input_schema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
    },
    {
        "name": "alert_caregiver",
        "description": "Notify the caregiver about something they should act on.",
        "input_schema": {"type": "object", "properties": {
            "message": {"type": "string"},
            "severity": {"type": "string", "enum": ["info", "warning", "urgent"]},
            "reason": {"type": "string", "enum": ["extra_removed", "wrong_medication", "missed", "photo_reuse", "expired", "expiring_soon", "other"]}},
            "required": ["message", "severity", "reason"]},
    },
    {
        "name": "request_human_review",
        "description": "Ask the caregiver to decide something the camera cannot settle (unclear pockets after a retake, a strip that looks replaced).",
        "input_schema": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["confirm_dose", "new_strip", "other"]},
            "question": {"type": "string"},
            "medication_id": {"type": "string"}},
            "required": ["kind", "question"]},
    },
    {
        "name": "draft_refill_request",
        "description": "Draft a refill reminder for a medication that is running low. It waits for caregiver approval.",
        "input_schema": {"type": "object", "properties": {"medication_id": {"type": "string"}}, "required": ["medication_id"]},
    },
]

TOOL_NAMES = {t["name"] for t in TOOL_SPECS}


def _action(ctx: AgentContext, type_: str, payload: dict, *, status: str = "done", medication_id: str | None = None) -> dict:
    doc = {
        "id": new_id("act"),
        "patient_id": ctx.patient["id"],
        "type": type_,
        "status": status,
        "payload": payload,
        "medication_id": medication_id,
        "source": {"event": ctx.event, "check_id": ctx.check["id"] if ctx.check else None},
        "created_at": domain.iso(ctx.now),
    }
    ctx.store.put("actions", doc)
    ctx.actions.append(doc)
    return doc


def execute(ctx: AgentContext, name: str, args: dict) -> dict:
    store, pid = ctx.store, ctx.patient["id"]
    related = {"check_id": ctx.check["id"]} if ctx.check else {}
    at = domain.iso(ctx.now)

    if name == "record_dose":
        mid = args["medication_id"]
        rec = ctx.reconciliation[mid]
        written = domain.apply_reconciliation(store, ctx.patient, ctx.medications[mid], rec, ctx.check["id"], ctx.now)
        ctx.recorded.add(mid)
        summary = [{"time": fmt_time(d["time"]), "status": d["status"]} for d in written if d.get("time")]
        _action(ctx, "record_dose", {"slots": summary, "extra_pills": rec["extra_pills"]}, medication_id=mid)
        return {"recorded": summary, "extra_pills": rec["extra_pills"]}

    if name in ("send_patient_message", "request_retake"):
        kind = args.get("kind", "retake_request") if name == "send_patient_message" else "retake_request"
        note = notify.send(store, pid, "patient", args["message"], kind=kind, related=related, at=at)
        _action(ctx, name, {"message": args["message"], "notification_id": note["id"]})
        return {"sent": True}

    if name == "alert_caregiver":
        note = notify.send(store, pid, "caregiver", args["message"], severity=args["severity"],
                           kind=args["reason"], related=related, at=at)
        _action(ctx, "alert_caregiver", {"message": args["message"], "severity": args["severity"],
                                         "reason": args["reason"], "notification_id": note["id"]})
        return {"sent": True, "channels": note["channels"]}

    if name == "request_human_review":
        act = _action(ctx, "request_human_review", {"kind": args["kind"], "question": args["question"], **related},
                      status="pending_approval", medication_id=args.get("medication_id"))
        notify.send(store, pid, "caregiver", args["question"], severity="warning", kind="review_needed",
                    related={**related, "action_id": act["id"]}, at=at)
        return {"pending": True, "action_id": act["id"]}

    if name == "draft_refill_request":
        mid = args["medication_id"]
        sup = domain.supply(store, ctx.patient, ctx.medications[mid])
        text = caregiver_text("refill", med=ctx.med_name(mid), patient=ctx.patient["name"],
                              left=sup["pills_left"], days=sup["days_left"])
        act = _action(ctx, "draft_refill_request", {"message": text, **sup}, status="pending_approval", medication_id=mid)
        notify.send(store, pid, "caregiver", text, severity="info", kind="refill", related={"action_id": act["id"]}, at=at)
        return {"pending": True, "action_id": act["id"], **sup}

    raise ValueError(f"unknown tool {name}")
