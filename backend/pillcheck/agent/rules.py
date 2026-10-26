"""Deterministic planner: what a careful human operator would do.

It runs on its own when no Bedrock model is configured (offline demo, tests)
and its `mandatory` steps form the safety floor under the Bedrock planner.
"""
from __future__ import annotations

from .context import AgentContext
from datetime import timedelta

from .messages import caregiver_text, fmt_expiry, fmt_time, patient_text
from ..config import settings


def _step(tool: str, args: dict, reason: str, mandatory: bool = False) -> dict:
    return {"tool": tool, "args": args, "reason": reason, "mandatory": mandatory}


def plan(ctx: AgentContext) -> list[dict]:
    return plan_photo(ctx) if ctx.event == "photo_check" else plan_tick(ctx)


def _now_str(ctx: AgentContext) -> str:
    from .. import domain
    return ctx.now.astimezone(domain.tz_of(ctx.patient)).strftime("%d %b, %I:%M %p")


def plan_photo(ctx: AgentContext) -> list[dict]:
    p = ctx.perception or {}
    status = p.get("status")
    lang, name = ctx.lang, ctx.patient["name"]
    steps: list[dict] = []

    if status == "duplicate":
        steps.append(_step("request_retake", {"message": patient_text("duplicate", lang)},
                           "the photo matches an earlier upload, so it cannot prove a new dose", True))
        if (ctx.check or {}).get("recent_duplicates", 0) >= 1:
            steps.append(_step("alert_caregiver", {
                "message": caregiver_text("photo_reuse", patient=name, now=_now_str(ctx)),
                "severity": "info", "reason": "photo_reuse"}, "repeated re-use of old photos", True))
        return steps

    if status == "retake":
        guidance = " ".join(p.get("guidance", [])) or patient_text("unclear", lang)
        steps.append(_step("request_retake", {"message": guidance}, f"capture problem: {', '.join(p.get('quality', {}).get('issues', []) or (p.get('registration') or {}).get('issues', []))}", True))
        return steps

    if status == "unrecognised":
        steps.append(_step("request_retake", {"message": patient_text("unrecognised", lang)},
                           "the strip did not match any registered medicine", True))
        return steps

    mid = p.get("medication_id")
    rec = ctx.reconciliation.get(mid, {})
    med = ctx.med_name(mid)

    if p.get("inconsistent"):
        steps.append(_step("request_human_review", {
            "kind": "new_strip", "medication_id": mid,
            "question": caregiver_text("new_strip", patient=name, med=med)},
            f"{len(p['inconsistent'])} pocket(s) that were empty now look full: probably a new strip", True))
        return steps

    expected = (p.get("identity") or {}).get("expected_medication_id")
    if status == "wrong_medication":
        steps.append(_step("send_patient_message", {
            "kind": "information",
            "message": patient_text("wrong_medication", lang, matched=med, expected=ctx.med_name(expected))},
            "the photographed strip is not the medicine that is due"))

    newly = len(p.get("newly_popped", []))
    unclear = [x for x in p.get("pockets", []) if x["state"] in ("uncertain", "unknown")]

    if newly and ctx.trusted:
        assigned = [fmt_time(a["time"]) for a in rec.get("assigned", [])]
        why = f"{newly} tablet{'s' if newly != 1 else ''} newly removed; " + (
            f"matched to the {', '.join(assigned)} dose" if assigned else "no dose was due")
        if rec.get("extra_pills"):
            why += f"; {rec['extra_pills']} beyond the schedule"
        steps.append(_step("record_dose", {"medication_id": mid}, why, True))
        if rec.get("assigned") and not rec.get("extra_pills"):
            a = rec["assigned"][-1]
            key = "dose_recorded_late" if a["late"] else "dose_recorded"
            steps.append(_step("send_patient_message", {"kind": "confirmation",
                                                        "message": patient_text(key, lang, time=fmt_time(a["time"]), med=med)},
                               "confirm the recorded dose"))
        if rec.get("extra_pills"):
            steps.append(_step("alert_caregiver", {
                "message": caregiver_text("extra_removed", patient=name, n=rec["extra_pills"], med=med, now=_now_str(ctx)),
                "severity": "urgent", "reason": "extra_removed"},
                f"{rec['extra_pills']} tablet{'s' if rec['extra_pills'] != 1 else ''} removed beyond the schedule: possible double dose", True))
            steps.append(_step("send_patient_message", {"kind": "information", "message": patient_text("extra_removed", lang)},
                               "let the patient know the caregiver was told"))
        if status == "wrong_medication" and not rec.get("assigned"):
            steps.append(_step("alert_caregiver", {
                "message": caregiver_text("wrong_medication", patient=name, matched=med, expected=ctx.med_name(expected), now=_now_str(ctx)),
                "severity": "warning", "reason": "wrong_medication"},
                "a tablet of a medicine that was not due was removed", True))

    elif unclear:
        if (ctx.check or {}).get("recent_unclear", 0) >= 1:
            slot = (rec.get("still_due") or [{}])[0]
            steps.append(_step("request_human_review", {
                "kind": "confirm_dose", "medication_id": mid,
                "question": caregiver_text("confirm_dose", patient=name, med=med, time=fmt_time(slot.get("time", "00:00")) if slot else "latest")},
                "the camera could not settle the pocket state after a retake", True))
        else:
            steps.append(_step("request_retake", {"message": patient_text("unclear", lang)},
                               f"pockets {[x['id'] + 1 for x in unclear]} could not be read", True))

    elif status == "ok":
        due = rec.get("still_due") or []
        if due:
            steps.append(_step("send_patient_message", {
                "kind": "reminder", "message": patient_text("not_taken_yet", lang, time=fmt_time(due[0]["time"]), med=med)},
                "a dose is due and the photo shows it has not been taken"))
        elif rec.get("next_slot"):
            steps.append(_step("send_patient_message", {
                "kind": "information", "message": patient_text("nothing_new", lang, time=fmt_time(rec["next_slot"]["time"]), med=med)},
                "nothing is due; tell the patient when the next dose is"))

    steps += _expiry_steps(ctx, [mid] if mid else [], tell_patient=True)
    steps += _refill_steps(ctx, [mid] if mid else [])
    return steps


def _already_alerted(ctx: AgentContext, kind: str, text: str, hours: float) -> bool:
    since = ctx.now - timedelta(hours=hours)
    from .. import domain
    for n in ctx.store.find("notifications", ctx.patient["id"], kind=kind):
        if n.get("message") == text and domain.parse_at(n.get("created_at")) >= since:
            return True
    return False


def _expiry_steps(ctx: AgentContext, medication_ids: list[str], tell_patient: bool = False) -> list[dict]:
    """Expired strip: tell the caregiver (once a day) and the patient. Close to expiry: a heads-up once a week."""
    from .. import domain
    out = []
    for mid in medication_ids:
        med = ctx.medications.get(mid)
        if not med:
            continue
        st = domain.expiry_status(ctx.patient, med, ctx.now)
        if st not in ("expired", "expiring_soon"):
            continue
        exp = fmt_expiry(med["expiry"]["date"])
        text = caregiver_text(st, patient=ctx.patient["name"], med=ctx.med_name(mid), exp=exp)
        if st == "expired":
            if not _already_alerted(ctx, "expired", text, 20):
                out.append(_step("alert_caregiver", {"message": text, "severity": "warning", "reason": "expired"},
                                 f"strip expired ({exp})", True))
            if tell_patient:
                out.append(_step("send_patient_message", {"kind": "information",
                                 "message": patient_text("expired", ctx.lang, med=ctx.med_name(mid), exp=exp)},
                                 "the strip in the photo is past its expiry date", True))
        elif not _already_alerted(ctx, "expiring_soon", text, 24 * 7):
            out.append(_step("alert_caregiver", {"message": text, "severity": "info", "reason": "expiring_soon"},
                             f"strip expires soon ({exp})"))
    return out


def _refill_steps(ctx: AgentContext, medication_ids: list[str]) -> list[dict]:
    from .. import domain
    out = []
    for mid in medication_ids:
        med = ctx.medications.get(mid)
        if not med:
            continue
        sup = domain.supply(ctx.store, ctx.patient, med)
        pending = ctx.store.find("actions", ctx.patient["id"], type="draft_refill_request",
                                 status="pending_approval", medication_id=mid)
        if sup["days_left"] is not None and sup["days_left"] <= settings.agent.refill_days_threshold and not pending:
            out.append(_step("draft_refill_request", {"medication_id": mid},
                             f"{sup['pills_left']} tablet(s) left, about {sup['days_left']} day(s)", True))
    return out


def plan_tick(ctx: AgentContext) -> list[dict]:
    steps: list[dict] = []
    lang, name = ctx.lang, ctx.patient["name"]
    for item in ctx.tick_findings.get("overdue", []):
        steps.append(_step("send_patient_message", {
            "kind": "reminder", "message": patient_text("reminder", lang, time=fmt_time(item["time"]), med=item["med"])},
            f"{item['med']} at {item['time']} is {item['minutes_late']} min overdue", True))
    for item in ctx.tick_findings.get("missed", []):
        steps.append(_step("alert_caregiver", {
            "message": caregiver_text("missed", patient=name, time=fmt_time(item["time"]), med=item["med"], minutes=item["minutes_late"]),
            "severity": "warning", "reason": "missed"},
            f"{item['med']} at {item['time']} still not taken after {item['minutes_late']} min", True))
    steps += _expiry_steps(ctx, list(ctx.medications))
    steps += _refill_steps(ctx, list(ctx.medications))
    return steps
