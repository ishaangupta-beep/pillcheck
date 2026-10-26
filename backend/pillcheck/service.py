"""Application operations used by the API (and by scripts and tests)."""
from __future__ import annotations

import logging
import re
from datetime import timedelta

import cv2 as cv
import numpy as np

from . import domain, notify
from .agent import runner
from .agent.context import AgentContext
from .storage import Store, new_id
from .vision import pipeline
from .vision.annotate import encode_jpeg, reference_preview
from .vision.pockets import Pocket, detect_strip_polygon

log = logging.getLogger("pillcheck.service")
HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


def _get(store: Store, collection: str, doc_id: str) -> dict:
    doc = store.get(collection, doc_id)
    if doc is None:
        raise NotFound(f"{collection[:-1]} {doc_id} not found")
    return doc


# ---------------------------------------------------------------- patients & medications


def create_patient(store: Store, data: dict) -> dict:
    doc = {"id": new_id("pat"), "name": data["name"].strip(), "language": data.get("language", "en"),
           "timezone": data.get("timezone") or domain.settings.default_timezone,
           "caregiver": data.get("caregiver") or {}}
    doc["patient_id"] = doc["id"]
    domain.tz_of(doc)  # validates the timezone name
    return store.put("patients", doc)


def create_medication(store: Store, patient_id: str, data: dict) -> dict:
    _get(store, "patients", patient_id)
    schedule = data.get("schedule") or []
    if not schedule:
        raise ValueError("schedule needs at least one time")
    for e in schedule:
        if not HHMM.match(e.get("time", "")):
            raise ValueError(f"invalid time {e.get('time')!r}, use HH:MM (24h)")
        if int(e.get("pills", 1)) < 1:
            raise ValueError("pills must be at least 1")
    doc = {"id": new_id("med"), "patient_id": patient_id, "name": data["name"].strip(),
           "strength": (data.get("strength") or "").strip(), "instructions": data.get("instructions", ""),
           "schedule": sorted([{"time": e["time"], "pills": int(e.get("pills", 1))} for e in schedule], key=lambda e: e["time"]),
           "status": "active", "active_reference_id": None}
    return store.put("medications", doc)


# ---------------------------------------------------------------- reference strips


def upload_reference(store: Store, medication_id: str, image_bytes: bytes) -> dict:
    med = _get(store, "medications", medication_id)
    reg = pipeline.register_reference(image_bytes)
    if reg["feature_count"] < 100:
        raise ValueError("The photo has too little detail to recognise the strip later. Photograph the printed side, closer, in good light.")
    ref = domain.save_reference(store, med, reg)
    store.update("medications", med["id"], pending_reference_id=ref["id"])
    return {**ref, "elapsed_ms": reg["elapsed_ms"]}


def confirm_reference(store: Store, reference_id: str, pockets: list[dict] | None, expiry: str | None = None) -> dict:
    ref = _get(store, "references", reference_id)
    if pockets is not None:
        if not pockets:
            raise ValueError("at least one pocket is required")
        w, h = ref["image_size"]
        cleaned = []
        for i, p in enumerate(pockets):
            q = Pocket(i, float(p["cx"]), float(p["cy"]), float(p["rx"]), float(p["ry"]), float(p.get("angle", 0)),
                       bool(p.get("initially_empty", False)))
            if not (0 <= q.cx < w and 0 <= q.cy < h and 3 <= q.rx < w / 2 and 3 <= q.ry < h / 2):
                raise ValueError(f"pocket {i + 1} is outside the image or has an invalid size")
            cleaned.append(q)
        img = cv.imdecode(np.frombuffer(store.get_blob(ref["image_key"]), np.uint8), cv.IMREAD_COLOR)
        poly = detect_strip_polygon(img, cleaned)
        ref["pockets"] = [q.to_dict() for q in cleaned]
        ref["strip_polygon"] = np.round(poly, 1).tolist()
        store.put_blob(ref["preview_key"], encode_jpeg(reference_preview(img, cleaned, poly)), "image/jpeg")
    if not ref["pockets"]:
        raise ValueError("no pockets defined; mark them on the photo first")
    ref["confirmed"] = True
    ref["updated_at"] = domain.iso(domain.parse_at(None))
    store.put("references", ref)
    read = ref.get("expiry_read") or {}
    if expiry is not None:
        exp = _expiry_doc(expiry, "caregiver")
    elif read.get("date") and read.get("confidence") in ("high", "medium"):
        exp = {"date": read["date"], "source": "photo", "confidence": read["confidence"]}
    else:
        exp = None  # unknown: the app asks the caregiver to type it
    store.update("medications", ref["medication_id"], active_reference_id=ref["id"], pending_reference_id=None, expiry=exp)
    return ref


def _expiry_doc(value: str, source: str) -> dict | None:
    value = (value or "").strip()
    if not value:
        return None
    if not re.fullmatch(r"20[2-4]\d-(0[1-9]|1[0-2])", value):
        raise ValueError("expiry must look like 2027-12 (year-month)")
    return {"date": value, "source": source, "confidence": "confirmed"}


def set_expiry(store: Store, medication_id: str, value: str | None) -> dict:
    _get(store, "medications", medication_id)
    return store.update("medications", medication_id, expiry=_expiry_doc(value or "", "caregiver"))


# ---------------------------------------------------------------- checks


def _recent(store: Store, patient_id: str, now, minutes: int, pred) -> int:
    since = domain.iso(now - timedelta(minutes=minutes))
    return sum(1 for c in store.list("checks", patient_id) if c["created_at"] >= since and pred(c))


def submit_check(store: Store, patient_id: str, image_bytes: bytes, expected_medication_id: str | None,
                 at: str | None = None) -> dict:
    patient = _get(store, "patients", patient_id)
    now = domain.parse_at(at)
    pairs = domain.active_references(store, patient_id)
    if not pairs:
        raise Conflict("No strip is registered yet. Register a reference photo for each medicine first.")
    if expected_medication_id and expected_medication_id not in {m["id"] for m, _ in pairs}:
        raise ValueError("expected medication is not an active medication with a registered strip")
    bundles = [domain.load_bundle(store, ref) for _, ref in pairs]
    refs_by_med = {m["id"]: ref for m, ref in pairs}
    prev_states = {m["id"]: domain.last_pocket_states(store, patient_id, m["id"], ref["id"]) for m, ref in pairs}
    prev_fps = [{"check_id": c["id"], **c["fingerprint"]} for c in store.list("checks", patient_id)[-300:] if c.get("fingerprint")]

    perception = pipeline.analyze_check(image_bytes, bundles, expected_medication_id, prev_states, prev_fps)
    evidence = perception.pop("_evidence_jpeg", None)
    fingerprint = perception.pop("fingerprint", None)

    check_id = new_id("chk")
    photo_key = store.put_blob(f"checks/{check_id}/photo.jpg", image_bytes, "image/jpeg")
    evidence_key = store.put_blob(f"checks/{check_id}/evidence.jpg", evidence, "image/jpeg") if evidence else None
    mid = perception.get("medication_id")
    trusted = perception["status"] in ("ok", "wrong_medication") and not perception.get("inconsistent")
    unclear = any(p["state"] in ("uncertain", "unknown") for p in perception.get("pockets", []))
    check = {
        "id": check_id, "patient_id": patient_id, "created_at": domain.iso(now),
        "expected_medication_id": expected_medication_id, "medication_id": mid,
        "reference_id": refs_by_med[mid]["id"] if mid else None,
        "status": perception["status"], "trusted": trusted,
        "pocket_states": domain.merge_states(prev_states[mid], perception["pockets"]) if trusted else None,
        "unclear": unclear and not perception.get("newly_popped"),
        "perception": perception, "fingerprint": fingerprint,
        "photo_key": photo_key, "evidence_key": evidence_key,
        "recent_unclear": _recent(store, patient_id, now, 30, lambda c: c.get("unclear") and c.get("medication_id") == mid) if mid else 0,
        "recent_duplicates": _recent(store, patient_id, now, 24 * 60, lambda c: c["status"] == "duplicate"),
    }
    store.put("checks", check)

    meds = {m["id"]: m for m, _ in pairs}
    ctx = AgentContext(store=store, patient=patient, medications=meds, now=now, event="photo_check",
                       check=check, perception=perception)
    if mid:
        n = len(perception.get("newly_popped", [])) if trusted else 0
        ctx.reconciliation[mid] = domain.reconcile(store, patient, meds[mid], n, now)
    check["agent_run"] = runner.run(ctx)
    store.put("checks", check)
    return check


# ---------------------------------------------------------------- human decisions


def decide_action(store: Store, action_id: str, approve: bool, note: str | None, at: str | None = None) -> dict:
    act = _get(store, "actions", action_id)
    if act["status"] != "pending_approval":
        raise Conflict(f"action is already {act['status']}")
    now = domain.parse_at(at)
    act["status"] = "approved" if approve else "rejected"
    act["decision"] = {"approve": approve, "note": note, "at": domain.iso(now)}
    patient = _get(store, "patients", act["patient_id"])
    kind = act["payload"].get("kind")

    if approve and act["type"] == "request_human_review" and kind == "confirm_dose" and act.get("medication_id"):
        med = _get(store, "medications", act["medication_id"])
        rec = domain.reconcile(store, patient, med, 0, now)
        if rec["still_due"]:
            slot = next(s for s in domain.day_slots(store, patient, med, now) if s["slot_id"] == rec["still_due"][0]["slot_id"])
            need = slot["pills"] - slot["record"]["pills_taken"]
            rec = domain.reconcile(store, patient, med, need, now)
            rec["extra_pills"] = 0
            written = domain.apply_reconciliation(store, patient, med, rec, act["payload"].get("check_id") or act["id"], now)
            act["result"] = {"recorded": [d["id"] for d in written], "by": "caregiver"}
    elif approve and act["type"] == "request_human_review" and kind == "new_strip" and act.get("medication_id"):
        store.update("medications", act["medication_id"], needs_new_reference=True)
        act["result"] = {"next_step": "register the new strip"}
    elif approve and act["type"] == "draft_refill_request":
        med = store.get("medications", act.get("medication_id") or "") or {}
        notify.send(store, patient["id"], "caregiver", f"Refill marked as needed for {med.get('name', 'the medicine')}.",
                    kind="refill_approved", related={"action_id": act["id"]}, at=domain.iso(now))
    store.put("actions", act)
    return act


# ---------------------------------------------------------------- views


def today(store: Store, patient_id: str, at: str | None = None) -> dict:
    patient = _get(store, "patients", patient_id)
    now = domain.parse_at(at)
    meds = []
    for med in store.find("medications", patient_id, status="active"):
        slots = domain.day_slots(store, patient, med, now, back_days=0)
        meds.append({
            "medication": med,
            "slots": [{"time": s["time"], "slot_at": domain.iso(s["at"]), "pills": s["pills"],
                       "status": s["record"]["status"], "pills_taken": s["record"]["pills_taken"]} for s in slots],
            "supply": domain.supply(store, patient, med),
            "has_strip": bool(med.get("active_reference_id")),
            "expiry": {**(med.get("expiry") or {}), "status": domain.expiry_status(patient, med, now)},
        })
    pending = [a for a in store.list("actions", patient_id) if a["status"] == "pending_approval"]
    return {"patient": patient, "now": domain.iso(now),
            "local_date": now.astimezone(domain.tz_of(patient)).date().isoformat(),
            "medications": meds, "pending_actions": pending}


def check_summary(c: dict) -> dict:
    p = c.get("perception", {})
    return {"id": c["id"], "created_at": c["created_at"], "status": c["status"], "medication_id": c.get("medication_id"),
            "expected_medication_id": c.get("expected_medication_id"),
            "newly_removed": len(p.get("newly_popped", [])), "counts": p.get("counts"),
            "has_evidence": bool(c.get("evidence_key")),
            "agent_summary": (c.get("agent_run") or {}).get("summary"),
            "agent_mode": (c.get("agent_run") or {}).get("mode")}


def timeline(store: Store, patient_id: str, limit: int = 100) -> list[dict]:
    _get(store, "patients", patient_id)
    # `entry` says what each item is; it never collides with fields of the documents.
    items = [{**check_summary(c), "entry": "check", "at": c["created_at"]} for c in store.list("checks", patient_id)]
    items += [{**n, "entry": "notification", "at": n["created_at"]} for n in store.list("notifications", patient_id)]
    items += [{**d, "entry": "dose", "at": d.get("taken_at") or d.get("caregiver_alerted_at") or d["created_at"]}
              for d in store.list("doses", patient_id) if d["status"] != "pending"]
    items += [{**a, "entry": "action", "at": a["created_at"]} for a in store.list("actions", patient_id)]
    return sorted(items, key=lambda x: x["at"], reverse=True)[:limit]


def tick_all(store: Store, at: str | None = None) -> list[dict]:
    now = domain.parse_at(at)
    runs = []
    for patient in store.list("patients"):
        if patient.get("demo"):  # demo sandboxes run on simulated time only
            continue
        try:
            r = runner.run_tick(store, patient, now)
            if r:
                runs.append(r)
        except Exception:
            log.exception("tick failed for %s", patient["id"])
    return runs


def tick_patient(store: Store, patient_id: str, at: str | None = None) -> dict | None:
    """Run the scheduler for one patient (demo: lets a visitor fast-forward time)."""
    patient = _get(store, "patients", patient_id)
    return runner.run_tick(store, patient, domain.parse_at(at))
