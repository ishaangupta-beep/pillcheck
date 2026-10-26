"""Domain logic shared by the agent and the API: references, dose slots, supply.

Arithmetic lives here, never in the language model. The agent is told the
facts ("2 pills newly removed; the 09:00 slot needed 1") and decides what to
do about them.
"""
from __future__ import annotations

import io
import threading
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import cv2 as cv
import numpy as np

from .config import settings
from .storage import Store, new_id
from .vision.models import features_from_npz, features_to_npz
from .vision.pipeline import ReferenceBundle
from .vision.pockets import Pocket

# ---------------------------------------------------------------- time


def parse_at(at: str | None) -> datetime:
    """`at` lets tests and the demo simulate time; production uses the clock."""
    if not at:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(at.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def tz_of(patient: dict) -> ZoneInfo:
    return ZoneInfo(patient.get("timezone") or settings.default_timezone)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- references

_bundle_cache: "OrderedDict[str, ReferenceBundle]" = OrderedDict()
_cache_lock = threading.Lock()


def save_reference(store: Store, medication: dict, reg: dict) -> dict:
    ref_id = new_id("ref")
    ok, png = cv.imencode(".png", reg["image"])
    buf = io.BytesIO()
    np.savez_compressed(buf, **features_to_npz(reg["features"]))
    image_key = store.put_blob(f"references/{ref_id}/image.png", png.tobytes(), "image/png")
    feat_key = store.put_blob(f"references/{ref_id}/features.npz", buf.getvalue(), "application/octet-stream")
    preview_key = store.put_blob(f"references/{ref_id}/preview.jpg", reg["preview_jpeg"], "image/jpeg")
    doc = {
        "id": ref_id,
        "patient_id": medication["patient_id"],
        "medication_id": medication["id"],
        "pockets": [p.to_dict() for p in reg["pockets"]],
        "pocket_confidence": reg["pocket_confidence"],
        "strip_polygon": np.round(reg["strip_polygon"], 1).tolist(),
        "image_size": list(reg["features"].image_size),
        "feature_count": reg["feature_count"],
        "engine": reg["engine"],
        "quality": reg["quality"],
        "expiry_read": reg.get("expiry"),
        "confirmed": False,
        "image_key": image_key,
        "features_key": feat_key,
        "preview_key": preview_key,
    }
    return store.put("references", doc)


def load_bundle(store: Store, ref: dict) -> ReferenceBundle:
    key = f"{ref['id']}:{len(ref['pockets'])}:{ref.get('updated_at', '')}"
    with _cache_lock:
        if key in _bundle_cache:
            _bundle_cache.move_to_end(key)
            return _bundle_cache[key]
    img = cv.imdecode(np.frombuffer(store.get_blob(ref["image_key"]), np.uint8), cv.IMREAD_COLOR)
    feats = features_from_npz(np.load(io.BytesIO(store.get_blob(ref["features_key"]))))
    bundle = ReferenceBundle(ref["medication_id"], img, feats, [Pocket.from_dict(p) for p in ref["pockets"]],
                             np.float32(ref["strip_polygon"]) if ref.get("strip_polygon") else None)
    with _cache_lock:
        _bundle_cache[key] = bundle
        while len(_bundle_cache) > 32:
            _bundle_cache.popitem(last=False)
    return bundle


def active_references(store: Store, patient_id: str) -> list[tuple[dict, dict]]:
    """(medication, reference) for every active medication with a confirmed strip."""
    out = []
    for med in store.find("medications", patient_id, status="active"):
        rid = med.get("active_reference_id")
        ref = store.get("references", rid) if rid else None
        if ref and ref.get("confirmed"):
            out.append((med, ref))
    return out


# ---------------------------------------------------------------- pocket state


def last_pocket_states(store: Store, patient_id: str, medication_id: str, reference_id: str) -> dict[int, str]:
    """Pocket states from the most recent trusted check on this strip."""
    checks = [c for c in store.find("checks", patient_id, medication_id=medication_id)
              if c.get("reference_id") == reference_id and c.get("trusted")]
    if not checks:
        ref = store.get("references", reference_id)
        return {p["id"]: ("popped" if p.get("initially_empty") else "intact") for p in (ref or {}).get("pockets", [])}
    return {int(k): v for k, v in checks[-1]["pocket_states"].items()}


def merge_states(previous: dict[int, str], readings: list[dict]) -> dict[int, str]:
    """New trusted state: confident readings override; unsure pockets keep their last value."""
    out = dict(previous)
    for r in readings:
        if r["state"] in ("intact", "popped"):
            out[r["id"]] = r["state"]
    return out


# ---------------------------------------------------------------- dose slots


def slot_id(medication_id: str, day: date, hhmm: str) -> str:
    return f"slot_{medication_id}_{day.isoformat()}_{hhmm.replace(':', '')}"


def slots_for(medication: dict, patient: dict, days: list[date]) -> list[dict]:
    tz = tz_of(patient)
    out = []
    for d in days:
        for entry in medication.get("schedule", []):
            hh, mm = (int(x) for x in entry["time"].split(":"))
            at = datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz)
            out.append({"slot_id": slot_id(medication["id"], d, entry["time"]), "medication_id": medication["id"],
                        "time": entry["time"], "date": d.isoformat(), "at": at, "pills": int(entry.get("pills", 1))})
    return sorted(out, key=lambda s: s["at"])


def slot_record(store: Store, patient_id: str, slot: dict) -> dict:
    """The stored dose record for a slot (created lazily, status 'pending')."""
    doc = store.get("doses", slot["slot_id"])
    if doc is None:
        doc = {"id": slot["slot_id"], "patient_id": patient_id, "medication_id": slot["medication_id"],
               "slot_at": iso(slot["at"]), "time": slot["time"], "date": slot["date"], "pills_due": slot["pills"],
               "pills_taken": 0, "status": "pending", "check_ids": []}
    return doc


def day_slots(store: Store, patient: dict, medication: dict, now: datetime, back_days: int = 1) -> list[dict]:
    today = now.astimezone(tz_of(patient)).date()
    days = [today - timedelta(days=i) for i in range(back_days, -1, -1)]
    start = medication.get("created_at", "")
    out = []
    for s in slots_for(medication, patient, days):
        if iso(s["at"]) < start:  # slots before the medication was added do not exist
            continue
        rec = slot_record(store, patient["id"], s)
        out.append({**s, "record": rec})
    return out


def reconcile(store: Store, patient: dict, medication: dict, newly_popped: int, now: datetime) -> dict:
    """Assign newly removed pills to the slots they most plausibly belong to.

    Eligible slots: not yet fully taken, and not more than the early window
    ahead of now. Most recent first: a pill taken at 20:40 belongs to the
    20:30 dose, not to the 08:30 dose that was already reported missed; only
    additional pills fill older open slots (recorded as late). Pills that fit
    no slot are 'extra' (possible double dose).
    """
    early = timedelta(minutes=settings.agent.early_window_min)
    max_late = timedelta(minutes=settings.agent.max_late_min)
    eligible = [s for s in day_slots(store, patient, medication, now)
                if s["record"]["pills_taken"] < s["pills"] and s["at"] - early <= now <= s["at"] + max_late]
    eligible.sort(key=lambda s: s["at"], reverse=True)
    remaining = newly_popped
    assigned = []
    for s in eligible:
        if remaining <= 0:
            break
        need = s["pills"] - s["record"]["pills_taken"]
        take = min(need, remaining)
        remaining -= take
        late = now > s["at"] + timedelta(minutes=settings.agent.caregiver_after_min)
        assigned.append({"slot_id": s["slot_id"], "time": s["time"], "date": s["date"], "pills": take,
                         "complete": take == need, "late": late})
    upcoming = [s for s in day_slots(store, patient, medication, now, back_days=0)
                if s["record"]["pills_taken"] < s["pills"] and s["at"] - early > now]
    still_due = [{"slot_id": s["slot_id"], "time": s["time"], "date": s["date"]} for s in eligible
                 if s["slot_id"] not in {a["slot_id"] for a in assigned if a["complete"]}]
    return {"assigned": assigned, "extra_pills": max(0, remaining), "still_due": still_due,
            "next_slot": ({"time": upcoming[0]["time"], "date": upcoming[0]["date"]} if upcoming else None)}


def apply_reconciliation(store: Store, patient: dict, medication: dict, rec: dict, check_id: str, now: datetime) -> list[dict]:
    written = []
    slots = {s["slot_id"]: s for s in day_slots(store, patient, medication, now)}
    for a in rec["assigned"]:
        doc = slots[a["slot_id"]]["record"]
        doc["pills_taken"] += a["pills"]
        doc["status"] = ("late" if a["late"] else "taken") if doc["pills_taken"] >= doc["pills_due"] else "partial"
        doc["check_ids"] = doc.get("check_ids", []) + [check_id]
        doc["taken_at"] = iso(now)
        written.append(store.put("doses", doc))
    if rec["extra_pills"]:
        written.append(store.put("doses", {
            "id": new_id("extra"), "patient_id": patient["id"], "medication_id": medication["id"],
            "status": "extra", "pills_taken": rec["extra_pills"], "check_ids": [check_id], "taken_at": iso(now),
            "created_at": iso(now)}))
    return written


# ---------------------------------------------------------------- supply


def supply(store: Store, patient: dict, medication: dict) -> dict:
    rid = medication.get("active_reference_id")
    ref = store.get("references", rid) if rid else None
    if not ref or not ref.get("confirmed"):
        return {"pills_left": None, "days_left": None}
    states = last_pocket_states(store, patient["id"], medication["id"], rid)
    left = sum(1 for v in states.values() if v == "intact")
    per_day = sum(int(e.get("pills", 1)) for e in medication.get("schedule", [])) or 1
    return {"pills_left": left, "pills_per_day": per_day, "days_left": round(left / per_day, 1)}


def expiry_status(patient: dict, medication: dict, now: datetime) -> str | None:
    """expired | expiring_soon | ok | None when the strip's date is not known."""
    from .vision.expiry import status
    return status((medication.get("expiry") or {}).get("date"), now.astimezone(tz_of(patient)).date(),
                  settings.agent.expiry_soon_days)
