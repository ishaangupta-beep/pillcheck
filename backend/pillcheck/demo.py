"""Public demo mode: ready-made sandbox patients and sample photos.

A judge opening the public URL has no medicine strip, so a demo session gives
them a patient with two registered medicines and a set of sample photos to
submit. Reference strips are registered once (template) and every session's
reference documents point at the same stored images and features, so starting
a session costs no vision work.
"""
from __future__ import annotations

import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path

from . import domain, service
from .config import BACKEND_ROOT
from .storage import Store, new_id
from .vision import pipeline

ASSETS = BACKEND_ROOT / "demo_assets"
_lock = threading.Lock()


@lru_cache(maxsize=1)
def manifest() -> dict:
    path = ASSETS / "manifest.json"
    if not path.exists():
        raise service.NotFound("demo assets are not installed (run scripts/make_demo_assets.py)")
    return json.loads(path.read_text())


def _template_id() -> str:
    m = manifest()
    digest = hashlib.sha1(json.dumps(m["medications"], sort_keys=True).encode())
    for med in m["medications"]:
        digest.update((ASSETS / med["reference"]).read_bytes())
    digest.update(b"template-v3:real-photos")
    return f"demo_template_{digest.hexdigest()[:12]}"


def ensure_template(store: Store) -> dict:
    """Register the demo reference strips once; reuse them for every session."""
    tid = _template_id()
    doc = store.get("demo_templates", tid)
    if doc:
        return doc
    with _lock:
        doc = store.get("demo_templates", tid)
        if doc:
            return doc
        refs = {}
        for med in manifest()["medications"]:
            reg = pipeline.register_reference((ASSETS / med["reference"]).read_bytes())
            if med.get("pockets"):  # labelled pockets shipped with real demo photos
                from .vision.annotate import encode_jpeg, reference_preview
                from .vision.pockets import Pocket, detect_strip_polygon
                reg["pockets"] = [Pocket.from_dict(p) for p in med["pockets"]]
                reg["pocket_confidence"] = 1.0
                reg["strip_polygon"] = detect_strip_polygon(reg["image"], reg["pockets"])
                reg["preview_jpeg"] = encode_jpeg(reference_preview(reg["image"], reg["pockets"], reg["strip_polygon"]))
            fake_med = {"id": f"template_{med['key']}", "patient_id": tid}
            ref = domain.save_reference(store, fake_med, reg)
            refs[med["key"]] = {k: ref[k] for k in ("pockets", "pocket_confidence", "strip_polygon", "image_size",
                                                    "feature_count", "engine", "quality", "image_key",
                                                    "features_key", "preview_key", "expiry_read")}
        doc = {"id": tid, "references": refs}
        return store.put("demo_templates", doc)


def create_session(store: Store, language: str = "en", name: str | None = None) -> dict:
    template = ensure_template(store)
    patient = service.create_patient(store, {"name": name or "Asha", "language": language,
                                             "caregiver": {"name": "You", "email": ""}})
    patient["demo"] = True
    store.put("patients", patient)
    meds = []
    for spec in manifest()["medications"]:
        med = service.create_medication(store, patient["id"], {
            "name": spec["name"], "strength": spec["strength"], "instructions": spec.get("instructions", ""),
            "schedule": spec["schedule"]})
        # Today's doses count from local midnight, so a visitor can simulate any
        # time of today; earlier days never existed for this patient.
        midnight = domain.parse_at(None).astimezone(domain.tz_of(patient)).replace(hour=0, minute=0, second=0, microsecond=0)
        med["created_at"] = domain.iso(midnight)
        store.put("medications", med)
        ref = {"id": new_id("ref"), "patient_id": patient["id"], "medication_id": med["id"], "confirmed": True,
               **template["references"][spec["key"]]}
        store.put("references", ref)
        read = ref.get("expiry_read") or {}
        expiry = ({"date": read["date"], "source": "photo", "confidence": read["confidence"]}
                  if read.get("date") and read.get("confidence") in ("high", "medium") else None)
        med = store.update("medications", med["id"], active_reference_id=ref["id"], demo_key=spec["key"], expiry=expiry)
        meds.append(med)
    return {"patient": patient, "medications": meds, "samples": samples()}


def samples() -> list[dict]:
    return manifest()["samples"]


def sample_bytes(sample_id: str) -> bytes:
    for s in manifest()["samples"]:
        if s["id"] == sample_id:
            return (ASSETS / s["file"]).read_bytes()
    raise service.NotFound(f"sample {sample_id} not found")


def asset_path(file_name: str) -> Path:
    names = {s["file"] for s in manifest()["samples"]} | {m["reference"] for m in manifest()["medications"]}
    if file_name not in names:
        raise service.NotFound("unknown demo asset")
    return ASSETS / file_name
