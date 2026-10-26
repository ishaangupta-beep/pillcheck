"""End-to-end perception for one photo.

register_reference(): a caregiver photographs a full strip once. We keep the
    image at working resolution, its ALIKED features, and proposed pockets.

analyze_check(): every later photo goes through
    decode -> quality gates -> duplicate check -> identity (which registered
    strip is this?) -> MAGSAC registration -> warp -> per-pocket classification
    -> comparison with the last known state -> evidence image.
The returned dict is the perception record the agent reasons over.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from ..config import settings
from . import annotate, dedupe, expiry, quality
from .align import register, resize_long_side, warp_to_reference
from .classify import classify_pockets, verify_appearance
from .models import Features, get_engine
from .pockets import Pocket, detect_pockets, detect_strip_polygon

HARD_QUALITY_ISSUES = {"low_resolution", "blurry", "too_dark", "overexposed"}


@dataclass
class ReferenceBundle:
    medication_id: str
    image: np.ndarray
    features: Features
    pockets: list[Pocket]
    strip_polygon: np.ndarray | None = None


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    # IMREAD_COLOR honours EXIF orientation, so portrait phone photos come out upright.
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the image. Please upload a JPEG or PNG photo.")
    return img


def register_reference(image_bytes: bytes) -> dict:
    t0 = time.perf_counter()
    raw = decode_image(image_bytes)
    img, _ = resize_long_side(raw, settings.vision.work_long_side)
    q = quality.assess(img)
    feats = get_engine().extract(img)
    pockets, conf = detect_pockets(img)
    strip = detect_strip_polygon(img, pockets)
    try:
        exp = expiry.read_expiry(raw)
    except Exception as e:  # reading the date must never block strip setup
        exp = {"date": None, "confidence": "none", "raw": "", "source": "photo", "note": str(e)[:120]}
    return {
        "expiry": exp,
        "strip_polygon": strip,
        "image": img,
        "features": feats,
        "pockets": pockets,
        "pocket_confidence": conf,
        "quality": q.to_dict(),
        "feature_count": int(len(feats.keypoints)),
        "engine": get_engine().name,
        "preview_jpeg": annotate.encode_jpeg(annotate.reference_preview(img, pockets, strip)),
        "elapsed_ms": round(1000 * (time.perf_counter() - t0)),
    }


def analyze_check(image_bytes: bytes, references: list[ReferenceBundle], expected_medication_id: str | None,
                  previous_states: dict[str, dict[int, str]], previous_fingerprints: list[dict]) -> dict:
    """previous_states: medication_id -> {pocket_id: "intact"|"popped"} from the last trusted check."""
    timings: dict[str, int] = {}
    t = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        timings[name] = round(1000 * (now - t))
        t = now

    raw = decode_image(image_bytes)
    img, _ = resize_long_side(raw, settings.vision.work_long_side)
    result: dict = {"status": "ok", "guidance": [], "engine": get_engine().name}
    q = quality.assess(img)
    result["quality"] = q.to_dict()
    fp = dedupe.fingerprint(image_bytes, raw)
    result["fingerprint"] = fp
    lap("decode_quality")

    dup = dedupe.find_duplicate(fp, previous_fingerprints)
    result["duplicate"] = dup
    if dup:
        result["status"] = "duplicate"
        result["guidance"].append("This looks like a photo that was already sent. Please take a new photo of the strip now.")
        result["timings_ms"] = timings
        return result

    hard = [i for i in q.issues if i in HARD_QUALITY_ISSUES]
    if hard:
        result["status"] = "retake"
        result["guidance"] += q.guidance
        result["timings_ms"] = timings
        return result

    engine = get_engine()
    cur_feats = engine.extract(img)
    lap("features")

    # Identity: register against every registered strip, then verify the
    # alignment by comparing the whole printed surface (see verify_appearance).
    cfg = settings.vision
    tried = []
    best = None  # (appearance, ref, reg, warped, valid)
    ordered = sorted(references, key=lambda r: r.medication_id != expected_medication_id)
    for ref in ordered:
        reg = register(img, ref.features, cur_feats)
        entry = {"medication_id": ref.medication_id, "inliers": reg.inliers, "aligned": reg.ok, "appearance": None}
        tried.append(entry)
        if reg.H is None or "degenerate_geometry" in reg.issues:
            continue
        warped, valid = warp_to_reference(img, reg.H, ref.features.image_size)
        app = verify_appearance(ref.image, warped, valid, ref.pockets, ref.strip_polygon)
        entry["appearance"] = None if app is None else round(app, 3)
        if app is None:
            # Too little printed foil to compare: accept only a strong geometric
            # match (unrelated strips give ~10 inliers, the same strip hundreds).
            strong = reg.ok and reg.inliers >= cfg.identity_min_inliers_geometry
            score = cfg.identity_min_appearance if strong else 0.0
            entry["verified_by"] = "geometry"
        else:
            # Print agrees: accept. Print only roughly agrees (shared layout,
            # similar foil): also require the strong geometric match.
            ok_geo = reg.inliers >= cfg.identity_min_inliers_geometry
            score = app if (app >= cfg.identity_min_appearance + 0.15 or ok_geo) else min(app, cfg.identity_min_appearance - 0.01)
            entry["verified_by"] = "appearance"
        if best is None or (score, reg.inliers) > (best[0], best[2].inliers):
            best = (score, ref, reg, warped, valid)
        # A confident match on the expected strip ends the search early.
        if ref.medication_id == expected_medication_id and reg.ok and (
                score >= cfg.identity_min_appearance + 0.08 or (app is None and score >= cfg.identity_min_appearance)):
            break
    lap("identify_register")

    identified = best is not None and best[0] >= cfg.identity_min_appearance and best[2].inliers >= cfg.min_inliers // 2
    ref, reg = (best[1], best[2]) if best else (None, None)
    result["identity"] = {
        "expected_medication_id": expected_medication_id,
        "matched_medication_id": ref.medication_id if identified else None,
        "appearance": round(best[0], 3) if best else None,
        "candidates": tried,
    }
    result["registration"] = reg.to_dict() if reg else None

    if not identified:
        result["status"] = "unrecognised"
        guidance = reg.guidance if (reg and reg.guidance) else []
        result["guidance"] += guidance or [
            "I could not recognise this strip. Make sure it is one of the registered medicines and the whole strip is in the photo."]
        result["timings_ms"] = timings
        return result
    if expected_medication_id and ref.medication_id != expected_medication_id:
        result["status"] = "wrong_medication"
    if not reg.ok:
        result["status"] = "retake"
        result["guidance"] += reg.guidance
        result["timings_ms"] = timings
        return result

    warped, valid = best[3], best[4]
    readings, stats = classify_pockets(ref.image, warped, valid, ref.pockets, ref.strip_polygon)
    lap("classify")

    prev = previous_states.get(ref.medication_id, {})
    newly, inconsistent = [], []
    for r in readings:
        before = prev.get(r.id, "popped" if any(p.id == r.id and p.initially_empty for p in ref.pockets) else "intact")
        if r.state == "popped" and before == "intact":
            newly.append(r.id)
        if r.state == "intact" and before == "popped":
            inconsistent.append(r.id)

    counts = {s: sum(1 for r in readings if r.state == s) for s in ("intact", "popped", "uncertain", "unknown")}
    counts["total"] = len(readings)
    result.update({
        "medication_id": ref.medication_id,
        "pockets": [r.to_dict() for r in readings],
        "counts": counts,
        "newly_popped": newly,
        "inconsistent": inconsistent,
        "classifier_stats": stats,
    })
    if counts["uncertain"] or counts["unknown"]:
        result["guidance"].append("Some pockets were hard to read. A clearer photo, without reflections, will help.")

    result["_evidence_jpeg"] = annotate.encode_jpeg(
        annotate.side_by_side(ref.image, warped, ref.pockets, readings, set(newly)))
    lap("annotate")
    result["timings_ms"] = timings
    return result
