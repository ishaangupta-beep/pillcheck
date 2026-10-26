"""Catch a photo being re-submitted instead of a fresh one.

Existing "photo proof" medication apps accept any photo. PillCheck compares
each upload with earlier ones using an exact SHA-256 and OpenCV's perceptual
hash (cv.img_hash.PHash, from opencv-contrib). Two genuine hand-held photos
are never pixel-identical; a hash distance of a few bits means the same image
(possibly re-compressed or resized) was uploaded again.
"""
from __future__ import annotations

import hashlib

import cv2 as cv
import numpy as np

_phash = cv.img_hash.PHash.create()
NEAR_DUPLICATE_BITS = 4


def fingerprint(image_bytes: bytes, image_bgr: np.ndarray) -> dict:
    h = _phash.compute(image_bgr)
    return {"sha256": hashlib.sha256(image_bytes).hexdigest(), "phash": h.tobytes().hex()}


def hamming(a_hex: str, b_hex: str) -> int:
    a = np.frombuffer(bytes.fromhex(a_hex), np.uint8)
    b = np.frombuffer(bytes.fromhex(b_hex), np.uint8)
    return int(np.unpackbits(a ^ b).sum())


def find_duplicate(fp: dict, previous: list[dict]) -> dict | None:
    """previous: [{"check_id", "sha256", "phash"}]. Returns the closest duplicate, if any."""
    best = None
    for prev in previous:
        if not prev.get("phash"):
            continue
        if prev.get("sha256") == fp["sha256"]:
            return {"check_id": prev["check_id"], "distance": 0, "exact": True}
        d = hamming(fp["phash"], prev["phash"])
        if d <= NEAR_DUPLICATE_BITS and (best is None or d < best["distance"]):
            best = {"check_id": prev["check_id"], "distance": d, "exact": False}
    return best
