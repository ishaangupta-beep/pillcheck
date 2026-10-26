"""Evidence images: what the system saw, in a form a caregiver can trust."""
from __future__ import annotations

import cv2 as cv
import numpy as np

from .classify import PocketReading
from .pockets import Pocket

COLOURS = {  # BGR
    "intact": (90, 180, 60),
    "popped": (60, 60, 220),
    "newly_popped": (0, 140, 255),
    "uncertain": (0, 200, 230),
    "unknown": (160, 160, 160),
}


def draw_pockets(image: np.ndarray, pockets: list[Pocket], readings: list[PocketReading] | None = None,
                 newly_popped: set[int] = frozenset()) -> np.ndarray:
    out = image.copy()
    by_id = {r.id: r for r in readings} if readings else {}
    overlay = out.copy()
    for p in pockets:
        r = by_id.get(p.id)
        state = r.state if r else "intact"
        key = "newly_popped" if p.id in newly_popped else state
        col = COLOURS[key]
        centre, axes = (round(p.cx), round(p.cy)), (round(p.rx), round(p.ry))
        cv.ellipse(overlay, centre, axes, p.angle, 0, 360, col, -1, cv.LINE_AA)
        cv.ellipse(out, centre, axes, p.angle, 0, 360, col, 3, cv.LINE_AA)
    out = cv.addWeighted(overlay, 0.22, out, 0.78, 0)
    scale = max(0.5, min(1.2, np.median([p.rx for p in pockets]) / 40)) if pockets else 0.8
    for p in pockets:
        r = by_id.get(p.id)
        label = str(p.id + 1)
        if r and r.state in ("uncertain", "unknown"):
            label += "?"
        org = (round(p.cx - 10 * scale * len(label)), round(p.cy + 10 * scale))
        cv.putText(out, label, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv.LINE_AA)
        cv.putText(out, label, org, cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv.LINE_AA)
    return out


def reference_preview(image: np.ndarray, pockets: list[Pocket], strip_polygon: np.ndarray | None) -> np.ndarray:
    out = draw_pockets(image, pockets)
    if strip_polygon is not None and len(strip_polygon) >= 3:
        cv.polylines(out, [np.int32(np.round(strip_polygon))], True, (255, 160, 40), 3, cv.LINE_AA)
    return out


def strip_crop(shape: tuple[int, int], pockets: list[Pocket]) -> tuple[int, int, int, int]:
    """Region around the pockets generous enough to include the strip's print."""
    if not pockets:
        return 0, 0, shape[1], shape[0]
    r = float(np.median([max(p.rx, p.ry) for p in pockets]))
    xs = [p.cx for p in pockets]
    ys = [p.cy for p in pockets]
    x0, x1 = int(min(xs) - 2.0 * r), int(max(xs) + 2.0 * r)
    y0, y1 = int(min(ys) - 2.6 * r), int(max(ys) + 2.6 * r)
    return max(0, x0), max(0, y0), min(shape[1], x1), min(shape[0], y1)


def side_by_side(reference: np.ndarray, aligned: np.ndarray, pockets: list[Pocket],
                 readings: list[PocketReading], newly_popped: set[int]) -> np.ndarray:
    x0, y0, x1, y1 = strip_crop(reference.shape[:2], pockets)
    left = draw_pockets(reference, pockets)[y0:y1, x0:x1]
    right = draw_pockets(aligned, pockets, readings, newly_popped)[y0:y1, x0:x1]
    h = left.shape[0]
    bar = np.full((h, 12, 3), 255, np.uint8)
    return np.hstack([left, bar, right])


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()
