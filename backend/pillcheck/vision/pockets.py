"""Find the blister pockets on a reference photo of a full strip.

Two independent detectors propose pockets:
  * contour detector: CLAHE -> Canny -> morphological closing -> contours ->
    ellipse fit, kept when the shape is convex and ellipse-like;
  * Hough circle detector (cv.HOUGH_GRADIENT_ALT) for round pockets.
Proposals are merged with non-maximum suppression, filtered to a consistent
size cluster (all pockets on one strip are the same size), and ordered in
reading order. The caregiver confirms or adjusts them once in the app; after
that every daily check reuses these exact regions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2 as cv
import numpy as np


@dataclass
class Pocket:
    id: int
    cx: float
    cy: float
    rx: float  # semi-axis along the ellipse's x after rotation
    ry: float
    angle: float = 0.0  # degrees
    initially_empty: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("cx", "cy", "rx", "ry", "angle"):
            d[k] = round(float(d[k]), 2)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Pocket":
        return Pocket(int(d["id"]), float(d["cx"]), float(d["cy"]), float(d["rx"]), float(d["ry"]),
                      float(d.get("angle", 0.0)), bool(d.get("initially_empty", False)))

    def mask(self, shape: tuple[int, int], shrink: float = 1.0) -> np.ndarray:
        m = np.zeros(shape, np.uint8)
        cv.ellipse(m, (round(self.cx), round(self.cy)),
                   (max(1, round(self.rx * shrink)), max(1, round(self.ry * shrink))),
                   self.angle, 0, 360, 255, -1)
        return m

    def bbox(self, shape: tuple[int, int], pad: float = 1.15) -> tuple[int, int, int, int]:
        r = max(self.rx, self.ry) * pad
        h, w = shape
        x0, y0 = max(0, int(self.cx - r)), max(0, int(self.cy - r))
        x1, y1 = min(w, int(self.cx + r) + 1), min(h, int(self.cy + r) + 1)
        return x0, y0, x1, y1


def _contour_candidates(gray: np.ndarray, min_r: float, max_r: float) -> list[tuple]:
    out = []
    v = float(np.median(gray))
    for lo_f, hi_f in ((0.5, 1.2), (0.3, 0.9)):
        edges = cv.Canny(gray, int(max(10, lo_f * v)), int(min(255, hi_f * v + 40)))
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=2)
        contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        for c in contours:
            if len(c) < 20:
                continue
            area = cv.contourArea(c)
            if area < np.pi * min_r * min_r or area > np.pi * max_r * max_r:
                continue
            hull_area = cv.contourArea(cv.convexHull(c))
            if hull_area <= 0 or area / hull_area < 0.85:
                continue
            (cx, cy), (ew, eh), ang = cv.fitEllipse(c)
            # Normalise so (ew, eh) are the extents along x and y.
            if 45 < ang % 180 < 135:
                ew, eh, ang = eh, ew, ang - 90
            ell_area = np.pi * ew * eh / 4
            if ell_area <= 0 or not (0.75 < area / ell_area < 1.15):
                continue
            if max(ew, eh) / max(1e-3, min(ew, eh)) > 3.0:
                continue
            out.append((cx, cy, ew / 2, eh / 2, ang, area / hull_area))
    return out


def _hough_candidates(gray: np.ndarray, min_r: float, max_r: float) -> list[tuple]:
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=1.6 * min_r,
                              param1=200, param2=0.8, minRadius=int(min_r), maxRadius=int(max_r))
    if circles is None:
        return []
    return [(x, y, r, r, 0.0, 0.8) for x, y, r in circles[0]]


def _nms(cands: list[tuple]) -> list[tuple]:
    cands = sorted(cands, key=lambda c: -c[5])
    kept: list[tuple] = []
    for c in cands:
        rc = max(c[2], c[3])
        if all(np.hypot(c[0] - k[0], c[1] - k[1]) > 0.7 * max(rc, max(k[2], k[3])) for k in kept):
            kept.append(c)
    return kept


def _size_cluster(cands: list[tuple]) -> list[tuple]:
    if len(cands) < 2:
        return cands
    areas = np.array([c[2] * c[3] for c in cands])
    best: list[int] = []
    for a in areas:  # pick the size that the most candidates agree with
        idx = [i for i, b in enumerate(areas) if 0.6 * a <= b <= 1.6 * a]
        if len(idx) > len(best):
            best = idx
    return [cands[i] for i in best]


def _reading_order(cands: list[tuple]) -> list[tuple]:
    if not cands:
        return cands
    # Row tolerance is the vertical semi-axis: rows of wide pockets sit close together.
    r = 0.9 * float(np.median([c[3] for c in cands]))
    rows: list[list[tuple]] = []
    for c in sorted(cands, key=lambda c: c[1]):
        if rows and abs(c[1] - np.mean([x[1] for x in rows[-1]])) < r:
            rows[-1].append(c)
        else:
            rows.append([c])
    return [c for row in rows for c in sorted(row, key=lambda c: c[0])]


def detect_pockets(reference_bgr: np.ndarray) -> tuple[list[Pocket], float]:
    """Return proposed pockets and a 0..1 confidence in the proposal."""
    gray = cv.cvtColor(reference_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    short = min(gray.shape)
    min_r, max_r = 0.04 * short, 0.30 * short

    cands = _contour_candidates(gray, min_r, max_r) + _hough_candidates(gray, min_r, max_r)
    cands = _reading_order(_size_cluster(_nms(cands)))
    # Pockets on one strip are identical, so snap every ellipse to the median
    # size; individual fits disagree on whether they caught the inner or outer rim.
    if cands:
        axes = np.array([sorted((c[2], c[3])) for c in cands])
        r_min, r_max = np.median(axes[:, 0]), np.median(axes[:, 1])
        wide = np.median([1.0 if c[2] >= c[3] else 0.0 for c in cands]) >= 0.5
        rx, ry = (r_max, r_min) if wide else (r_min, r_max)
        cands = [(c[0], c[1], rx, ry, 0.0, c[5]) for c in cands]
    pockets = [Pocket(i, float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4])) for i, c in enumerate(cands)]

    if len(pockets) < 2:
        return pockets, 0.0
    # Confidence: consistent sizes and a regular, filled grid both look like a real strip.
    radii = np.array([max(p.rx, p.ry) for p in pockets])
    size_cv = float(radii.std() / radii.mean())
    tol = 0.9 * float(np.median([p.ry for p in pockets]))
    rows: list[list[Pocket]] = []
    for p in pockets:  # already in reading order
        if rows and abs(p.cy - np.mean([q.cy for q in rows[-1]])) < tol:
            rows[-1].append(p)
        else:
            rows.append([p])
    grid_fill = len(pockets) / (len(rows) * max(len(r) for r in rows))
    conf = max(0.0, 1.0 - 2.5 * size_cv) * grid_fill
    return pockets, round(float(conf), 3)


def detect_strip_polygon(reference_bgr: np.ndarray, pockets: list[Pocket]) -> np.ndarray:
    """Outline of the strip in the reference photo, via GrabCut seeded by the pockets.

    The pockets are certain foreground, the image border certain background.
    The result's convex hull is the strip: everything the lighting correction
    and noise estimate may use, and nothing from the table it was lying on.
    """
    h, w = reference_bgr.shape[:2]
    if not pockets:
        return np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    rx = float(np.median([p.rx for p in pockets]))
    ry = float(np.median([p.ry for p in pockets]))
    xs = [p.cx for p in pockets]
    ys = [p.cy for p in pockets]
    x0, x1 = max(1, int(min(xs) - 3.0 * rx)), min(w - 2, int(max(xs) + 3.0 * rx))
    y0, y1 = max(1, int(min(ys) - 4.0 * ry)), min(h - 2, int(max(ys) + 4.0 * ry))

    scale = 640.0 / max(h, w)
    small = cv.resize(reference_bgr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    sh, sw = small.shape[:2]
    mask = np.full((sh, sw), cv.GC_BGD, np.uint8)
    mask[int(y0 * scale):int(y1 * scale), int(x0 * scale):int(x1 * scale)] = cv.GC_PR_FGD
    for p in pockets:
        cv.ellipse(mask, (round(p.cx * scale), round(p.cy * scale)),
                   (max(1, round(p.rx * scale)), max(1, round(p.ry * scale))), p.angle, 0, 360, cv.GC_FGD, -1)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv.grabCut(small, mask, None, bgd, fgd, 4, cv.GC_INIT_WITH_MASK)
    except cv.error:
        pass
    fg = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv.connectedComponentsWithStats(fg)
    # Keep the component that contains the most pocket centres.
    votes = np.zeros(n, int)
    for p in pockets:
        lx, ly = min(sw - 1, round(p.cx * scale)), min(sh - 1, round(p.cy * scale))
        votes[labels[ly, lx]] += 1
    votes[0] = -1
    comp = (labels == int(np.argmax(votes))).astype(np.uint8)
    contours, _ = cv.findContours(comp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    hull = cv.convexHull(max(contours, key=cv.contourArea)).reshape(-1, 2).astype(np.float32) / scale
    # Never let the strip exceed the generous box around the pockets.
    hull[:, 0] = np.clip(hull[:, 0], x0, x1)
    hull[:, 1] = np.clip(hull[:, 1], y0, y1)
    return hull


def polygon_mask(shape: tuple[int, int], polygon: np.ndarray, erode_px: int = 4) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv.fillPoly(m, [np.int32(np.round(polygon))], 255)
    if erode_px:
        m = cv.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    return m
