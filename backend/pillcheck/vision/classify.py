"""Decide, for every pocket, whether it still holds its pill.

The photo has already been warped into the reference frame, so each pocket is
compared with its own appearance on the day the strip was registered.

1. Photometric normalisation: a per-channel gain/offset is fitted (trimmed
   least squares) on the foil around the pockets, which cancels lighting and
   white-balance changes between the two photos.
2. Per pocket, inside a slightly shrunk ellipse:
   * colour change: mean CIE Lab distance, chroma weighted above lightness;
   * structure change: 1 - SSIM on the luminance;
   both evaluated over small shifts (+-3 px), keeping the best, so residual
   registration error does not read as change.
3. The two terms are divided by fixed units (calibrated on the evaluation
   set) and averaged into one change score. The photo's own noise floor,
   measured on foil patches that cannot have changed, widens those units for
   noisy photos. Thresholds split the score into intact / popped / uncertain.
4. Pockets hidden by glare or lying outside the photo are marked unknown
   rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from ..config import settings
from .pockets import Pocket, polygon_mask
from .quality import glare_mask

SHIFTS = [(dx, dy) for dx in (-3, 0, 3) for dy in (-3, 0, 3)]


@dataclass
class PocketReading:
    id: int
    state: str  # intact | popped | uncertain | unknown
    score: float  # normalised change score
    colour_change: float
    structure_change: float
    confidence: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "score": round(self.score, 2),
            "colour_change": round(self.colour_change, 2),
            "structure_change": round(self.structure_change, 3),
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
        }


def _foil_mask(shape: tuple[int, int], pockets: list[Pocket], valid: np.ndarray,
               strip_polygon: np.ndarray | None = None) -> np.ndarray:
    """Printed foil: inside the strip outline, outside every pocket, inside the photo."""
    if strip_polygon is not None and len(strip_polygon) >= 3:
        m = polygon_mask(shape, strip_polygon)
    else:  # fallback: a tight box around the pockets
        m = np.zeros(shape, np.uint8)
        xs0 = min(p.cx - 1.3 * p.rx for p in pockets)
        xs1 = max(p.cx + 1.3 * p.rx for p in pockets)
        ys0 = min(p.cy - 1.3 * p.ry for p in pockets)
        ys1 = max(p.cy + 1.3 * p.ry for p in pockets)
        m[max(0, int(ys0)):int(ys1), max(0, int(xs0)):int(xs1)] = 255
    for p in pockets:
        m[p.mask(shape, shrink=1.35) > 0] = 0
    return cv.bitwise_and(m, valid)


def _fit_photometric(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fit dst ~ a*src + b per channel on masked pixels (trimmed LSQ); return corrected src."""
    idx = mask > 0
    out = src.astype(np.float32).copy()
    if idx.sum() < 500:
        return out
    for ch in range(3):
        x = src[..., ch][idx].astype(np.float32)
        y = dst[..., ch][idx].astype(np.float32)
        keep = np.ones_like(x, bool)
        a, b = 1.0, 0.0
        for _ in range(3):
            A = np.stack([x[keep], np.ones(keep.sum(), np.float32)], 1)
            sol, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
            a, b = float(sol[0]), float(sol[1])
            res = np.abs(a * x + b - y)
            keep = res <= np.percentile(res, 80)
        a = float(np.clip(a, 0.4, 2.5))
        out[..., ch] = a * out[..., ch] + b
    return np.clip(out, 0, 255)


def _ssim(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (7, 7)
    mu_a, mu_b = cv.GaussianBlur(a, k, 1.5), cv.GaussianBlur(b, k, 1.5)
    s_a = cv.GaussianBlur(a * a, k, 1.5) - mu_a ** 2
    s_b = cv.GaussianBlur(b * b, k, 1.5) - mu_b ** 2
    s_ab = cv.GaussianBlur(a * b, k, 1.5) - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_a + s_b + C2))
    return float(ssim[m > 0].mean())


def _lab_dist(a_lab: np.ndarray, b_lab: np.ndarray, m: np.ndarray) -> float:
    d = a_lab.astype(np.float32) - b_lab.astype(np.float32)
    d[..., 0] *= 0.5  # lightness is the channel most affected by shading
    return float(np.sqrt((d ** 2).sum(-1))[m > 0].mean())


def _patch_metrics(ref_lab, cur_lab, ref_l, cur_l, box, mask) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    best_c, best_s = np.inf, np.inf
    rl, rL = ref_lab[y0:y1, x0:x1], ref_l[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    H, W = cur_l.shape
    for dx, dy in SHIFTS:
        if x0 + dx < 0 or y0 + dy < 0 or x1 + dx > W or y1 + dy > H:
            continue
        cl = cur_lab[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        cL = cur_l[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        c = _lab_dist(rl, cl, m)
        s = 1.0 - _ssim(rL, cL, m)
        # Pick the shift that best explains both terms together.
        if c / 10 + s < best_c / 10 + best_s:
            best_c, best_s = c, s
    return best_c, best_s


def _noise_floor(ref_lab, cur_lab, ref_l, cur_l, foil: np.ndarray, size: tuple[int, int], rng) -> tuple[float, float]:
    ys, xs = np.nonzero(foil)
    if len(xs) < 50:
        return 4.0, 0.05
    hw, hh = size
    cs, ss = [], []
    for _ in range(24):
        i = rng.integers(len(xs))
        cx, cy = xs[i], ys[i]
        x0, y0, x1, y1 = cx - hw, cy - hh, cx + hw, cy + hh
        if x0 < 3 or y0 < 3 or x1 > foil.shape[1] - 3 or y1 > foil.shape[0] - 3:
            continue
        m = foil[y0:y1, x0:x1]
        if (m > 0).mean() < 0.9:
            continue
        c, s = _patch_metrics(ref_lab, cur_lab, ref_l, cur_l, (x0, y0, x1, y1), foil)
        cs.append(c)
        ss.append(s)
    if len(cs) < 4:
        return 4.0, 0.05
    # 75th percentile: robust to a few patches that still catch print edges.
    return max(2.5, float(np.percentile(cs, 75))), max(0.03, float(np.percentile(ss, 75)))


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    return cv.magnitude(gx, gy)


def verify_appearance(ref_bgr: np.ndarray, warped_bgr: np.ndarray, valid: np.ndarray, pockets: list[Pocket],
                      strip_polygon: np.ndarray | None = None) -> float | None:
    """Does the aligned photo show the *same* printed strip as the reference?

    Feature matching can lock onto print that different medicines share
    (manufacturer layout, "Rx only", batch lines), and two products from one
    maker can differ only in the name. So the printed foil is cut into tiles,
    gradient correlation is measured per textured tile, and the score is a low
    percentile: every printed region, the name included, has to agree.

    Returns None when the strip shows too little foil between its pockets to
    compare (common on real strips photographed from the blister side); the
    caller then relies on the geometric match alone.
    """
    foil = _foil_mask(ref_bgr.shape[:2], pockets, valid, strip_polygon)
    if (foil > 0).sum() < 500:
        return None
    cur = _fit_photometric(warped_bgr, ref_bgr, foil).astype(np.uint8)
    ga = _grad_mag(cv.GaussianBlur(cv.cvtColor(ref_bgr, cv.COLOR_BGR2GRAY), (3, 3), 0))
    gb = _grad_mag(cv.GaussianBlur(cv.cvtColor(cur, cv.COLOR_BGR2GRAY), (3, 3), 0))
    r = int(np.median([max(p.rx, p.ry) for p in pockets])) if pockets else 40
    t = int(np.clip(r * 0.9, 20, 96))
    energy, scores = [], []
    ys, xs = np.nonzero(foil)
    for y0 in range(ys.min(), ys.max() - t + 1, t // 2):
        for x0 in range(xs.min(), xs.max() - t + 1, t // 2):
            m = foil[y0:y0 + t, x0:x0 + t] > 0
            if m.mean() < 0.8:
                continue
            a, b = ga[y0:y0 + t, x0:x0 + t][m], gb[y0:y0 + t, x0:x0 + t][m]
            energy.append(a.mean())
            sa, sb = a.std(), b.std()
            scores.append(float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb + 1e-6)))
    if len(scores) < 4:
        return None
    energy, scores = np.array(energy), np.array(scores)
    # Only tiles with real print in the reference carry identity information.
    textured = scores[energy >= np.percentile(energy, 60)]
    return float(np.percentile(textured, 10))


def new_glare(ref_bgr: np.ndarray, warped_bgr: np.ndarray) -> np.ndarray:
    """Glare present today but not in the reference (white pills are not glare)."""
    ref_bright = cv.dilate(glare_mask(ref_bgr), np.ones((7, 7), np.uint8))
    cur = glare_mask(warped_bgr)
    hsv_ref = cv.cvtColor(ref_bgr, cv.COLOR_BGR2HSV)
    ref_bright |= ((hsv_ref[..., 2] > 215) & (hsv_ref[..., 1] < 50)).astype(np.uint8) * 255
    return cv.bitwise_and(cur, cv.bitwise_not(ref_bright))


def classify_pockets(ref_bgr: np.ndarray, warped_bgr: np.ndarray, valid: np.ndarray,
                     pockets: list[Pocket], strip_polygon: np.ndarray | None = None, seed: int = 0) -> tuple[list[PocketReading], dict]:
    cfg = settings.vision
    # Compare pockets at a fixed size: a 7x7 SSIM window should see the shape
    # of the foil (a dome vs. a crumple), not the micro-texture of a big pocket.
    r_med = float(np.median([max(p.rx, p.ry) for p in pockets])) if pockets else 0.0
    f = min(1.0, cfg.pocket_compare_px / (2 * r_med)) if r_med > 0 else 1.0
    if f < 0.95:
        size = (max(8, round(ref_bgr.shape[1] * f)), max(8, round(ref_bgr.shape[0] * f)))
        ref_bgr = cv.resize(ref_bgr, size, interpolation=cv.INTER_AREA)
        warped_bgr = cv.resize(warped_bgr, size, interpolation=cv.INTER_AREA)
        valid = cv.resize(valid, size, interpolation=cv.INTER_NEAREST)
        pockets = [Pocket(p.id, p.cx * f, p.cy * f, p.rx * f, p.ry * f, p.angle, p.initially_empty) for p in pockets]
        if strip_polygon is not None:
            strip_polygon = (np.asarray(strip_polygon, np.float32) * f).astype(np.int32)
    shape = ref_bgr.shape[:2]
    glare = new_glare(ref_bgr, warped_bgr)
    # Glare is local and handled per pocket; keep it out of the global
    # lighting fit and the noise estimate, or one reflection inflates both.
    foil = _foil_mask(shape, pockets, valid, strip_polygon)
    foil = cv.bitwise_and(foil, cv.bitwise_not(cv.dilate(glare, np.ones((15, 15), np.uint8))))
    cur = _fit_photometric(warped_bgr, ref_bgr, foil).astype(np.uint8)

    ref_lab = cv.cvtColor(ref_bgr, cv.COLOR_BGR2Lab)
    cur_lab = cv.cvtColor(cur, cv.COLOR_BGR2Lab)
    ref_l, cur_l = ref_lab[..., 0], cur_lab[..., 0]

    r = int(np.median([max(p.rx, p.ry) for p in pockets])) if pockets else 20
    floor_c, floor_s = _noise_floor(ref_lab, cur_lab, ref_l, cur_l, foil, (max(6, r // 2), max(6, r // 2)),
                                    np.random.default_rng(seed))

    # Fixed units calibrated on the evaluation set; a noisy photo (high floor)
    # widens them so camera noise cannot masquerade as a popped pocket.
    unit_c = max(cfg.colour_unit, 1.5 * floor_c)
    unit_s = max(cfg.structure_unit, 1.5 * floor_s)

    readings: list[PocketReading] = []
    for p in pockets:
        m = p.mask(shape, shrink=0.85)
        area = (m > 0).sum()
        inside = (cv.bitwise_and(m, valid) > 0).sum() / max(1, area)
        if inside < 0.9:
            readings.append(PocketReading(p.id, "unknown", 0, 0, 0, 0, "pocket is outside the photo"))
            continue
        glare_frac = (cv.bitwise_and(m, glare) > 0).sum() / max(1, area)
        box = p.bbox(shape, pad=1.0)
        c, s = _patch_metrics(ref_lab, cur_lab, ref_l, cur_l, box, m)
        score = 0.5 * (c / unit_c) + 0.5 * (s / unit_s)

        if glare_frac > cfg.pocket_glare_max_fraction:
            readings.append(PocketReading(p.id, "unknown", score, c, s, 0, f"glare covers {glare_frac:.0%} of the pocket"))
            continue
        if score >= cfg.popped_threshold:
            changed = True
            conf = min(1.0, 0.6 + 0.4 * (score - cfg.popped_threshold) / cfg.popped_threshold)
        elif score <= cfg.intact_threshold:
            changed = False
            conf = min(1.0, 0.6 + 0.4 * (cfg.intact_threshold - score) / cfg.intact_threshold)
        else:
            readings.append(PocketReading(p.id, "uncertain", score, c, s, 0.3, "change score in the uncertain band"))
            continue
        if p.initially_empty:
            state = "popped" if not changed else "uncertain"
            reason = "empty since registration" if not changed else "registered empty but now looks different"
        else:
            state = "popped" if changed else "intact"
            reason = ""
        readings.append(PocketReading(p.id, state, score, c, s, round(conf, 2), reason))

    stats = {"noise_floor_colour": round(floor_c, 2), "noise_floor_structure": round(floor_s, 3),
             "unit_colour": round(unit_c, 2), "unit_structure": round(unit_s, 3),
             "foil_pixels": int((foil > 0).sum())}
    return readings, stats
