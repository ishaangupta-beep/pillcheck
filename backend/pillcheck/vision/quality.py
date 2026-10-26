"""Capture-quality gates run before any expensive analysis.

Each failed gate carries a plain-language retake instruction the agent can
pass straight to an elderly user.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from ..config import settings


@dataclass
class QualityReport:
    ok: bool
    blur_score: float
    brightness: float
    glare_fraction: float
    issues: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "blur_score": round(self.blur_score, 1),
            "brightness": round(self.brightness, 1),
            "glare_fraction": round(self.glare_fraction, 3),
            "issues": self.issues,
            "guidance": self.guidance,
        }


def glare_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Specular highlights: near-saturated, low-saturation pixels, cleaned up."""
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    mask = ((v >= settings.vision.glare_pixel_value) & (s < 40)).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def assess(image_bgr: np.ndarray) -> QualityReport:
    cfg = settings.vision
    h, w = image_bgr.shape[:2]
    issues: list[str] = []
    guidance: list[str] = []

    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    # Measure sharpness at a fixed scale so the score is resolution independent.
    scale = 800.0 / max(h, w)
    small = cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) if scale < 1 else gray
    blur = float(cv.Laplacian(small, cv.CV_64F).var())
    # Judge exposure on the subject, not the frame: a strip photographed on a
    # black cloth or white paper is fine as long as the bright part (the foil)
    # is neither murky nor blown out.
    brightness = float(np.percentile(gray, 90))
    blown = float((gray >= 250).mean())
    glare = float((glare_mask(image_bgr) > 0).mean())

    if max(h, w) < cfg.min_long_side:
        issues.append("low_resolution")
        guidance.append("The photo is too small. Please take it with the main camera.")
    if blur < cfg.blur_min_laplacian_var:
        issues.append("blurry")
        guidance.append("The photo is blurry. Hold the phone still and tap the strip to focus.")
    if brightness < cfg.dark_max_mean:
        issues.append("too_dark")
        guidance.append("It is too dark. Turn on a light or move near a window.")
    elif blown > 0.5:
        issues.append("overexposed")
        guidance.append("It is too bright. Move away from direct light.")
    if glare > cfg.glare_max_fraction:
        issues.append("glare")
        guidance.append("There is a strong reflection on the foil. Tilt the strip slightly or turn off the flash.")

    return QualityReport(not issues, blur, brightness, glare, issues, guidance)
