"""Register a new photo onto the medication's reference photo.

Pipeline (all OpenCV 5):
  ALIKED keypoints/descriptors -> LightGlue matching -> MAGSAC++ homography
  (cv.findHomography with cv.USAC_MAGSAC) -> geometric sanity checks.

Registration is the backbone of PillCheck: once the photo is warped into the
reference frame, every pocket can be compared pixel-for-pixel with how it
looked when the strip was full. The same inlier count doubles as an identity
check: a photo of a different strip simply does not register.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from ..config import settings
from .models import Features, get_engine


@dataclass
class Registration:
    ok: bool
    engine: str
    matches: int
    inliers: int
    inlier_ratio: float
    reproj_rmse: float
    H: np.ndarray | None  # maps current-photo pixels -> reference pixels
    coverage: float = 0.0  # fraction of the photo the strip occupies
    issues: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """0..1 blend of inlier count, inlier ratio and residual."""
        if self.H is None:
            return 0.0
        c_count = min(1.0, self.inliers / 200.0)
        c_ratio = min(1.0, self.inlier_ratio / 0.6)
        c_rmse = max(0.0, 1.0 - self.reproj_rmse / (2 * settings.vision.max_reproj_rmse_px))
        return round(0.45 * c_count + 0.35 * c_ratio + 0.20 * c_rmse, 3)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "engine": self.engine,
            "matches": self.matches,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 3),
            "reproj_rmse_px": round(self.reproj_rmse, 2),
            "coverage": round(self.coverage, 3),
            "confidence": self.confidence,
            "issues": self.issues,
            "guidance": self.guidance,
        }


def resize_long_side(image: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    s = long_side / max(h, w)
    if abs(s - 1.0) < 1e-3:
        return image, 1.0
    interp = cv.INTER_AREA if s < 1 else cv.INTER_CUBIC
    return cv.resize(image, (round(w * s), round(h * s)), interpolation=interp), s


def _homography_is_sane(H: np.ndarray, ref_size: tuple[int, int]) -> bool:
    if H is None or not np.all(np.isfinite(H)):
        return False
    det = np.linalg.det(H[:2, :2])
    if det <= 0.02 or det > 50:  # mirrored, collapsed or exploded
        return False
    # Strong projective terms mean a degenerate fit for a flat strip.
    if np.abs(H[2, :2]).max() > 0.01:
        return False
    w, h = ref_size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        back = cv.perspectiveTransform(corners, np.linalg.inv(H)).reshape(-1, 2)
    except np.linalg.LinAlgError:
        return False
    return bool(cv.isContourConvex(back.astype(np.float32)))


def _framing_guidance(H: np.ndarray, ref_size: tuple[int, int], cur_size: tuple[int, int]) -> tuple[float, list[str], list[str]]:
    """Where does the strip sit in the user's photo? Turn that into instructions."""
    w, h = ref_size
    cw, ch = cur_size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv.perspectiveTransform(corners, np.linalg.inv(H)).reshape(-1, 2)
    area = abs(cv.contourArea(quad.astype(np.float32)))
    coverage = float(area / (cw * ch))
    issues, guidance = [], []
    margin = 0.02
    xs, ys = quad[:, 0] / cw, quad[:, 1] / ch
    cut = []
    if xs.min() < -margin:
        cut.append("left")
    if xs.max() > 1 + margin:
        cut.append("right")
    if ys.min() < -margin:
        cut.append("top")
    if ys.max() > 1 + margin:
        cut.append("bottom")
    if cut:
        issues.append("strip_cut_off")
        guidance.append(f"Part of the strip is outside the photo ({', '.join(cut)}). Move the phone back a little so the whole strip fits.")
    if coverage < 0.12:
        issues.append("too_far")
        guidance.append("The strip looks very small. Move the phone closer so the strip fills most of the photo.")
    return coverage, issues, guidance


def register(current_bgr: np.ndarray, ref_features: Features, cur_features: Features | None = None) -> Registration:
    """Register `current_bgr` (already at working resolution) onto the reference."""
    cfg = settings.vision
    engine = get_engine()
    cur = cur_features or engine.extract(current_bgr)
    pts_cur, pts_ref = engine.match(cur, ref_features)
    n = len(pts_cur)
    if n < 8:
        return Registration(False, engine.name, n, 0, 0.0, float("inf"), None,
                            issues=["no_match"],
                            guidance=["I could not recognise this strip. Make sure it is the right medicine and the printed side is visible."])

    H, mask = cv.findHomography(pts_cur, pts_ref, cv.USAC_MAGSAC, cfg.magsac_threshold_px,
                                maxIters=5000, confidence=0.999)
    if H is None or mask is None:
        return Registration(False, engine.name, n, 0, 0.0, float("inf"), None,
                            issues=["no_geometry"],
                            guidance=["I could not line this photo up with the strip. Please take it again from straight above."])

    inl = mask.ravel().astype(bool)
    inliers = int(inl.sum())
    proj = cv.perspectiveTransform(pts_cur[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((proj - pts_ref[inl]) ** 2, axis=1)))) if inliers else float("inf")
    ratio = inliers / n

    reg = Registration(True, engine.name, n, inliers, ratio, rmse, H)
    if not _homography_is_sane(H, ref_features.image_size):
        reg.ok = False
        reg.issues.append("degenerate_geometry")
        reg.guidance.append("The angle is too steep. Hold the phone flat above the strip.")
        return reg

    cov, issues, guidance = _framing_guidance(H, ref_features.image_size, cur.image_size)
    reg.coverage = cov
    reg.issues += issues
    reg.guidance += guidance
    if inliers < cfg.min_inliers or ratio < cfg.min_inlier_ratio or rmse > cfg.max_reproj_rmse_px:
        reg.ok = False
        reg.issues.append("weak_alignment")
        if not reg.guidance:
            reg.guidance.append("I could not line this photo up precisely. Place the strip on a plain surface and take the photo from straight above.")
    return reg


def warp_to_reference(current_bgr: np.ndarray, H: np.ndarray, ref_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Warp the photo into the reference frame. Returns (warped, valid_mask)."""
    w, h = ref_size
    warped = cv.warpPerspective(current_bgr, H, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    valid = cv.warpPerspective(np.full(current_bgr.shape[:2], 255, np.uint8), H, (w, h), flags=cv.INTER_NEAREST)
    return warped, valid
