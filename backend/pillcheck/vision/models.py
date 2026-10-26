"""Lazy, process-wide loading of the OpenCV 5 deep feature models.

ALIKED (detector + descriptor) and LightGlue (matcher) are ONNX models run by
the new OpenCV 5 DNN engine. Loading costs ~1 s, so we load once and reuse.
If the files are missing we fall back to ORB so the service still runs, and
report which engine was used in every result.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from ..config import settings

log = logging.getLogger(__name__)
_lock = threading.Lock()
_engine: "FeatureEngine | None" = None


@dataclass
class Features:
    keypoints: np.ndarray  # (N, 2) float32 pixel coordinates
    descriptors: np.ndarray  # (N, D)
    image_size: tuple[int, int]  # (width, height)


class FeatureEngine:
    name = "base"

    def extract(self, image_bgr: np.ndarray) -> Features:
        raise NotImplementedError

    def match(self, a: Features, b: Features) -> tuple[np.ndarray, np.ndarray]:
        """Return matched point arrays (pts_a, pts_b), each (M, 2) float32."""
        raise NotImplementedError


class AlikedLightGlue(FeatureEngine):
    name = "aliked+lightglue"

    def __init__(self) -> None:
        cfg = settings.vision
        params = cv.ALIKED.Params()
        params.inputSize = (cfg.aliked_input, cfg.aliked_input)
        self._detector = cv.ALIKED.create(str(cfg.models_dir / cfg.aliked_model), params)
        self._matcher = cv.LightGlueMatcher.create(
            str(cfg.models_dir / cfg.lightglue_model), 0.0
        )
        # The DNN nets are not re-entrant; serialise inference.
        self._infer_lock = threading.Lock()

    def extract(self, image_bgr: np.ndarray) -> Features:
        with self._infer_lock:
            kps, desc = self._detector.detectAndCompute(image_bgr, None)
        pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
        h, w = image_bgr.shape[:2]
        return Features(pts, np.asarray(desc, np.float32), (w, h))

    def match(self, a: Features, b: Features) -> tuple[np.ndarray, np.ndarray]:
        if len(a.keypoints) < 4 or len(b.keypoints) < 4:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        # OpenCV 5.0.0's LightGlueMatcher dies with SIGFPE (whole process) when
        # a pair yields zero matches, e.g. two unrelated strips. Append a few
        # identical anchor points to both sides so there is always a match,
        # then drop matches that involve them.
        na, nb = len(a.keypoints), len(b.keypoints)
        ka = np.vstack([a.keypoints, _anchor_points(a.image_size)])
        kb = np.vstack([b.keypoints, _anchor_points(b.image_size)])
        da = np.vstack([a.descriptors, _ANCHOR_DESC]).astype(np.float32)
        db = np.vstack([b.descriptors, _ANCHOR_DESC]).astype(np.float32)
        with self._infer_lock:
            self._matcher.setPairInfo(ka, kb, a.image_size, b.image_size)
            matches = self._matcher.match(da, db)
        matches = [m for m in matches if m.queryIdx < na and m.trainIdx < nb]
        if not matches:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        qi = np.array([m.queryIdx for m in matches])
        ti = np.array([m.trainIdx for m in matches])
        return a.keypoints[qi], b.keypoints[ti]


_rng = np.random.default_rng(1234)
_ANCHOR_DESC = _rng.normal(size=(4, 128)).astype(np.float32)
_ANCHOR_DESC /= np.linalg.norm(_ANCHOR_DESC, axis=1, keepdims=True)


def _anchor_points(size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.float32([[2, 2], [w - 3, 2], [w - 3, h - 3], [2, h - 3]])


class OrbFallback(FeatureEngine):
    name = "orb"

    def __init__(self) -> None:
        self._orb = cv.ORB_create(nfeatures=3000)
        self._bf = cv.BFMatcher(cv.NORM_HAMMING)

    def extract(self, image_bgr: np.ndarray) -> Features:
        gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
        kps, desc = self._orb.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
        if desc is None:
            desc = np.empty((0, 32), np.uint8)
        h, w = image_bgr.shape[:2]
        return Features(pts, desc, (w, h))

    def match(self, a: Features, b: Features) -> tuple[np.ndarray, np.ndarray]:
        if len(a.descriptors) < 2 or len(b.descriptors) < 2:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        pairs = self._bf.knnMatch(a.descriptors, b.descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if not good:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        qi = np.array([m.queryIdx for m in good])
        ti = np.array([m.trainIdx for m in good])
        return a.keypoints[qi], b.keypoints[ti]


def get_engine() -> FeatureEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                cfg = settings.vision
                have = all((cfg.models_dir / f).exists() for f in (cfg.aliked_model, cfg.lightglue_model))
                if os.environ.get("PILLCHECK_FEATURES", "").lower() == "orb":
                    _engine = OrbFallback()  # baseline for the evaluation
                elif have:
                    _engine = AlikedLightGlue()
                else:
                    log.warning("ALIKED/LightGlue models not found in %s; using ORB fallback", cfg.models_dir)
                    _engine = OrbFallback()
    return _engine


def features_to_npz(f: Features) -> dict[str, np.ndarray]:
    return {"kp": f.keypoints, "desc": f.descriptors, "size": np.array(f.image_size)}


def features_from_npz(d) -> Features:
    size = tuple(int(x) for x in d["size"])
    return Features(d["kp"].astype(np.float32), d["desc"], (size[0], size[1]))
