"""Procedural blister-strip photos with exact ground truth.

Used by the test-suite and the evaluation harness, alongside real photos, to
measure the pipeline under controlled nuisances: viewpoint, rotation, scale,
lighting gradients, white-balance shift, blur, sensor noise, specular glare and
JPEG compression. Every generated photo comes with the true set of popped
pockets and the true homography.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

PILL_COLOURS = [(235, 235, 240), (120, 170, 235), (90, 120, 220), (200, 220, 245), (150, 210, 160)]


@dataclass
class StripSpec:
    name: str = "TELMISARTAN 40mg"
    rows: int = 2
    cols: int = 5
    seed: int = 0
    pill_colour: tuple = (235, 235, 240)
    foil_colour: tuple = (192, 192, 198)
    ink_colour: tuple = (40, 60, 150)
    width: int = 1000
    height: int = 460


@dataclass
class Shot:
    image: np.ndarray
    popped: set[int]
    H_ref_to_photo: np.ndarray
    meta: dict = field(default_factory=dict)


def _layout(spec: StripSpec) -> list[tuple[int, int, int, int]]:
    w, h = spec.width, spec.height
    top, bottom = 0.22 * h, 0.86 * h
    rx = int(0.36 * w / spec.cols)
    ry = int(0.34 * (bottom - top) / spec.rows)
    out = []
    for r in range(spec.rows):
        for c in range(spec.cols):
            cx = int((c + 0.5) * w / spec.cols)
            cy = int(top + (r + 0.5) * (bottom - top) / spec.rows)
            out.append((cx, cy, rx, ry))
    return out


def render_strip(spec: StripSpec, popped: set[int] = frozenset()) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Render the strip flat, in its own frame. Returns (image, pocket ellipses)."""
    rng = np.random.default_rng(spec.seed)
    w, h = spec.width, spec.height
    base = np.zeros((h, w, 3), np.float32)
    base[:] = spec.foil_colour
    # Brushed-foil texture: anisotropic noise, fixed per strip (seeded).
    tex = rng.normal(0, 1, (h, w)).astype(np.float32)
    tex = cv.GaussianBlur(tex, (0, 0), sigmaX=6, sigmaY=0.7) * 9
    base += tex[..., None]
    img = np.clip(base, 0, 255).astype(np.uint8)

    font = cv.FONT_HERSHEY_DUPLEX
    cv.putText(img, spec.name, (int(0.04 * w), int(0.13 * h)), font, 1.3 * w / 1000, spec.ink_colour, 2, cv.LINE_AA)
    cv.putText(img, "Rx only  |  Store below 25C", (int(0.60 * w), int(0.13 * h)), cv.FONT_HERSHEY_SIMPLEX, 0.6 * w / 1000, (60, 60, 60), 1, cv.LINE_AA)
    cv.putText(img, f"B.No. {1000 + spec.seed}  MFD 01/2026  EXP 12/2027", (int(0.04 * w), int(0.96 * h)),
               cv.FONT_HERSHEY_SIMPLEX, 0.7 * w / 1000, (35, 35, 35), 2, cv.LINE_AA)
    # Perforation lines between pockets.
    for c in range(1, spec.cols):
        x = int(c * w / spec.cols)
        for y in range(int(0.2 * h), int(0.9 * h), 14):
            cv.line(img, (x, y), (x, y + 6), (150, 150, 155), 2)

    pockets = _layout(spec)
    for i, (cx, cy, rx, ry) in enumerate(pockets):
        cv.ellipse(img, (cx, cy), (rx + 6, ry + 6), 0, 0, 360, (160, 160, 168), 3, cv.LINE_AA)
        if i in popped:
            # Torn foil: dark cavity, jagged rim, curled flap catching light.
            cv.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (62, 58, 55), -1, cv.LINE_AA)
            pts = []
            for t in np.linspace(0, 2 * np.pi, 28, endpoint=False):
                k = 1.0 + rng.uniform(-0.12, 0.05)
                pts.append((cx + k * rx * np.cos(t), cy + k * ry * np.sin(t)))
            cv.polylines(img, [np.int32(pts)], True, (205, 205, 210), 2, cv.LINE_AA)
            flap = np.int32([(cx - rx * 0.7, cy - ry * 0.2), (cx + rx * 0.4, cy - ry * 0.55), (cx + rx * 0.2, cy + ry * 0.1)])
            cv.fillConvexPoly(img, flap, (215, 215, 220), cv.LINE_AA)
        else:
            # Dome with the pill seen through it, plus a specular highlight.
            cv.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (178, 178, 186), -1, cv.LINE_AA)
            cv.ellipse(img, (cx, cy), (int(rx * 0.78), int(ry * 0.72)), 0, 0, 360, spec.pill_colour, -1, cv.LINE_AA)
            cv.ellipse(img, (cx, cy), (int(rx * 0.78), int(ry * 0.72)), 0, 0, 360, (120, 120, 125), 1, cv.LINE_AA)
            cv.line(img, (cx - int(rx * 0.5), cy), (cx + int(rx * 0.5), cy), (150, 150, 155), 1, cv.LINE_AA)
            cv.ellipse(img, (cx - int(rx * 0.35), cy - int(ry * 0.35)), (int(rx * 0.22), int(ry * 0.1)), -25, 0, 360, (245, 245, 248), -1, cv.LINE_AA)
    return img, pockets


def _background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    colour = rng.uniform(60, 170, 3)
    bg = np.ones((h, w, 3), np.float32) * colour
    grain = cv.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=1.5, sigmaY=12) * 12
    return np.clip(bg + grain[..., None], 0, 255).astype(np.uint8)


def photograph(strip: np.ndarray, rng: np.random.Generator, *, out_size=(1280, 960), difficulty: float = 0.5,
               glare: bool = False, blur_sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a hand-held phone photo of a flat strip. Returns (photo, H strip->photo)."""
    ow, oh = out_size
    sh, sw = strip.shape[:2]
    d = difficulty
    scale = rng.uniform(0.62, 0.9) * min(ow / sw, oh / sh)
    angle = rng.uniform(-25, 25) * d
    cx, cy = ow / 2 + rng.uniform(-0.06, 0.06) * ow, oh / 2 + rng.uniform(-0.06, 0.06) * oh
    A = cv.getRotationMatrix2D((sw / 2, sh / 2), angle, scale)
    A[0, 2] += cx - sw / 2
    A[1, 2] += cy - sh / 2
    H = np.vstack([A, [0, 0, 1]])
    # Perspective tilt.
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    dst = cv.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    jitter = rng.uniform(-1, 1, (4, 2)) * 0.07 * d * np.array([sw, sh]) * scale
    H = cv.getPerspectiveTransform(src, np.float32(dst + jitter))

    photo = _background(oh, ow, rng)
    warped = cv.warpPerspective(strip, H, (ow, oh))
    m = cv.warpPerspective(np.full(strip.shape[:2], 255, np.uint8), H, (ow, oh))
    m = cv.GaussianBlur(m, (3, 3), 0).astype(np.float32)[..., None] / 255
    shadow = cv.GaussianBlur(cv.warpPerspective(np.full(strip.shape[:2], 255, np.uint8), H @ np.array([[1, 0, 8], [0, 1, 10], [0, 0, 1]]), (ow, oh)), (31, 31), 0)
    photo = (photo.astype(np.float32) * (1 - 0.35 * shadow[..., None] / 255))
    photo = photo * (1 - m) + warped.astype(np.float32) * m

    # Lighting: gain, offset, linear gradient and a white-balance cast.
    gain = rng.uniform(1 - 0.35 * d, 1 + 0.2 * d)
    gx, gy = rng.uniform(-0.25, 0.25, 2) * d
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
    grad = 1 + gx * (xx / ow - 0.5) + gy * (yy / oh - 0.5)
    cast = 1 + rng.uniform(-0.08, 0.08, 3) * d
    photo = photo * gain * grad[..., None] * cast + rng.uniform(-12, 12) * d

    if glare:
        gxc, gyc = rng.uniform(0.3, 0.7) * ow, rng.uniform(0.3, 0.7) * oh
        g = np.exp(-(((xx - gxc) / (0.07 * ow)) ** 2 + ((yy - gyc) / (0.05 * oh)) ** 2))
        photo = photo + 255 * g[..., None]
    photo = np.clip(photo + rng.normal(0, 2.5 + 3 * d, photo.shape), 0, 255).astype(np.uint8)
    if blur_sigma > 0:
        photo = cv.GaussianBlur(photo, (0, 0), blur_sigma)
    q = int(rng.uniform(70, 95))
    ok, enc = cv.imencode(".jpg", photo, [cv.IMWRITE_JPEG_QUALITY, q])
    return cv.imdecode(enc, cv.IMREAD_COLOR), H


def make_shot(spec: StripSpec, popped: set[int], seed: int, **kw) -> Shot:
    rng = np.random.default_rng(seed)
    strip, _ = render_strip(spec, popped)
    img, H = photograph(strip, rng, **kw)
    return Shot(img, set(popped), H, {"seed": seed, **{k: v for k, v in kw.items() if k != "out_size"}})


def pockets_in_strip_frame(spec: StripSpec) -> list[tuple[int, int, int, int]]:
    return _layout(spec)


def match_ground_truth(spec: StripSpec, H_strip_to_photo: np.ndarray, scale: float, detected_centres: list[tuple[float, float]]) -> dict[int, int]:
    """Map synthetic pocket index -> detected pocket id, by nearest centre."""
    centres = np.float32([(cx, cy) for cx, cy, _, _ in _layout(spec)]).reshape(-1, 1, 2)
    proj = cv.perspectiveTransform(centres, H_strip_to_photo).reshape(-1, 2) * scale
    det = np.float32(detected_centres)
    mapping = {}
    for i, p in enumerate(proj):
        d = np.linalg.norm(det - p, axis=1)
        mapping[i] = int(np.argmin(d))
    return mapping
