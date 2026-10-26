"""Build the demo photos from REAL strip photos.

Sources (backend/demo_sources/): real phone photos from the public MedCheck
dataset (github.com/MahiiBamba/mini_project_medcheck), with its per-pocket
labels (YOLO format: class 0 = intact, 1 = broken).

  setup photo   the real photo; pockets already broken are registered as empty
  later photos  the same real strip after a dose: intact pockets replaced by a
                real broken pocket cut from the same strip, then re-shot with a
                small change of angle, light and JPEG (eval/run_real_web.py)

    python scripts/make_real_demo_assets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from pillcheck.vision.align import resize_long_side  # noqa: E402
from pillcheck.vision.pockets import Pocket  # noqa: E402
from run_real_web import camera_change  # noqa: E402

SRC, OUT = ROOT / "demo_sources", ROOT / "demo_assets"
STRIPS = {"telmi": "t-34.png", "metf": "k-59.jpeg", "other": "b_1.jpeg"}

MEDICATIONS = [
    {"key": "telmi", "name": "Telmisartan", "strength": "40 mg", "instructions": "After breakfast and dinner",
     "schedule": [{"time": "09:00", "pills": 1}, {"time": "21:00", "pills": 1}], "reference": "telmisartan_reference.jpg"},
    {"key": "metf", "name": "Metformin", "strength": "500 mg", "instructions": "With meals",
     "schedule": [{"time": "08:30", "pills": 1}, {"time": "20:30", "pills": 1}], "reference": "metformin_reference.jpg"},
]


def load(name: str) -> tuple[np.ndarray, list[Pocket]]:
    raw = cv.imread(str(SRC / name))
    img, s = resize_long_side(raw, 1280)
    H, W = raw.shape[:2]
    pockets = []
    for i, line in enumerate((SRC / (Path(name).stem + ".txt")).read_text().split("\n")):
        if not line.strip():
            continue
        c, x, y, w, h = line.split()[:5]
        pockets.append(Pocket(len(pockets), float(x) * W * s, float(y) * H * s, 0.5 * float(w) * W * s,
                              0.5 * float(h) * H * s, 0.0, initially_empty=(c == "1")))
    # reading order: rows top to bottom, then left to right
    r = float(np.median([p.ry for p in pockets]))
    pockets.sort(key=lambda p: (round(p.cy / r), p.cx))
    return img, [Pocket(i, p.cx, p.cy, p.rx, p.ry, 0.0, p.initially_empty) for i, p in enumerate(pockets)]


def take(img: np.ndarray, pockets: list[Pocket], targets: list[int]) -> np.ndarray:
    """Replace the given intact pockets with a real broken pocket from the same strip."""
    out = img.copy()
    broken = [p for p in pockets if p.initially_empty]
    for k, t in enumerate(targets):
        dst, src = pockets[t], broken[k % len(broken)]
        pad = 1.12
        patch = img[int(src.cy - src.ry * pad):int(src.cy + src.ry * pad), int(src.cx - src.rx * pad):int(src.cx + src.rx * pad)]
        w, h = int(2 * dst.rx * pad), int(2 * dst.ry * pad)
        patch = cv.resize(patch, (w, h), interpolation=cv.INTER_AREA)
        mask = np.zeros((h, w), np.uint8)
        cv.ellipse(mask, (w // 2, h // 2), (int(w / 2 / pad), int(h / 2 / pad)), 0, 0, 360, 255, -1)
        out = cv.seamlessClone(patch, out, mask, (int(dst.cx), int(dst.cy)), cv.NORMAL_CLONE)
    return out


def shot(img: np.ndarray, seed: int, blur: float = 0.0, glare_on: Pocket | None = None) -> np.ndarray:
    if glare_on is not None:  # a flash reflection sitting right on one pocket
        p = glare_on
        g = np.zeros(img.shape[:2], np.float32)
        cv.ellipse(g, (int(p.cx), int(p.cy)), (int(p.rx * 1.3), int(p.ry * 1.3)), 0, 0, 360, 1.0, -1)
        img = np.clip(img + 230 * cv.GaussianBlur(g, (0, 0), p.rx * 0.25)[..., None], 0, 255).astype(np.uint8)
    out, _ = camera_change(img, np.random.default_rng(seed), strength=0.5)
    if blur:
        out = cv.GaussianBlur(out, (0, 0), blur)
    return out


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.jpg"):
        f.unlink()
    imgs = {k: load(v) for k, v in STRIPS.items()}
    ti, tp = imgs["telmi"]
    mi, mp = imgs["metf"]
    t_int = [p.id for p in tp if not p.initially_empty]
    m_int = [p.id for p in mp if not p.initially_empty]
    a, b, c = t_int[0], t_int[1], t_int[2]
    files = {
        "telmisartan_reference.jpg": ti,
        "metformin_reference.jpg": mi,
        "telmisartan_dose1.jpg": shot(take(ti, tp, [a]), 21),
        "telmisartan_dose2.jpg": shot(take(ti, tp, [a, b]), 22),
        "telmisartan_extra.jpg": shot(take(ti, tp, [a, b, c]), 23),
        "telmisartan_blurry.jpg": shot(take(ti, tp, [a]), 24, blur=6.0),
        "telmisartan_glare.jpg": shot(take(ti, tp, [a]), 25, glare_on=tp[t_int[3]]),
        "metformin_dose1.jpg": shot(take(mi, mp, [m_int[0]]), 26),
        "unregistered_strip.jpg": shot(imgs["other"][0], 27),
    }
    for name, im in files.items():
        cv.imwrite(str(OUT / name), im, [cv.IMWRITE_JPEG_QUALITY, 90])
    meds = [dict(m, pockets=[p.to_dict() for p in (tp if m["key"] == "telmi" else mp)]) for m in MEDICATIONS]
    S = lambda i, m, t, h: {"id": i, "file": f"{i}.jpg", "medication": m, "title": t, "hint": h}
    samples = [
        S("telmisartan_dose1", "telmi", "Telmisartan, after the 9 AM dose", "One tablet removed. Expect: dose recorded."),
        S("telmisartan_dose2", "telmi", "Telmisartan, after the 9 PM dose", "A second tablet removed. In the morning this is an extra pill; in the evening it is the 9 PM dose."),
        S("telmisartan_extra", "telmi", "Telmisartan, one tablet too many", "More removed than scheduled. Expect: urgent caregiver alert."),
        S("telmisartan_blurry", "telmi", "Telmisartan, blurry photo", "Expect: a retake request with instructions."),
        S("telmisartan_glare", "telmi", "Telmisartan, with flash glare", "Glare on the foil. Pockets under it are not guessed."),
        S("metformin_dose1", "metf", "Metformin, after the 8:30 AM dose", "Try it while Telmisartan is the expected medicine: expect 'wrong strip'."),
        S("unregistered_strip", None, "A strip that was never registered", "Expect: not recognised."),
    ]
    note = ("Real strip photos from the public MedCheck dataset (github.com/MahiiBamba/mini_project_medcheck). "
            "Dose photos: a real empty pocket from the same strip swapped in, re-shot with a small camera change.")
    (OUT / "manifest.json").write_text(json.dumps({"version": 2, "medications": meds, "samples": samples, "note": note}, indent=2))
    print("pockets:", {k: (len(v[1]), sum(p.initially_empty for p in v[1])) for k, v in imgs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
