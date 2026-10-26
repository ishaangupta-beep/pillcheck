"""Expiry-date reading benchmark (semi-real).

No public strip photo we could reach shows a legible expiry date (web images are
~480 px; the date is tiny edge print). So: real strip photos from the MedCheck
dataset, with an expiry line stamped onto the foil in the formats Indian strips
use (inkjet dot-matrix, plain print, embossed-look), then the same random
camera change as the vision benchmark (angle, light, blur, noise, JPEG).

Scored outcomes per photo
  correct   month and year read right
  asked     no date read with confidence; the app asks the caregiver to type it
  wrong     a different date was read (the dangerous case if it is later)

Usage:  python eval/run_expiry.py --data <MedCheck_Combined dir> [--n 150]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pillcheck.vision.align import resize_long_side  # noqa: E402
from pillcheck.vision.expiry import read_expiry  # noqa: E402
from run_real_web import camera_change, load_items  # noqa: E402

MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
OUT = Path(__file__).resolve().parent / "results" / "real_web"


def line_for(y: int, m: int, rng) -> str:
    my, mm = y - int(rng.integers(1, 3)), int(rng.integers(1, 13))
    b = f"{'ABCDEFGHKLMNPRST'[rng.integers(16)]}{rng.integers(1000, 99999)}"
    styles = [
        f"B.No. {b}  MFD {mm:02d}/{my}  EXP {m:02d}/{y}",
        f"EXP. {m:02d}/{y}",
        f"MFG.DT. {MON[mm - 1]}.{my}  EXP.DT. {MON[m - 1]}.{y}",
        f"Exp: {m:02d}/{y % 100:02d}",
        f"BATCH {b}  MFG {mm:02d}/{my % 100:02d}  EXP {m:02d}/{y % 100:02d}",
        f"USE BEFORE {MON[m - 1]} {y}",
        f"M.{mm:02d}/{my}  E.{m:02d}/{y}",  # no readable keyword: falls back to the later date
    ]
    return styles[int(rng.integers(len(styles)))]


def stamp(img: np.ndarray, text: str, rng) -> np.ndarray:
    h, w = img.shape[:2]
    font = [cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_DUPLEX, cv.FONT_HERSHEY_PLAIN][int(rng.integers(3))]
    px = rng.uniform(0.016, 0.03) * max(h, w)  # cap height on the photo, like real edge print
    scale = px / cv.getTextSize("0", font, 1.0, 1)[0][1]
    thick = max(1, int(round(scale * rng.uniform(1.2, 2.2))))
    (tw, th), base = cv.getTextSize(text, font, scale, thick)
    pad = int(th)
    canvas = np.zeros((th + base + 2 * pad, tw + 2 * pad), np.uint8)
    cv.putText(canvas, text, (pad, pad + th), font, scale, 255, thick, cv.LINE_AA)
    kind = rng.choice(["inkjet", "print", "emboss"], p=[0.45, 0.35, 0.2])
    if kind == "inkjet":  # dot-matrix: sample the glyphs on a dot grid
        step = max(2, int(th / 7))
        dots = np.zeros_like(canvas)
        for yy in range(0, canvas.shape[0], step):
            for xx in range(0, canvas.shape[1], step):
                if canvas[yy, xx] > 100:
                    cv.circle(dots, (xx, yy), max(1, int(step * 0.55)), 255, -1)
        canvas = dots
    angle = rng.uniform(-6, 6) + (90 if rng.random() < 0.15 else 0)
    M = cv.getRotationMatrix2D((canvas.shape[1] / 2, canvas.shape[0] / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(canvas.shape[0] * sin + canvas.shape[1] * cos), int(canvas.shape[0] * cos + canvas.shape[1] * sin)
    M[0, 2] += nw / 2 - canvas.shape[1] / 2
    M[1, 2] += nh / 2 - canvas.shape[0] / 2
    canvas = cv.warpAffine(canvas, M, (nw, nh))
    if nw >= w - 4 or nh >= h - 4:
        return None
    # Expiry is printed on the strip's foil: pick a spot on bright foil, not the table.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for _ in range(40):
        x0 = int(rng.integers(2, w - nw - 2))
        y0 = int(rng.integers(2, h - nh - 2))
        if gray[y0:y0 + nh, x0:x0 + nw].mean() > 120:
            break
    else:
        return None
    out = img.astype(np.float32)
    a = cv.GaussianBlur(canvas, (0, 0), 0.6).astype(np.float32)[..., None] / 255.0
    region = out[y0:y0 + nh, x0:x0 + nw]
    if kind == "emboss":  # raised foil: light and dark edge, no ink
        k = np.float32([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        e = cv.filter2D(canvas.astype(np.float32), -1, k)[..., None] / 255.0
        region += 70 * e
    else:
        ink = np.array([rng.uniform(10, 60), rng.uniform(10, 50), rng.uniform(10, 50)], np.float32)
        region[:] = region * (1 - 0.92 * a) + ink * 0.92 * a
    out[y0:y0 + nh, x0:x0 + nw] = region
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    items = [it for it in load_items(Path(a.data)) if it["long_side"] >= 640]
    rng.shuffle(items)
    rows = []
    for it in items[:a.n]:
        img, _ = resize_long_side(it["raw"], 1280)
        y, m = int(rng.integers(2024, 2030)), int(rng.integers(1, 13))
        text = line_for(y, m, rng)
        stamped = stamp(img, text, rng)
        if stamped is None:  # no bright foil in this photo
            continue
        photo, cam = camera_change(stamped, rng, strength=0.6)
        t0 = time.perf_counter()
        r = read_expiry(photo)
        ms = round(1000 * (time.perf_counter() - t0))
        truth = f"{y:04d}-{m:02d}"
        if r["date"] is None:
            outcome = "asked"
        elif r["date"] == truth:
            outcome = "correct"
        else:
            outcome = "wrong_later" if r["date"] > truth else "wrong_earlier"
        rows.append({"name": it["name"], "printed": text, "truth": truth, "read": r["date"], "confidence": r["confidence"],
                     "raw": r.get("raw"), "outcome": outcome, "ms": ms})
        if len(rows) % 25 == 0:
            print(len(rows), dict(Counter(x["outcome"] for x in rows)), flush=True)
    c = Counter(x["outcome"] for x in rows)
    n = len(rows)
    answered = c["correct"] + c["wrong_later"] + c["wrong_earlier"]
    by_conf = {k: dict(Counter(x["outcome"] for x in rows if x["confidence"] == k)) for k in ("high", "medium", "low", "none")}
    rep = {"photos": n, "outcomes": dict(c), "read_correctly": round(c["correct"] / n, 3),
           "accuracy_when_answered": round(c["correct"] / max(1, answered), 3),
           "asked_caregiver": round(c["asked"] / n, 3), "by_confidence": by_conf,
           "median_ms": int(np.median([x["ms"] for x in rows])), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report_expiry.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
