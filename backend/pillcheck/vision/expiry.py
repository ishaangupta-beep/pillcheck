"""Read the expiry date printed on a strip, with OpenCV 5's DNN text models.

    cv.dnn.TextDetectionModel_DB   PP-OCRv3 text detector (opencv_zoo)
    cv.dnn.TextRecognitionModel    CRNN recogniser, 0-9 a-z (opencv_zoo)

Runs once, on the setup photo of a full strip. The caregiver sees what was read
and can correct it, so a misread never silently decides anything: if the date
is not read with confidence the app asks for it instead of assuming.

Indian strips print it in many ways: "EXP. 12/2027", "EXP.DT.:DEC.2027",
"Exp 12/27", "MFG 01/2025 EXP 12/2026", often in dot-matrix ink. The
recogniser has no punctuation, so "12/2027" comes back as "122027" or as two
words; the parser below handles those shapes.
"""
from __future__ import annotations

import calendar
import logging
import re
import threading
from datetime import date

import cv2 as cv
import numpy as np

from ..config import settings

log = logging.getLogger(__name__)

DET_MODEL = "text_detection_en_ppocrv3_2023may.onnx"
REC_MODEL = "text_recognition_CRNN_EN_2021sep.onnx"
VOCAB = list("0123456789abcdefghijklmnopqrstuvwxyz")
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MONTHS.update({"sept": 9})

_lock = threading.Lock()
_models: tuple | None = None


def available() -> bool:
    d = settings.vision.models_dir
    return (d / DET_MODEL).exists() and (d / REC_MODEL).exists()


def _get_models():
    global _models
    with _lock:
        if _models is None:
            d = settings.vision.models_dir
            det = cv.dnn.TextDetectionModel_DB(str(d / DET_MODEL))
            det.setBinaryThreshold(0.3).setPolygonThreshold(0.5).setMaxCandidates(300).setUnclipRatio(2.0)
            det.setInputParams(1.0 / 255.0, (736, 736), (122.67891434, 116.66876762, 104.00698793), True)
            rec = cv.dnn.TextRecognitionModel(str(d / REC_MODEL))
            rec.setDecodeType("CTC-greedy")
            rec.setVocabulary(VOCAB)
            rec.setInputParams(1 / 127.5, (100, 32), (127.5, 127.5, 127.5))
            _models = (det, rec)
        return _models


def _crop(img: np.ndarray, quad) -> np.ndarray:
    q = np.array(quad, np.float32)  # bottom-left, top-left, top-right, bottom-right
    w = int(max(np.linalg.norm(q[1] - q[2]), np.linalg.norm(q[0] - q[3])))
    h = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
    if w < 4 or h < 4:
        return np.zeros((0, 0, 3), np.uint8)
    dst = np.float32([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]])
    return cv.warpPerspective(img, cv.getPerspectiveTransform(q, dst), (w, h))


def read_words(img: np.ndarray, long_side: int = 1920) -> list[dict]:
    """Detect and read every word. Returns [{text, cx, cy, h}] in reading order."""
    det, rec = _get_models()
    H, W = img.shape[:2]
    s = long_side / max(H, W)  # small edge print needs ~1920 px; upsample smaller photos
    work = cv.resize(img, (max(32, int(W * s) // 32 * 32), max(32, int(H * s) // 32 * 32)),
                     interpolation=cv.INTER_AREA if s < 1 else cv.INTER_CUBIC)
    det.setInputSize((work.shape[1], work.shape[0]))
    gray = cv.cvtColor(cv.cvtColor(work, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    quads, _ = det.detect(work)
    words = []
    for q in quads:
        c = _crop(gray, q)
        if c.size == 0:
            continue
        if c.shape[0] > 1.5 * c.shape[1]:
            c = cv.rotate(c, cv.ROTATE_90_CLOCKWISE)
        # Read at the crop's own aspect ratio: squeezing a whole printed line
        # into the default 100x32 input makes it unreadable.
        width = int(np.clip(round(32 * c.shape[1] / max(1, c.shape[0])), 32, 1600))
        rec.setInputParams(1 / 127.5, (width, 32), (127.5, 127.5, 127.5))
        t = rec.recognize(c)
        if t:
            pts = np.array(q, np.float32)
            words.append({"text": t, "cx": float(pts[:, 0].mean()), "cy": float(pts[:, 1].mean()),
                          "h": float(np.linalg.norm(pts[0] - pts[1]))})
    # Reading order: group into lines by vertical position.
    words.sort(key=lambda w: w["cy"])
    lines: list[list[dict]] = []
    for w in words:
        if lines and abs(w["cy"] - np.mean([x["cy"] for x in lines[-1]])) < 0.6 * max(w["h"], 6):
            lines[-1].append(w)
        else:
            lines.append([w])
    return [w for ln in lines for w in sorted(ln, key=lambda w: w["cx"])]


# ---------------------------------------------------------------- parsing

EXP_KEY = re.compile(r"(exp|expiry|expdt|expdate|expires|usebefore|useby|bestbefore|eyp|cxp|exd)")
MFG_KEY = re.compile(r"(mfg|mfd|mfgdt|mfddt|mfgdate|manuf|mf9|mtg|mfo)")


def _year(y: str) -> int | None:
    v = int(y)
    if len(y) == 2:
        v += 2000
    return v if 2015 <= v <= 2045 else None


KEYS = re.compile(r"(exp(?:iry|dt|date|ires)?|use ?before|use ?by|best ?before|mf[gd](?:dt|date)?|manuf\w*)")
DIGITISH = str.maketrans({"o": "0", "q": "0", "d": "0", "i": "1", "l": "1", "j": "1", "z": "2", "s": "5", "b": "8", "g": "9", "t": "7"})


def normalise(text: str) -> str:
    """Split glued keywords off and undo the recogniser's usual digit mix-ups
    ('o' for 0, 'i'/'l' for 1, '/' read as 1) inside number-like runs."""
    text = KEYS.sub(lambda m: f" {m.group(0)} ", text.lower())
    month = "|".join(sorted(MONTHS, key=len, reverse=True))
    # "sepzo29" -> "sep 2029": a month name glued to a digit-like run
    text = re.sub(rf"(?<![a-z])({month})\.?([0-9oqdilzsbgt]{{2,4}})(?![a-z])",
                  lambda m: f" {m.group(1)} {m.group(2).translate(DIGITISH)} ", text)
    out = []
    for tok in text.split():
        digits = sum(ch.isdigit() for ch in tok)
        if KEYS.fullmatch(tok) or tok in MONTHS or digits < 0.5 * len(tok):
            out.append(tok)
        else:
            out.append(tok.translate(DIGITISH))
    return " ".join(out)


def _dates(s: str) -> list[tuple[int, int, int, int]]:
    """All (start, end, year, month) date shapes found in a normalised string."""
    out = []
    for m in re.finditer(r"(jan|feb|mar|apr|may|jun|jul|aug|sept|sep|oct|nov|dec)\s?(\d{4}|\d{2})(?!\d)", s):
        y = _year(m.group(2))
        if y:
            out.append((m.start(), m.end(), y, MONTHS[m.group(1)]))
    # 12/2027 read as "12 2027", "122027" or, slash taken for a 1, "1212027"
    for m in re.finditer(r"(?<!\d)(\d{1,2})\s?1?\s?(20[1-4]\d)(?!\d)", s):
        mo, y = int(m.group(1)), _year(m.group(2))
        if 1 <= mo <= 12 and y and not any(a <= m.start() < b for a, b, *_ in out):
            out.append((m.start(), m.end(), y, mo))
    for m in re.finditer(r"(?<!\d)(\d{2})\s?1?\s?(\d{2})(?!\d)", s):  # 12/27
        mo, y = int(m.group(1)), _year(m.group(2))
        if 1 <= mo <= 12 and y and not any(a <= m.start() < b for a, b, *_ in out):
            out.append((m.start(), m.end(), y, mo))
    return sorted(out)


def parse_expiry(words: list[str]) -> dict | None:
    """Pick the expiry date out of the recognised words."""
    text = normalise(" ".join(words))
    dates = _dates(text)
    if not dates:
        return None
    best = None
    for k in EXP_KEY.finditer(text):
        after = [d for d in dates if d[0] >= k.start() and d[0] - k.end() <= 14]
        if after:
            best = (after[0], "high")
            break
    if best is None:
        mfg = []
        for k in MFG_KEY.finditer(text):
            nxt = [d for d in dates if d[0] >= k.start() and d[0] - k.end() <= 14]
            if nxt:
                mfg.append(nxt[0])
        rest = [d for d in dates if d not in mfg]
        if mfg and rest:
            best = (max(rest, key=lambda d: (d[2], d[3])), "medium")
        elif len(dates) >= 2:
            best = (max(dates, key=lambda d: (d[2], d[3])), "low")  # MFG then EXP, keyword unreadable
        else:
            best = (dates[0], "low")
    (a, b, y, mo), conf = best
    return {"date": f"{y:04d}-{mo:02d}", "confidence": conf, "raw": text[max(0, a - 12):b]}


def read_expiry(img: np.ndarray) -> dict:
    """Expiry from a strip photo: {date: 'YYYY-MM' | None, confidence, raw, source}."""
    if not available():
        return {"date": None, "confidence": "none", "raw": "", "source": "photo", "note": "text models not installed"}
    found = None
    for rot in (None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_90_COUNTERCLOCKWISE, cv.ROTATE_180):
        im = img if rot is None else cv.rotate(img, rot)
        try:
            words = [w["text"] for w in read_words(im)]
        except cv.error as e:  # pragma: no cover - model/runtime problems must not block setup
            log.warning("text reading failed: %s", e)
            break
        r = parse_expiry(words)
        if r and (found is None or ("high", "medium", "low").index(r["confidence"]) < ("high", "medium", "low").index(found["confidence"])):
            found = r
        if found and found["confidence"] == "high":
            break
    if not found:
        return {"date": None, "confidence": "none", "raw": "", "source": "photo"}
    return {**found, "source": "photo"}


# ---------------------------------------------------------------- status

def last_valid_day(yyyy_mm: str) -> date:
    """'EXP 12/2027' means usable through 31 Dec 2027."""
    y, m = map(int, yyyy_mm.split("-"))
    return date(y, m, calendar.monthrange(y, m)[1])


def status(yyyy_mm: str | None, today: date, soon_days: int = 30) -> str | None:
    if not yyyy_mm:
        return None
    last = last_valid_day(yyyy_mm)
    if today > last:
        return "expired"
    if (last - today).days <= soon_days:
        return "expiring_soon"
    return "ok"
