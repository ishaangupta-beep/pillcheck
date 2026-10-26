"""Benchmark on real, publicly available blister-strip photos.

Data: MedCheck dataset (github.com/MahiiBamba/mini_project_medcheck), 459 real
phone photos of (mostly Indian) strips, every pocket labelled intact/broken.

The online photos are single shots, while PillCheck compares a later photo of a
strip with the setup photo of the same strip. So each real photo is used as:

  setup photo   the real photo as-is; pockets already broken are marked empty
                (what the caregiver does by tapping them in Setup)
  later photo   the same real photo re-shot with a random camera change
                (angle, distance, rotation, light, white balance, blur, noise,
                JPEG, sometimes a reflection) and k = 0, 1 or 2 intact pockets
                replaced by a real broken pocket cut from the same strip.

Tests
  A  pocket finding at setup (detector vs. labelled pockets)
  B  taken / not taken per photo (ground truth: which pockets were swapped)
  C  identity: is this the right medicine, a different registered one, or none

Usage:  python eval/run_real_web.py --data <MedCheck_Combined dir> [--engine aliked|orb] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pillcheck.config import settings  # noqa: E402
from pillcheck.vision.align import resize_long_side  # noqa: E402
from pillcheck.vision.models import get_engine  # noqa: E402
from pillcheck.vision.pipeline import ReferenceBundle, analyze_check  # noqa: E402
from pillcheck.vision.pockets import Pocket, detect_pockets, detect_strip_polygon  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "real_web"


# ------------------------------------------------------------------ data
def load_items(root: Path) -> list[dict]:
    items = []
    for img_path in sorted(root.glob("images/*/*")):
        lab = root / "labels" / img_path.parent.name / (img_path.stem + ".txt")
        if not lab.exists():
            continue
        raw = cv.imread(str(img_path))
        if raw is None:
            continue
        H, W = raw.shape[:2]
        rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        boxes = []
        for r in rows:
            c, x, y, w, h = int(r[0]), *map(float, r[1:5])
            boxes.append((c, x * W, y * H, w * W, h * H))
        items.append({"name": img_path.name, "path": str(img_path), "raw": raw, "boxes": boxes,
                      "long_side": max(H, W)})
    return items


def working(item: dict) -> tuple[np.ndarray, list[Pocket]]:
    img, s = resize_long_side(item["raw"], settings.vision.work_long_side)
    pockets = []
    for i, (c, x, y, w, h) in enumerate(item["boxes"]):
        pockets.append(Pocket(i, x * s, y * s, 0.5 * w * s, 0.5 * h * s, 0.0, initially_empty=(c == 1)))
    return img, pockets


# ------------------------------------------------------------------ realistic changes
def swap_in_broken(img: np.ndarray, pockets: list[Pocket], k: int, rng) -> tuple[np.ndarray, list[int]]:
    """Replace k intact pockets with real broken pockets cut from the same strip."""
    intact = [p for p in pockets if not p.initially_empty]
    broken = [p for p in pockets if p.initially_empty]
    if k == 0 or not broken or len(intact) < k:
        return img.copy(), []
    out = img.copy()
    targets = list(rng.choice(len(intact), size=k, replace=False))
    swapped = []
    for t in targets:
        dst, src = intact[t], broken[rng.integers(len(broken))]
        pad = 1.12
        sx0, sy0 = int(src.cx - src.rx * pad), int(src.cy - src.ry * pad)
        sx1, sy1 = int(src.cx + src.rx * pad), int(src.cy + src.ry * pad)
        patch = img[max(0, sy0):sy1, max(0, sx0):sx1]
        w, h = int(2 * dst.rx * pad), int(2 * dst.ry * pad)
        if patch.size == 0 or w < 6 or h < 6:
            continue
        patch = cv.resize(patch, (w, h), interpolation=cv.INTER_AREA)
        mask = np.zeros((h, w), np.uint8)
        cv.ellipse(mask, (w // 2, h // 2), (int(w / 2 / pad * 1.02), int(h / 2 / pad * 1.02)), 0, 0, 360, 255, -1)
        center = (int(dst.cx), int(dst.cy))
        x0, y0 = center[0] - w // 2, center[1] - h // 2
        if x0 < 1 or y0 < 1 or x0 + w >= out.shape[1] - 1 or y0 + h >= out.shape[0] - 1:
            continue
        out = cv.seamlessClone(patch, out, mask, center, cv.NORMAL_CLONE)
        swapped.append(dst.id)
    return out, swapped


def camera_change(img: np.ndarray, rng, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    j = 0.06 * strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-j, j, (4, 2)).astype(np.float32) * np.float32([w, h])
    ang, sc = rng.uniform(-12, 12) * strength, rng.uniform(0.85, 1.08)
    R = cv.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    R = np.vstack([R, [0, 0, 1]])
    H = R @ cv.getPerspectiveTransform(src, dst)
    out = cv.warpPerspective(img, H, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
    f = out.astype(np.float32)
    gain, bias, gamma = rng.uniform(0.75, 1.25), rng.uniform(-15, 15), rng.uniform(0.85, 1.2)
    wb = rng.uniform(0.92, 1.08, 3)
    f = np.clip(255 * ((np.clip(f * gain + bias, 0, 255) / 255) ** gamma) * wb, 0, 255)
    glare = rng.random() < 0.2
    if glare:
        gm = np.zeros((h, w), np.float32)
        cv.ellipse(gm, (int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h)),
                   (int(0.08 * w), int(0.05 * h)), rng.uniform(0, 180), 0, 360, 1.0, -1)
        gm = cv.GaussianBlur(gm, (0, 0), 0.03 * w)
        f = f + 160 * gm[..., None]
    sigma = rng.uniform(0, 1.0)
    if sigma > 0.3:
        f = cv.GaussianBlur(f, (0, 0), sigma)
    f = f + rng.normal(0, rng.uniform(0, 4), f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    q = int(rng.integers(70, 93))
    ok, buf = cv.imencode(".jpg", out, [cv.IMWRITE_JPEG_QUALITY, q])
    return cv.imdecode(buf, cv.IMREAD_COLOR), {"rot": round(ang, 1), "scale": round(sc, 2), "blur": round(sigma, 2),
                                               "jpeg": q, "glare": glare}


def jpeg_bytes(img: np.ndarray) -> bytes:
    return cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def bundle(mid: str, img: np.ndarray, pockets: list[Pocket]) -> ReferenceBundle:
    feats = get_engine().extract(img)
    return ReferenceBundle(mid, img, feats, pockets, detect_strip_polygon(img, pockets))


# ------------------------------------------------------------------ tests
def test_detection(items: list[dict]) -> dict:
    """A: does Setup find the right pockets on a real strip photo?"""
    rows = []
    for it in items:
        img, gt = working(it)
        fullish = sum(not p.initially_empty for p in gt) / max(1, len(gt)) >= 0.7
        det, conf = detect_pockets(img)
        used = set()
        hit = 0
        for d in det:
            for g in gt:
                if g.id in used:
                    continue
                if abs(d.cx - g.cx) <= g.rx and abs(d.cy - g.cy) <= g.ry:
                    used.add(g.id)
                    hit += 1
                    break
        rows.append({"name": it["name"], "gt": len(gt), "det": len(det), "hit": hit, "fullish": fullish, "conf": conf})

    def summary(rs):
        gt = sum(r["gt"] for r in rs)
        det = sum(r["det"] for r in rs)
        hit = sum(r["hit"] for r in rs)
        exact = sum(r["hit"] == r["gt"] == r["det"] for r in rs)
        return {"photos": len(rs), "pockets": gt, "recall": round(hit / max(1, gt), 3),
                "precision": round(hit / max(1, det), 3), "all_pockets_exact": round(exact / max(1, len(rs)), 3)}

    return {"all": summary(rows), "mostly_full_strips": summary([r for r in rows if r["fullish"]]), "rows": rows}


def test_taken(items: list[dict], rng, limit: int | None) -> dict:
    """B: one later photo per strip; was the right pocket (or none) detected as taken?"""
    rows = []
    pool = [it for it in items if sum(c == 0 for c, *_ in it["boxes"]) >= 2]
    if limit:
        pool = pool[:limit]
    for n, it in enumerate(pool):
        img, pockets = working(it)
        has_broken = any(p.initially_empty for p in pockets)
        k = int(rng.choice([0, 1, 1, 2])) if has_broken else 0
        later, swapped = swap_in_broken(img, pockets, k, rng)
        later, cam = camera_change(later, rng)
        ref = bundle("med", img, pockets)
        t0 = time.perf_counter()
        res = analyze_check(jpeg_bytes(later), [ref], "med", {}, [])
        ms = round(1000 * (time.perf_counter() - t0))
        status = res["status"]
        newly = sorted(res.get("newly_popped", []))
        relevant = [r for r in res.get("pockets", []) if not pockets[r["id"]].initially_empty]
        unsure = [r["id"] for r in relevant if r["state"] in ("uncertain", "unknown")]
        truth = sorted(swapped)
        if status != "ok":
            outcome = "asked_retake" if status in ("retake", "unrecognised") else status
        elif newly == truth and not unsure:
            outcome = "correct"
        elif set(newly) <= set(truth) and unsure and set(truth) - set(newly) <= set(unsure):
            outcome = "asked_retake"  # nothing wrong recorded; the agent asks for a clearer photo
        else:
            outcome = "wrong"
        false_taken = sorted(set(newly) - set(truth))
        missed = sorted(set(truth) - set(newly) - set(unsure))
        rows.append({"name": it["name"], "k": len(truth), "truth": truth, "pred": newly, "unsure": unsure,
                     "status": status, "outcome": outcome, "false_taken": false_taken, "missed": missed,
                     "inliers": (res.get("registration") or {}).get("inliers"),
                     "issues": (res.get("quality") or {}).get("issues"), "camera": cam, "ms": ms})
        if outcome == "wrong" and len([r for r in rows if r["outcome"] == "wrong"]) <= 12:
            ev = res.get("_evidence_jpeg")
            if ev:
                (OUT / "failures").mkdir(parents=True, exist_ok=True)
                (OUT / "failures" / f"taken_{it['name'].rsplit('.', 1)[0]}.jpg").write_bytes(ev)
        if (n + 1) % 25 == 0:
            c = Counter(r["outcome"] for r in rows)
            print(f"  taken test {n + 1}/{len(pool)}: {dict(c)}", flush=True)

    c = Counter(r["outcome"] for r in rows)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    pk_total = sum(len(pockets) for pockets in [r["truth"] for r in ok_rows])
    return {
        "photos": len(rows),
        "outcomes": dict(c),
        "accuracy_when_answered": round(c["correct"] / max(1, c["correct"] + c["wrong"]), 3),
        "correct_rate": round(c["correct"] / max(1, len(rows)), 3),
        "asked_retake_rate": round(c["asked_retake"] / max(1, len(rows)), 3),
        "wrong_rate": round(c["wrong"] / max(1, len(rows)), 3),
        "false_dose_photos": sum(bool(r["false_taken"]) for r in rows),
        "missed_dose_photos": sum(bool(r["missed"]) for r in rows),
        "taken_pockets_truth": sum(r["k"] for r in rows), "taken_pockets_found": sum(len(set(r["pred"]) & set(r["truth"])) for r in rows),
        "median_ms": int(np.median([r["ms"] for r in rows])) if rows else None,
        "rows": rows,
        "_unused": pk_total,
    }


def test_identity(items: list[dict], rng, n_refs: int) -> dict:
    """C: register n different real strips; is each later photo matched to its own strip?"""
    pool = [it for it in items if sum(c == 0 for c, *_ in it["boxes"]) >= 2 and it["long_side"] >= 600]
    idx = rng.choice(len(pool), size=min(n_refs, len(pool)), replace=False)
    chosen = [pool[i] for i in idx]
    refs = []
    for i, it in enumerate(chosen):
        img, pockets = working(it)
        refs.append((f"m{i}", it, img, pockets, bundle(f"m{i}", img, pockets)))
    rows = []
    for i, (mid, it, img, pockets, b) in enumerate(refs):
        later, _ = camera_change(img, rng)
        data = jpeg_bytes(later)
        # Hard case: the app expects a *different* medicine to be due now.
        other = refs[(i + 1 + int(rng.integers(len(refs) - 1))) % len(refs)][0]
        res = analyze_check(data, [r[4] for r in refs], other, {}, [])
        got = (res.get("identity") or {}).get("matched_medication_id")
        rows.append({"name": it["name"], "registered": True, "expected_due": other, "truth": mid, "matched": got,
                     "status": res["status"], "appearance": (res.get("identity") or {}).get("appearance")})
        # Unregistered: the same photo when its own strip is NOT registered.
        res2 = analyze_check(data, [r[4] for r in refs if r[0] != mid], other, {}, [])
        got2 = (res2.get("identity") or {}).get("matched_medication_id")
        rows.append({"name": it["name"], "registered": False, "expected_due": other, "truth": None, "matched": got2,
                     "status": res2["status"], "appearance": (res2.get("identity") or {}).get("appearance")})
        print(f"  identity {i + 1}/{len(refs)}: own={got == mid} unregistered_rejected={got2 is None}", flush=True)
    reg = [r for r in rows if r["registered"]]
    unreg = [r for r in rows if not r["registered"]]
    return {
        "registered_strips": len(refs),
        "right_strip_identified": sum(r["matched"] == r["truth"] for r in reg),
        "wrong_medicine_flagged": sum(r["matched"] == r["truth"] and r["status"] == "wrong_medication" for r in reg),
        "confused_with_other_strip": sum(r["matched"] not in (None, r["truth"]) for r in reg),
        "not_recognised": sum(r["matched"] is None for r in reg),
        "unregistered_tests": len(unreg),
        "unregistered_rejected": sum(r["matched"] is None for r in unreg),
        "unregistered_false_match": sum(r["matched"] is not None for r in unreg),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--engine", default=None, help="aliked (default) or orb")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refs", type=int, default=40)
    ap.add_argument("--skip", default="", help="comma list of tests to skip: detect,taken,identity")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.engine == "orb":
        os.environ["PILLCHECK_FEATURES"] = "orb"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    items = load_items(Path(a.data))
    usable = [it for it in items if it["long_side"] >= settings.vision.min_long_side]
    engine = get_engine().name
    print(f"{len(items)} labelled real photos, {len(usable)} at phone resolution (>= {settings.vision.min_long_side}px); engine {engine}")
    skip = set(a.skip.split(","))
    report = {"dataset": "MedCheck (github.com/MahiiBamba/mini_project_medcheck)", "engine": engine,
              "photos_total": len(items), "photos_used": len(usable), "seed": a.seed}
    if "detect" not in skip:
        report["A_pocket_finding"] = test_detection(usable)
        print("A", json.dumps({k: v for k, v in report["A_pocket_finding"].items() if k != "rows"}))
    if "taken" not in skip:
        report["B_taken"] = test_taken(usable, rng, a.limit)
        print("B", json.dumps({k: v for k, v in report["B_taken"].items() if k not in ("rows", "_unused")}))
    if "identity" not in skip:
        report["C_identity"] = test_identity(usable, rng, a.refs)
        print("C", json.dumps({k: v for k, v in report["C_identity"].items() if k != "rows"}))
    path = OUT / f"report_{engine.split('+')[0]}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    print("saved", path)


if __name__ == "__main__":
    main()
