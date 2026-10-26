"""Evaluation harness for the technical report.

    python eval/run_eval.py                      # synthetic benchmark, ALIKED+LightGlue
    PILLCHECK_FEATURES=orb python eval/run_eval.py --tag orb   # classic ORB baseline
    python eval/run_eval.py --real eval/real     # your labelled real photos

Synthetic: 6 strip designs (layouts 1x6 to 3x4; two share one layout and differ
only in the printed name) photographed under three difficulty levels plus glare
and mild blur. Every photo has exact ground truth.

Real data layout (pocket numbers as shown in try_photos.py's reference image):
    eval/real/<strip>/reference.jpg
    eval/real/<strip>/<photo>.jpg ...
    eval/real/<strip>/labels.json   {"<photo>.jpg": {"popped": [1, 2]}}

Metrics: registration success, per-pocket accuracy / precision / recall on
decided pockets, abstention rate (uncertain + unreadable), identity accuracy
(right strip accepted, other registered strip named, unregistered rejected),
latency. Results go to eval/results/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pillcheck import synth  # noqa: E402
from pillcheck.vision.align import resize_long_side  # noqa: E402
from pillcheck.vision.models import get_engine  # noqa: E402
from pillcheck.vision.pipeline import ReferenceBundle, analyze_check, register_reference  # noqa: E402

NAMES = ["TELMISARTAN 40mg", "METFORMIN 500mg", "AMLODIPINE 5mg", "ECOSPRIN 75", "PAN 40", "ATORVASTATIN 10"]
LAYOUTS = [(2, 5), (3, 4), (1, 6), (2, 7), (2, 4), (1, 6)]
INKS = [(40, 60, 150), (30, 120, 40), (150, 60, 40), (40, 40, 40), (120, 40, 120), (60, 100, 160)]
CONDITIONS = [("easy", dict(difficulty=0.3)), ("medium", dict(difficulty=0.6)), ("hard", dict(difficulty=0.9)),
              ("glare", dict(difficulty=0.6, glare=True)), ("soft_blur", dict(difficulty=0.6, blur_sigma=1.2))]


def jpeg(img) -> bytes:
    return cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


class Tally:
    def __init__(self):
        self.tp = self.fp = self.tn = self.fn = self.abstain = 0
        self.checks = self.registered = 0
        self.latency = []

    def add_pockets(self, states: dict[int, str], popped_truth: set[int]):
        for pid, st in states.items():
            truth = pid in popped_truth
            if st in ("uncertain", "unknown"):
                self.abstain += 1
            elif st == "popped":
                self.tp += truth
                self.fp += not truth
            else:
                self.fn += truth
                self.tn += not truth

    def summary(self) -> dict:
        decided = self.tp + self.fp + self.tn + self.fn
        total = decided + self.abstain
        return {
            "checks": self.checks,
            "registration_success": round(self.registered / max(1, self.checks), 4),
            "pockets": total,
            "pocket_accuracy_on_decided": round((self.tp + self.tn) / max(1, decided), 4),
            "popped_precision": round(self.tp / max(1, self.tp + self.fp), 4),
            "popped_recall": round(self.tp / max(1, self.tp + self.fn), 4),
            "abstention_rate": round(self.abstain / max(1, total), 4),
            "false_popped": self.fp, "missed_popped": self.fn,
            "latency_ms_p50": int(np.percentile(self.latency, 50)) if self.latency else None,
            "latency_ms_p95": int(np.percentile(self.latency, 95)) if self.latency else None,
        }


def synthetic(shots_per_condition: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    specs = [synth.StripSpec(name=NAMES[i], rows=LAYOUTS[i][0], cols=LAYOUTS[i][1], seed=100 + i,
                             pill_colour=synth.PILL_COLOURS[i % 5], ink_colour=INKS[i]) for i in range(6)]
    registered, unregistered = specs[:5], specs[5]
    bundles, gt_maps = [], []
    for i, spec in enumerate(registered):
        shot = synth.make_shot(spec, set(), seed=seed * 1000 + i, difficulty=0.15)
        reg = register_reference(jpeg(shot.image))
        _, scale = resize_long_side(shot.image, 1280)
        if len(reg["pockets"]) != spec.rows * spec.cols:
            raise SystemExit(f"pocket detection failed on {spec.name}: {len(reg['pockets'])} found")
        gt_maps.append(synth.match_ground_truth(spec, shot.H_ref_to_photo, scale, [(p.cx, p.cy) for p in reg["pockets"]]))
        bundles.append(ReferenceBundle(str(i), reg["image"], reg["features"], reg["pockets"], reg["strip_polygon"]))

    by_condition = {name: Tally() for name, _ in CONDITIONS}
    identity = {"right_accepted": 0, "right_total": 0, "wrong_named": 0, "wrong_total": 0, "unregistered_rejected": 0, "unregistered_total": 0}
    for cname, kw in CONDITIONS:
        t = by_condition[cname]
        for i, spec in enumerate(registered):
            n = spec.rows * spec.cols
            for k in range(shots_per_condition):
                popped = set(rng.choice(n, size=int(rng.integers(0, n)), replace=False).tolist())
                shot = synth.make_shot(spec, popped, seed=int(rng.integers(1 << 30)), **kw)
                t0 = time.perf_counter()
                res = analyze_check(jpeg(shot.image), bundles, str(i), {}, [])
                t.latency.append(1000 * (time.perf_counter() - t0))
                t.checks += 1
                identity["right_total"] += 1
                if res.get("medication_id") == str(i):
                    identity["right_accepted"] += 1
                if "pockets" in res and res.get("medication_id") == str(i):
                    t.registered += 1
                    truth = {gt_maps[i][j] for j in popped}
                    t.add_pockets({p["id"]: p["state"] for p in res["pockets"]}, truth)
                else:  # a failed check abstains on every pocket
                    t.abstain += n

        # Identity probes: photograph strip j while strip i is expected; and an unregistered strip.
        for i, spec in enumerate(registered):
            j = (i + 1) % len(registered)
            res = analyze_check(jpeg(synth.make_shot(spec, set(), seed=int(rng.integers(1 << 30)), **kw).image), bundles, str(j), {}, [])
            identity["wrong_total"] += 1
            identity["wrong_named"] += res["status"] == "wrong_medication" and res.get("medication_id") == str(i)
        res = analyze_check(jpeg(synth.make_shot(unregistered, set(), seed=int(rng.integers(1 << 30)), **kw).image), bundles, "0", {}, [])
        identity["unregistered_total"] += 1
        identity["unregistered_rejected"] += res["status"] == "unrecognised"

    overall = Tally()
    for t in by_condition.values():
        for f in ("tp", "fp", "tn", "fn", "abstain", "checks", "registered"):
            setattr(overall, f, getattr(overall, f) + getattr(t, f))
        overall.latency += t.latency
    return {"conditions": {k: v.summary() for k, v in by_condition.items()}, "overall": overall.summary(), "identity": identity}


def real(root: Path) -> dict:
    t = Tally()
    per_strip = {}
    for strip in sorted(p for p in root.iterdir() if p.is_dir()):
        labels = json.loads((strip / "labels.json").read_text())
        reg = register_reference((strip / "reference.jpg").read_bytes())
        bundle = ReferenceBundle(strip.name, reg["image"], reg["features"], reg["pockets"], reg["strip_polygon"])
        st = Tally()
        for name, lab in labels.items():
            t0 = time.perf_counter()
            res = analyze_check((strip / name).read_bytes(), [bundle], strip.name, {}, [])
            ms = 1000 * (time.perf_counter() - t0)
            truth = {x - 1 for x in lab.get("popped", [])}
            for tt in (t, st):
                tt.latency.append(ms)
                tt.checks += 1
                if "pockets" in res:
                    tt.registered += 1
                    tt.add_pockets({p["id"]: p["state"] for p in res["pockets"]}, truth)
                else:
                    tt.abstain += len(reg["pockets"])
        per_strip[strip.name] = {"pockets_detected": len(reg["pockets"]), **st.summary()}
    return {"overall": t.summary(), "strips": per_strip}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=6, help="photos per strip per condition")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--real", type=Path, help="folder of labelled real photos")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    engine = get_engine().name
    tag = args.tag or engine.replace("+", "_")
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    result = {"engine": engine, "opencv": cv.__version__}
    if args.real:
        result["real"] = real(args.real)
    else:
        result["synthetic"] = synthetic(args.shots, args.seed)
    result["wall_seconds"] = round(time.time() - t0, 1)
    path = out_dir / f"{tag}_{'real' if args.real else 'synthetic'}.json"
    path.write_text(json.dumps(result, indent=2))

    print(f"\nEngine: {engine}   OpenCV {cv.__version__}   ({result['wall_seconds']} s)")
    if "synthetic" in result:
        s = result["synthetic"]
        print(f"{'condition':<10} {'checks':>6} {'regist.':>8} {'acc':>7} {'prec':>7} {'recall':>7} {'abstain':>8} {'p50 ms':>7}")
        for name, m in list(s["conditions"].items()) + [("OVERALL", s["overall"])]:
            print(f"{name:<10} {m['checks']:>6} {m['registration_success']:>8.1%} {m['pocket_accuracy_on_decided']:>7.1%} "
                  f"{m['popped_precision']:>7.1%} {m['popped_recall']:>7.1%} {m['abstention_rate']:>8.1%} {m['latency_ms_p50']:>7}")
        i = s["identity"]
        print(f"identity: right strip accepted {i['right_accepted']}/{i['right_total']}, other registered strip named "
              f"{i['wrong_named']}/{i['wrong_total']}, unregistered rejected {i['unregistered_rejected']}/{i['unregistered_total']}")
    else:
        print(json.dumps(result["real"], indent=2))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
