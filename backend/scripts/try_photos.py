"""Try the vision pipeline on your own photos, no server needed.

    python scripts/try_photos.py full_strip.jpg after_1.jpg after_2.jpg

The first photo must show the FULL strip (every pill still in). Each later
photo is compared with the reference and with the photo before it, exactly as
the app does day by day. Images are written to ./try_output/:
    reference_pockets.jpg  - detected pockets and strip outline (check these!)
    check_1_evidence.jpg   - reference vs aligned photo, pockets coloured
Tips: photograph the printed (foil) side or the pill side consistently, from
roughly above, the whole strip in frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pillcheck.vision.pipeline import ReferenceBundle, analyze_check, register_reference  # noqa: E402

STATE_MARK = {"intact": "●", "popped": "○", "uncertain": "?", "unknown": "·"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path, help="photo of the full strip")
    ap.add_argument("photos", type=Path, nargs="*", help="later photos, in order")
    ap.add_argument("--out", type=Path, default=Path("try_output"))
    ap.add_argument("--json", action="store_true", help="also dump full results as JSON")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reg = register_reference(args.reference.read_bytes())
    (args.out / "reference_pockets.jpg").write_bytes(reg["preview_jpeg"])
    print(f"\nReference: {args.reference.name}")
    print(f"  engine={reg['engine']}  features={reg['feature_count']}  pockets={len(reg['pockets'])} "
          f"(confidence {reg['pocket_confidence']})  quality={reg['quality']['issues'] or 'ok'}  {reg['elapsed_ms']} ms")
    print(f"  -> check {args.out / 'reference_pockets.jpg'}: every pocket should have one ellipse.")
    if not reg["pockets"]:
        print("  No pockets found. Retake the reference closer, from straight above, on a plain surface.")
        return 1

    bundle = ReferenceBundle("strip", reg["image"], reg["features"], reg["pockets"], reg["strip_polygon"])
    prev_states: dict[int, str] = {}
    fingerprints: list[dict] = []
    results = []
    for i, path in enumerate(args.photos, 1):
        res = analyze_check(path.read_bytes(), [bundle], "strip", {"strip": prev_states}, fingerprints)
        evidence = res.pop("_evidence_jpeg", None)
        fp = res.pop("fingerprint", None)
        if fp:
            fingerprints.append({"check_id": f"photo_{i}", **fp})
        print(f"\nPhoto {i}: {path.name}  ->  {res['status'].upper()}   ({sum(res['timings_ms'].values())} ms)")
        if res.get("registration"):
            r = res["registration"]
            print(f"  alignment: {r['inliers']} inliers / {r['matches']} matches, rmse {r['reproj_rmse_px']} px, "
                  f"identity score {res['identity']['appearance']}")
        if "pockets" in res:
            row = " ".join(f"{p['id'] + 1}{STATE_MARK[p['state']]}" for p in res["pockets"])
            print(f"  pockets: {row}      (● intact  ○ popped  ? unsure  · unreadable)")
            print(f"  scores:  " + " ".join(f"{p['id'] + 1}:{p['score']}" for p in res["pockets"]))
            print(f"  newly removed since previous photo: {[x + 1 for x in res['newly_popped']] or 'none'}")
            if res["inconsistent"]:
                print(f"  pockets that look full again: {[x + 1 for x in res['inconsistent']]}")
            for p in res["pockets"]:
                if p["state"] in ("intact", "popped"):
                    prev_states[p["id"]] = p["state"]
        for g in res.get("guidance", []):
            print(f"  guidance: {g}")
        if evidence:
            out = args.out / f"check_{i}_evidence.jpg"
            out.write_bytes(evidence)
            print(f"  evidence: {out}")
        results.append({"photo": path.name, **res})

    if args.json:
        (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
