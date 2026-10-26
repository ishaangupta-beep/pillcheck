"""Perception unit tests on synthetic strips with exact ground truth."""
import numpy as np
import pytest
from conftest import METF, OTHER, TELMI, jpeg, photo

from pillcheck import synth
from pillcheck.vision import dedupe, quality
from pillcheck.vision.align import resize_long_side
from pillcheck.vision.pipeline import ReferenceBundle, analyze_check, register_reference
from pillcheck.vision.pockets import detect_pockets


@pytest.fixture(scope="module")
def refs():
    out = []
    for mid, spec in (("telmi", TELMI), ("metf", METF)):
        r = register_reference(photo(spec, seed=11, difficulty=0.1))
        out.append(ReferenceBundle(mid, r["image"], r["features"], r["pockets"], r["strip_polygon"]))
    return out


@pytest.mark.parametrize("rows,cols", [(2, 5), (3, 4), (1, 6), (2, 7)])
def test_pocket_detection_counts(rows, cols):
    spec = synth.StripSpec(rows=rows, cols=cols, seed=rows * 10 + cols)
    shot = synth.make_shot(spec, set(), seed=3, difficulty=0.2)
    img, scale = resize_long_side(shot.image, 1280)
    pockets, conf = detect_pockets(img)
    assert len(pockets) == rows * cols and conf > 0.5
    # Reading order matches the strip layout.
    gt = synth.match_ground_truth(spec, shot.H_ref_to_photo, scale, [(p.cx, p.cy) for p in pockets])
    assert gt == {i: i for i in range(rows * cols)}


@pytest.mark.parametrize("seed,difficulty", [(31, 0.4), (32, 0.7), (33, 1.0)])
def test_pocket_states_under_viewpoint_and_lighting(refs, seed, difficulty):
    popped = {0, 3, 7}
    res = analyze_check(photo(TELMI, popped, seed=seed, difficulty=difficulty), refs, "telmi", {}, [])
    assert res["status"] == "ok"
    states = {p["id"]: p["state"] for p in res["pockets"]}
    assert {i for i, s in states.items() if s == "popped"} == popped
    assert sorted(res["newly_popped"]) == sorted(popped)


def test_identity_wrong_and_unknown(refs):
    wrong = analyze_check(photo(METF, {2}, seed=5, difficulty=0.6), refs, "telmi", {}, [])
    assert wrong["status"] == "wrong_medication" and wrong["medication_id"] == "metf"
    unknown = analyze_check(photo(OTHER, seed=6), refs, "telmi", {}, [])
    assert unknown["status"] == "unrecognised"


def test_quality_gates():
    blurry = synth.make_shot(TELMI, set(), seed=7, blur_sigma=5).image
    assert "blurry" in quality.assess(blurry).issues
    dark = (synth.make_shot(TELMI, set(), seed=8).image * 0.15).astype(np.uint8)
    assert "too_dark" in quality.assess(dark).issues


def test_duplicate_survives_recompression():
    img = synth.make_shot(TELMI, {1}, seed=9).image
    a, b = jpeg(img), __import__("cv2").imencode(".jpg", img, [1, 60])[1].tobytes()
    fa = dedupe.fingerprint(a, img)
    fb = dedupe.fingerprint(b, __import__("cv2").imdecode(np.frombuffer(b, np.uint8), 1))
    assert fa["sha256"] != fb["sha256"]
    assert dedupe.find_duplicate(fb, [{"check_id": "c1", **fa}])["check_id"] == "c1"
    other = synth.make_shot(TELMI, {1}, seed=10).image  # a genuinely new photo of the same strip
    fo = dedupe.fingerprint(jpeg(other), other)
    assert dedupe.find_duplicate(fo, [{"check_id": "c1", **fa}]) is None
