import os
import sys
import tempfile
from pathlib import Path

# Isolated data directory and rules-only agent for every test session.
os.environ["PILLCHECK_DATA_DIR"] = tempfile.mkdtemp(prefix="pillcheck-test-")
os.environ["PILLCHECK_STORAGE"] = "local"
os.environ["PILLCHECK_BEDROCK_MODEL_ID"] = ""
os.environ.pop("PILLCHECK_API_TOKEN", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2 as cv  # noqa: E402
import pytest  # noqa: E402

from pillcheck import synth  # noqa: E402

TELMI = synth.StripSpec(name="TELMISARTAN 40mg", seed=1)
METF = synth.StripSpec(name="METFORMIN 500mg", rows=3, cols=4, seed=7, pill_colour=(150, 210, 160), ink_colour=(30, 120, 40))
OTHER = synth.StripSpec(name="ATORVASTATIN 10", seed=33, rows=1, cols=6)


def jpeg(img) -> bytes:
    return cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def photo(spec, popped=(), seed=0, **kw) -> bytes:
    return jpeg(synth.make_shot(spec, set(popped), seed=seed, **kw).image)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from pillcheck.api.main import app
    return TestClient(app)
