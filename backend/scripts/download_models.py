"""Download the ONNX models PillCheck runs with OpenCV 5's DNN engine, and verify checksums.

ALIKED + LightGlue: the exact files OpenCV's own test-suite uses
(opencv_extra/testdata/dnn/download_models.py).
PP-OCRv3 text detector + CRNN recogniser: from the OpenCV model zoo (opencv_zoo),
used to read the expiry date printed on a strip.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/"
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {  # name: (url, sha1)
    "aliked-n16rot-top1k-640.onnx": (BASE + "aliked-n16rot-top1k-640.onnx", "41faa7bf5d7eb68a2851471ba03aa20c9db30e4c"),
    "aliked_lightglue.onnx": (BASE + "aliked_lightglue.onnx", "02723aa521990e57fe33d90b67977590c460e351"),
    "text_detection_en_ppocrv3_2023may.onnx": (ZOO + "text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
                                               "5f737e11427f01dfba9f645aa197afe5558e721d"),
    "text_recognition_CRNN_EN_2021sep.onnx": (ZOO + "text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx",
                                              "dc8c70a52c6880f11859bf074bcd294a45860821"),
}


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "models"
    dest.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in MODELS.items():
        path = dest / name
        if path.exists() and sha1(path) == digest:
            print(f"ok       {name}")
            continue
        print(f"download {name} ...", flush=True)
        urllib.request.urlretrieve(url, path)
        if sha1(path) != digest:
            print(f"checksum mismatch for {name}", file=sys.stderr)
            return 1
        print(f"ok       {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
