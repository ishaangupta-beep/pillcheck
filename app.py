"""Hugging Face Space entry point (Gradio SDK, free CPU).

The Space runs `python app.py`; we serve the full PillCheck app (React UI +
FastAPI + OpenCV 5) on the Space's port instead of a Gradio UI.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))
os.environ.setdefault("PILLCHECK_MODELS_DIR", "/tmp/pillcheck-models")
os.environ.setdefault("PILLCHECK_DATA_DIR", "/tmp/pillcheck")

# ALIKED, LightGlue, PP-OCRv3, CRNN: downloaded once, checksum-verified.
subprocess.run([sys.executable, os.path.join(ROOT, "backend/scripts/download_models.py"),
                os.environ["PILLCHECK_MODELS_DIR"]], check=True)

import uvicorn  # noqa: E402

uvicorn.run("pillcheck.api.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
