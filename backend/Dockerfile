# PillCheck API. Builds for x86_64 and arm64 (AWS Graviton):
#   docker build -t pillcheck .
#   docker buildx build --platform linux/arm64 -t pillcheck:arm64 .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the OpenCV 5 ALIKED + LightGlue models into the image (checksum-verified).
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py /app/models

COPY pillcheck/ pillcheck/
ENV PILLCHECK_MODELS_DIR=/app/models PILLCHECK_DATA_DIR=/data
RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "pillcheck.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
