# PillCheck on Hugging Face Spaces (Docker SDK): web app + API + OpenCV 5 in one container.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/scripts/download_models.py scripts/
RUN python scripts/download_models.py /app/models
COPY backend/pillcheck/ pillcheck/
COPY backend/demo_assets/ demo_assets/
COPY --from=web /web/dist/ static/
ENV PILLCHECK_MODELS_DIR=/app/models PILLCHECK_DATA_DIR=/tmp/pillcheck
RUN chown -R user /app
USER user
EXPOSE 7860
CMD ["uvicorn", "pillcheck.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
