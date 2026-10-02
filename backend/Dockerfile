# Backend image. Build from the repository root, because the backend imports the training
# corruption code (src/data/corruptions.py, pure NumPy):
#   docker build -f backend/Dockerfile -t genai-backend .
#   docker run -p 8000:8000 -v "$PWD/models:/models:ro" genai-backend
# The ONNX files and their sidecar JSONs are mounted at /models (MODEL_DIR); missing models do
# not stop the server, /api/health lists them. backend/Dockerfile.dockerignore limits the build
# context to the files copied below.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/models

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Single source of truth for corruptions, shared with training; no other training code is needed.
COPY src/__init__.py src/__init__.py
COPY src/data/__init__.py src/data/__init__.py
COPY src/data/corruptions.py src/data/corruptions.py
COPY backend/app ./app

RUN useradd --create-home --uid 10001 appuser && mkdir -p /models && chown appuser /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
