# Single image serving both the API and the ingestion job. Cloud Run Jobs and
# Cloud Run Services run the same container with a different entrypoint, which
# keeps the loader/chunker/embedder identical between ingest and query time --
# a mismatch there silently destroys retrieval accuracy.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# pdfplumber needs libmagic-adjacent tooling for some PDFs; kept minimal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY frontend/ ./frontend/

# Cloud Run injects $PORT; default matches local `uvicorn` for parity.
ENV PORT=8080 DATA_DIR=/data DB_PATH=/data/store.sqlite3
RUN mkdir -p /data

EXPOSE 8080
CMD exec uvicorn app.api:app --host 0.0.0.0 --port ${PORT} --workers 1
