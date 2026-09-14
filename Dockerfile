# Single image serving both the API and the ingestion job. Cloud Run Jobs and
# Cloud Run Services run the same container with a different entrypoint, which
# keeps the loader/chunker/embedder identical between ingest and query time --
# a mismatch there silently destroys retrieval accuracy while tests still pass.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-cloud.txt

COPY app/ ./app/
COPY frontend/ ./frontend/

# The embedded corpus ships IN the image, not in a volume or a database.
#
# It is read-only at query time (ingestion is a separate batch job), so baking it
# in makes the deployed unit immutable and self-contained: cold starts are
# instant, there is no bucket to read on startup, and the corpus version is
# pinned to the image version. Updating manuals means re-ingesting and
# redeploying, which is the honest description of what actually changed --
# the service's knowledge is part of the artifact, not config it picks up later.
#
# Build fails loudly if the store is missing; a silently empty corpus would
# deploy fine and mark every claim unverifiable.
COPY data/store.sqlite3 ./data/store.sqlite3
COPY data/eval/golden.yaml ./data/eval/golden.yaml
RUN test -s ./data/store.sqlite3 || (echo "ERROR: data/store.sqlite3 missing or empty -- run 'make ingest' before building" && exit 1)

ENV PORT=8080 \
    DATA_DIR=/app/data \
    DB_PATH=/app/data/store.sqlite3 \
    RUNS_BACKEND=cloud

EXPOSE 8080
CMD exec uvicorn app.api:app --host 0.0.0.0 --port ${PORT} --workers 1
