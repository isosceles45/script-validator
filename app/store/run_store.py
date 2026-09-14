"""Run persistence.

Two tiers, matching how the data is actually used:
  - full artifact JSON on disk (data/runs/<run_id>.json) -- everything needed to
    audit or replay a verdict, including every retrieved chunk;
  - a queryable row in SQLite -- for trend queries (score drift, retrieval recall
    over time, latency, cost) without parsing hundreds of JSON blobs.

On Cloud Run the disk tier maps to a GCS bucket and the SQLite tier to
Firestore -- see CloudRunStore below. Both implement the same four methods, so
the scoring pipeline never knows which one it is writing to.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    campaign        TEXT,
    product_hint    TEXT,
    provider        TEXT NOT NULL,
    llm_model       TEXT NOT NULL,
    embed_model     TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    overall_score   REAL,
    brief_score     REAL,
    message_score   REAL,
    claim_score     REAL,
    n_claims        INTEGER,
    n_supported     INTEGER,
    n_contradicted  INTEGER,
    n_unverifiable  INTEGER,
    grounding_rate  REAL,
    recall_at_k     REAL,
    mrr             REAL,
    latency_ms      INTEGER,
    total_tokens    INTEGER,
    artifact_path   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
"""


class RunStore:
    def __init__(self, db_path: Path, runs_dir: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = runs_dir
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def save(self, result: dict[str, Any]) -> str:
        run_id = result["run_id"]
        artifact = self.runs_dir / f"{run_id}.json"
        artifact.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

        scores = result.get("scores", {})
        claims = result.get("claim_validity", {}).get("tally", {})
        ev = result.get("retrieval_eval", {})
        meta = result.get("meta", {})
        self._conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, result.get("created_at", time.time()), result.get("campaign"),
             result.get("product_hint"), meta.get("provider", ""),
             meta.get("llm_model", ""), meta.get("embed_model", ""),
             meta.get("prompt_version", ""),
             scores.get("overall"), scores.get("brief_alignment"),
             scores.get("message_quality"), scores.get("claim_validity"),
             claims.get("total"), claims.get("supported"), claims.get("contradicted"),
             claims.get("unverifiable"),
             ev.get("live", {}).get("grounding_rate"),
             ev.get("golden", {}).get("recall_at_k"),
             ev.get("golden", {}).get("mrr"),
             meta.get("latency_ms"), meta.get("total_tokens"), str(artifact)))
        self._conn.commit()
        log.info("run persisted", extra={"artifact": str(artifact)})
        return str(artifact)

    def get(self, run_id: str) -> dict[str, Any] | None:
        artifact = self.runs_dir / f"{run_id}.json"
        if not artifact.exists():
            return None
        return json.loads(artifact.read_text(encoding="utf-8"))

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def trends(self) -> dict[str, Any]:
        """Aggregates backing the observability view: is scoring drifting, is
        retrieval degrading, what does a run cost."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, AVG(overall_score) AS avg_overall, "
            "AVG(brief_score) AS avg_brief, AVG(message_score) AS avg_message, "
            "AVG(claim_score) AS avg_claim, AVG(grounding_rate) AS avg_grounding, "
            "AVG(recall_at_k) AS avg_recall, AVG(mrr) AS avg_mrr, "
            "AVG(latency_ms) AS avg_latency_ms, AVG(total_tokens) AS avg_tokens "
            "FROM runs").fetchone()
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in dict(row).items()}

    def close(self) -> None:
        self._conn.close()


class CloudRunStore:
    """GCS for run artifacts, Firestore for the queryable index.

    Cloud Run instances are ephemeral and concurrent, so local disk loses every
    run on cold start and splits /metrics across instances. Both backends here
    scale to zero, which keeps the cost profile of the deployment intact.

    Deliberately mirrors RunStore's interface rather than sharing a base class:
    the two have nothing in common internally, and an abstract base would only
    add indirection between the pipeline and a dict write.
    """

    def __init__(self, *, project: str, bucket: str, collection: str) -> None:
        from google.cloud import firestore, storage

        if not bucket:
            raise ValueError("RUNS_BACKEND=cloud requires RUNS_BUCKET")
        self._bucket = storage.Client(project=project or None).bucket(bucket)
        self._collection = firestore.Client(project=project or None).collection(collection)
        self.bucket_name = bucket
        log.info("cloud run store ready",
                 extra={"bucket": bucket, "collection": collection})

    def save(self, result: dict[str, Any]) -> str:
        run_id = result["run_id"]
        path = f"runs/{run_id}.json"
        self._bucket.blob(path).upload_from_string(
            json.dumps(result, indent=2, default=str), content_type="application/json")

        scores = result.get("scores", {})
        claims = result.get("claim_validity", {}).get("tally", {})
        ev = result.get("retrieval_eval", {})
        meta = result.get("meta", {})
        # Index row only -- the full artifact stays in GCS. Firestore documents
        # cap at 1MB and a run with every retrieved chunk can exceed that.
        self._collection.document(run_id).set({
            "run_id": run_id,
            "created_at": result.get("created_at", time.time()),
            "campaign": result.get("campaign"),
            "product_hint": result.get("product_hint"),
            "verdict": result.get("verdict"),
            "provider": meta.get("provider"),
            "llm_model": meta.get("llm_model"),
            "embed_model": meta.get("embed_model"),
            "prompt_version": meta.get("prompt_version"),
            "overall_score": scores.get("overall"),
            "brief_score": scores.get("brief_alignment"),
            "message_score": scores.get("message_quality"),
            "claim_score": scores.get("claim_validity"),
            "n_claims": claims.get("total"),
            "n_supported": claims.get("supported"),
            "n_contradicted": claims.get("contradicted"),
            "n_unverifiable": claims.get("unverifiable"),
            "grounding_rate": ev.get("live", {}).get("grounding_rate"),
            "recall_at_k": ev.get("golden", {}).get("recall_at_k"),
            "mrr": ev.get("golden", {}).get("mrr"),
            "latency_ms": meta.get("latency_ms"),
            "total_tokens": meta.get("total_tokens"),
            "artifact_path": f"gs://{self.bucket_name}/{path}",
        })
        uri = f"gs://{self.bucket_name}/{path}"
        log.info("run persisted", extra={"artifact": uri})
        return uri

    def get(self, run_id: str) -> dict[str, Any] | None:
        blob = self._bucket.blob(f"runs/{run_id}.json")
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        from google.cloud.firestore import Query
        docs = (self._collection.order_by("created_at", direction=Query.DESCENDING)
                .limit(limit).stream())
        return [d.to_dict() for d in docs]

    def trends(self) -> dict[str, Any]:
        """Firestore has no AVG aggregation over an unindexed scan, so the
        averages are computed over the most recent window rather than the whole
        collection. Bounded cost, and a trend over the last 500 runs is the more
        useful number anyway."""
        from google.cloud.firestore import Query
        rows = [d.to_dict() for d in
                (self._collection.order_by("created_at", direction=Query.DESCENDING)
                 .limit(500).stream())]
        if not rows:
            return {"n": 0, "window": 500}

        def mean(field: str) -> float | None:
            values = [r[field] for r in rows
                      if isinstance(r.get(field), (int, float))]
            return round(sum(values) / len(values), 4) if values else None

        return {"n": len(rows), "window": 500,
                "avg_overall": mean("overall_score"), "avg_brief": mean("brief_score"),
                "avg_message": mean("message_score"), "avg_claim": mean("claim_score"),
                "avg_grounding": mean("grounding_rate"), "avg_recall": mean("recall_at_k"),
                "avg_mrr": mean("mrr"), "avg_latency_ms": mean("latency_ms"),
                "avg_tokens": mean("total_tokens")}

    def close(self) -> None:
        return None


def build_run_store(settings) -> "RunStore | CloudRunStore":
    if settings.runs_backend == "cloud":
        return CloudRunStore(project=settings.gcp_project,
                             bucket=settings.runs_bucket,
                             collection=settings.firestore_collection)
    if settings.runs_backend != "local":
        raise ValueError(f"unknown RUNS_BACKEND={settings.runs_backend!r} "
                         f"(expected 'local' or 'cloud')")
    return RunStore(settings.db_path, settings.runs_dir)
