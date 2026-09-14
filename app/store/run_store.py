"""Run persistence.

Two tiers, matching how the data is actually used:
  - full artifact JSON on disk (data/runs/<run_id>.json) -- everything needed to
    audit or replay a verdict, including every retrieved chunk;
  - a queryable row in SQLite -- for trend queries (score drift, retrieval recall
    over time, latency, cost) without parsing hundreds of JSON blobs.

On Cloud Run the disk tier maps to a GCS bucket (RUNS_BUCKET); the interface is
the same, which is why writes go through this one class.
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

    def save(self, result: dict[str, Any]) -> Path:
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
        return artifact

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
