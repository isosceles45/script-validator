"""SQLite + numpy vector store with hybrid (dense + lexical) retrieval.

Why SQLite rather than pgvector for the default path: manual corpora are in the
thousands-of-chunks range, where a brute-force numpy dot product over float32 is
sub-millisecond and exact -- an ANN index would add ops burden and *approximate*
recall for no gain. The interface below is deliberately the pgvector interface
(upsert / search / filter by product), so swapping the backend is one class.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..ingest.chunker import Chunk

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS manuals (
    manual_id    TEXT PRIMARY KEY,
    source_file  TEXT NOT NULL,
    checksum     TEXT NOT NULL,
    product      TEXT NOT NULL,
    n_chunks     INTEGER NOT NULL,
    embed_model  TEXT NOT NULL,
    ingested_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    manual_id   TEXT NOT NULL REFERENCES manuals(manual_id) ON DELETE CASCADE,
    product     TEXT NOT NULL,
    source_file TEXT NOT NULL,
    section     TEXT,
    page        INTEGER,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_manual ON chunks(manual_id);
CREATE INDEX IF NOT EXISTS idx_chunks_product ON chunks(product);
"""

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    chunk_id: str
    product: str
    section: str | None
    page: int | None
    kind: str
    text: str
    source_file: str
    score: float          # fused rank score, comparable only within one query
    dense_score: float    # raw cosine similarity, comparable across queries
    lexical_score: float

    def citation(self) -> str:
        parts = [self.product]
        if self.section:
            parts.append(self.section)
        if self.page is not None:
            parts.append(f"p.{self.page}")
        return " > ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "citation": self.citation(),
                "product": self.product, "section": self.section, "page": self.page,
                "kind": self.kind, "source_file": self.source_file,
                "score": round(self.score, 4),
                "dense_score": round(self.dense_score, 4),
                "lexical_score": round(self.lexical_score, 4),
                "text": self.text}


class VectorStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._cache: tuple[int, list[sqlite3.Row], np.ndarray, list[list[str]]] | None = None

    # ---------- ingestion side ----------

    def existing_checksum(self, manual_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT checksum FROM manuals WHERE manual_id = ?", (manual_id,)).fetchone()
        return row["checksum"] if row else None

    def delete_manual(self, manual_id: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE manual_id = ?", (manual_id,))
        self._conn.execute("DELETE FROM manuals WHERE manual_id = ?", (manual_id,))
        self._conn.commit()
        self._cache = None

    def upsert_manual(self, *, manual_id: str, source_file: str, checksum: str,
                      product: str, chunks: list[Chunk], embeddings: list[list[float]],
                      embed_model: str) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunk/embedding count mismatch")
        # Replace wholesale: a re-ingested manual is a new version, and leaving
        # stale chunks behind would let a retired claim stay "supported".
        self.delete_manual(manual_id)
        self._conn.execute(
            "INSERT INTO manuals VALUES (?,?,?,?,?,?,?)",
            (manual_id, source_file, checksum, product, len(chunks), embed_model, time.time()))
        self._conn.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
            [(c.chunk_id, c.manual_id, c.product, c.source_file, c.section, c.page,
              c.kind, c.text, np.asarray(e, dtype=np.float32).tobytes())
             for c, e in zip(chunks, embeddings)])
        self._conn.commit()
        self._cache = None

    # ---------- retrieval side ----------

    def stats(self) -> dict[str, Any]:
        manuals = self._conn.execute(
            "SELECT manual_id, product, source_file, n_chunks, embed_model, ingested_at "
            "FROM manuals ORDER BY product").fetchall()
        n_chunks = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return {"n_manuals": len(manuals), "n_chunks": n_chunks,
                "manuals": [dict(m) for m in manuals]}

    def products(self) -> list[str]:
        return [r["product"] for r in
                self._conn.execute("SELECT DISTINCT product FROM chunks ORDER BY product")]

    def _load(self):
        """Cache the corpus in memory, invalidated by row count. Manuals change on
        ingestion, not per request, so this is loaded once per process lifetime."""
        count = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        if self._cache and self._cache[0] == count:
            return self._cache
        rows = self._conn.execute(
            "SELECT chunk_id, product, section, page, kind, text, source_file, embedding "
            "FROM chunks ORDER BY chunk_id").fetchall()
        if not rows:
            empty = (0, [], np.zeros((0, 0), dtype=np.float32), [])
            self._cache = empty
            return empty
        matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-9, None)  # pre-normalised => cosine is a dot
        tokens = [tokenize(r["text"]) for r in rows]
        self._cache = (count, rows, matrix, tokens)
        return self._cache

    def _bm25(self, query: str, token_docs: list[list[str]],
              candidate_idx: np.ndarray) -> np.ndarray:
        """BM25 over the candidate set. The lexical half of the hybrid exists for
        exact-token claims -- "2% salicylic acid", an SKU code, a temperature --
        where dense similarity alone happily returns a topically-adjacent chunk
        with the wrong number in it."""
        k1, b = 1.5, 0.75
        q_terms = set(tokenize(query))
        if not q_terms or candidate_idx.size == 0:
            return np.zeros(candidate_idx.size, dtype=np.float32)
        docs = [token_docs[i] for i in candidate_idx]
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        avgdl = float(lengths.mean()) or 1.0
        n_docs = len(docs)
        counters = [Counter(d) for d in docs]
        scores = np.zeros(n_docs, dtype=np.float32)
        for term in q_terms:
            df = sum(1 for c in counters if term in c)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            tf = np.array([c.get(term, 0) for c in counters], dtype=np.float32)
            denom = tf + k1 * (1 - b + b * lengths / avgdl)
            scores += idf * (tf * (k1 + 1)) / np.clip(denom, 1e-9, None)
        return scores

    def search(self, query: str, query_embedding: list[float], *, top_k: int = 5,
               product: str | None = None) -> list[Hit]:
        count, rows, matrix, token_docs = self._load()
        if count == 0:
            return []

        if product:
            wanted = product.strip().lower()
            candidate_idx = np.array(
                [i for i, r in enumerate(rows)
                 if wanted in r["product"].lower() or r["product"].lower() in wanted],
                dtype=np.int64)
            # A product hint that matches nothing is a hint, not a constraint:
            # fall back to the full corpus rather than returning zero evidence.
            if candidate_idx.size == 0:
                log.info("product filter matched no chunks; searching full corpus",
                         extra={"product_hint": product})
                candidate_idx = np.arange(len(rows), dtype=np.int64)
        else:
            candidate_idx = np.arange(len(rows), dtype=np.int64)

        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        dense = matrix[candidate_idx] @ q
        lexical = self._bm25(query, token_docs, candidate_idx)

        # Reciprocal rank fusion: combines two incommensurable score scales
        # without tuning a weight, and is robust to one side being flat.
        def rrf(scores: np.ndarray) -> np.ndarray:
            order = np.argsort(-scores)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(scores))
            return 1.0 / (60.0 + ranks + 1)

        fused = rrf(dense) + rrf(lexical)
        top = np.argsort(-fused)[:top_k]

        hits: list[Hit] = []
        for local in top:
            row = rows[candidate_idx[local]]
            hits.append(Hit(
                chunk_id=row["chunk_id"], product=row["product"], section=row["section"],
                page=row["page"], kind=row["kind"], text=row["text"],
                source_file=row["source_file"], score=float(fused[local]),
                dense_score=float(dense[local]), lexical_score=float(lexical[local])))
        return hits

    def close(self) -> None:
        self._conn.close()
