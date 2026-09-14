"""Golden-set retrieval evaluation, executed on every scoring run.

Relevance is defined by (source file + required keywords), never by chunk id.
Chunk ids change whenever chunk size, overlap or the loader changes -- which is
exactly when you most need the benchmark to still be valid. Pinning to ids would
mean every retrieval improvement silently invalidates its own evidence.

A sample of the set runs per request (GOLDEN_SAMPLE_SIZE) so the eval adds
bounded latency; `run_full` evaluates everything for offline comparison, e.g.
OpenAI vs Gemini embeddings on identical cases.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import Settings
from ..providers import Embedder
from ..store.vector_store import Hit, VectorStore
from .metrics import RankedEval, aggregate, reciprocal_rank

log = logging.getLogger(__name__)


@dataclass
class GoldenCase:
    id: str
    claim: str
    source_file: str | None = None
    product: str | None = None
    keywords: list[str] | None = None

    def is_relevant(self, hit: Hit) -> bool:
        """A hit counts as relevant if it comes from the expected manual AND
        contains the discriminating keywords. The file check alone is too coarse
        (a 40-slide deck has many irrelevant chunks); keywords alone are too
        loose (the same term appears across product lines)."""
        if self.source_file and self.source_file.lower() not in hit.source_file.lower():
            return False
        if self.keywords:
            body = hit.text.lower()
            if not any(kw.lower() in body for kw in self.keywords):
                return False
        return True


def load_cases(path: Path) -> list[GoldenCase]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = []
    for raw in data.get("cases", []):
        cases.append(GoldenCase(
            id=str(raw.get("id")),
            claim=str(raw.get("claim", "")),
            source_file=raw.get("source_file"),
            product=raw.get("product"),
            keywords=[str(k) for k in (raw.get("keywords") or [])] or None,
        ))
    return [c for c in cases if c.claim]


def evaluate(settings: Settings, embedder: Embedder, store: VectorStore,
             cases: list[GoldenCase], *, k: int | None = None) -> RankedEval:
    k = k or settings.top_k
    if not cases:
        return aggregate([], k)

    queries = [f"{c.product}: {c.claim}" if c.product else c.claim for c in cases]
    vectors = embedder.embed(queries, stage="eval_embed")

    per_case: list[dict[str, Any]] = []
    for case, query, vector in zip(cases, queries, vectors):
        hits = store.search(query, vector, top_k=k, product=case.product)
        relevance = [case.is_relevant(h) for h in hits]
        rr = reciprocal_rank(relevance)
        per_case.append({
            "case_id": case.id,
            "claim": case.claim,
            "hit": bool(any(relevance)),
            "rank": (relevance.index(True) + 1) if any(relevance) else None,
            "reciprocal_rank": rr,
            "expected_source": case.source_file,
            "top_citation": hits[0].citation() if hits else None,
            "top_source_file": hits[0].source_file if hits else None,
        })

    result = aggregate(per_case, k)
    log.info("golden-set retrieval eval",
             extra={"recall_at_k": result.recall_at_k, "mrr": result.mrr,
                    "n_cases": result.n_cases, "k": k})
    return result


def sample_cases(cases: list[GoldenCase], size: int, *, seed: int | None = None
                 ) -> list[GoldenCase]:
    if size <= 0 or size >= len(cases):
        return cases
    rng = random.Random(seed)
    return rng.sample(cases, size)


def run_per_request(settings: Settings, embedder: Embedder, store: VectorStore,
                    *, seed: int | None = None) -> dict[str, Any]:
    cases = load_cases(settings.golden_path)
    if not cases:
        # Say so explicitly rather than reporting a vacuous 0.0 that reads like a
        # failing retriever.
        return {"status": "no_golden_set",
                "note": f"no golden cases found at {settings.golden_path}; "
                        f"retrieval health cannot be reported for this run",
                "recall_at_k": None, "mrr": None, "n_cases": 0}
    sampled = sample_cases(cases, settings.golden_sample_size, seed=seed)
    result = evaluate(settings, embedder, store, sampled).as_dict()
    return {"status": "ok", "sampled_from": len(cases), **result}


def run_full(settings: Settings, embedder: Embedder, store: VectorStore,
             *, k: int | None = None) -> dict[str, Any]:
    cases = load_cases(settings.golden_path)
    if not cases:
        return {"status": "no_golden_set", "n_cases": 0}
    return {"status": "ok", **evaluate(settings, embedder, store, cases, k=k).as_dict()}
