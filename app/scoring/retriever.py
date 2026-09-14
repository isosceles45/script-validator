"""Stage 2: per-claim retrieval.

Retrieval is run once per claim, not once per script. A single embedding of a
whole script is dominated by its narrative and tone; the specific assertion
("contains 5% niacinamide") contributes a small fraction of the vector, so the
chunk that actually adjudicates it frequently falls outside top-k. Per-claim
retrieval is the single biggest lever on claim-validity accuracy in this system.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..providers import Embedder
from ..store.vector_store import Hit, VectorStore
from .claims import Claim

log = logging.getLogger(__name__)


@dataclass
class Retrieved:
    claim: Claim
    hits: list[Hit]

    @property
    def best_dense(self) -> float:
        return max((h.dense_score for h in self.hits), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {"claim_id": self.claim.id,
                "best_dense_score": round(self.best_dense, 4),
                "hits": [h.as_dict() for h in self.hits]}


def _query_text(claim: Claim, product_hint: str | None) -> str:
    """The claim plus its product context. The product token matters: "reduces
    redness in 7 days" is ambiguous across a 40-deck corpus without it."""
    product = claim.product or product_hint
    return f"{product}: {claim.text}" if product else claim.text


def retrieve_for_claims(settings: Settings, embedder: Embedder, store: VectorStore,
                        claims: list[Claim], *,
                        product_hint: str | None = None) -> list[Retrieved]:
    if not claims:
        return []

    queries = [_query_text(c, product_hint) for c in claims]
    # One batched embedding call for all claims -- N round-trips per script is
    # the dominant latency cost otherwise.
    vectors = embedder.embed(queries, stage="query_embed")

    out: list[Retrieved] = []
    for claim, query, vector in zip(claims, queries, vectors):
        hits = store.search(query, vector, top_k=settings.top_k,
                            product=claim.product or product_hint)
        out.append(Retrieved(claim=claim, hits=hits))
        log.info("claim retrieval",
                 extra={"claim_id": claim.id, "n_hits": len(hits),
                        "best_dense": round(max((h.dense_score for h in hits), default=0.0), 4),
                        "top_citation": hits[0].citation() if hits else None})
    return out


def format_evidence(hits: list[Hit], *, max_chars: int = 1400) -> str:
    """Render excerpts for the verifier prompt. Each carries its chunk id so the
    model can cite what it used, and the verdict stays traceable to a source."""
    if not hits:
        return "(no manual excerpts were retrieved for this claim)"
    parts = []
    for hit in hits:
        body = hit.text.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + " ..."
        parts.append(f"[{hit.chunk_id}] source: {hit.citation()} "
                     f"(file: {hit.source_file}, type: {hit.kind})\n{body}")
    return "\n\n".join(parts)
