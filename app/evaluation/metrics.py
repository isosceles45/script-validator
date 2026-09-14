"""Retrieval metrics.

Two distinct questions get conflated in most RAG demos, and this module keeps
them apart:

  1. "Is my retrieval system healthy?" -- answered by a fixed golden set with
     known-correct answers. Comparable across runs, across providers, and across
     chunking changes. This is a benchmark.

  2. "How well grounded was THIS script?" -- no ground truth exists for a script
     submitted five seconds ago, so this is a proxy: how many of its claims
     retrieved anything convincing at all. This is a health signal for one run.

Reporting only (1) hides that a particular script hit a hole in the catalogue.
Reporting only (2) lets a degraded retriever look fine on a script that happens
to make vague claims. Both are attached to every run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..store.vector_store import Hit


@dataclass
class RankedEval:
    recall_at_k: float
    mrr: float
    hit_rate: float          # fraction of cases with at least one relevant hit
    n_cases: int
    k: int
    per_case: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {"recall_at_k": round(self.recall_at_k, 4),
                "mrr": round(self.mrr, 4),
                "hit_rate": round(self.hit_rate, 4),
                "n_cases": self.n_cases, "k": self.k,
                "per_case": self.per_case}


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    for index, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1.0 / index
    return 0.0


def aggregate(per_case: list[dict[str, Any]], k: int) -> RankedEval:
    if not per_case:
        return RankedEval(0.0, 0.0, 0.0, 0, k, [])
    n = len(per_case)
    return RankedEval(
        # Recall@k here is "was a relevant chunk retrieved in the top k" averaged
        # over cases. Each golden case has one correct *source*, possibly spread
        # over several chunks, so set-recall over chunk ids would be misleading.
        recall_at_k=sum(c["hit"] for c in per_case) / n,
        mrr=sum(c["reciprocal_rank"] for c in per_case) / n,
        hit_rate=sum(c["hit"] for c in per_case) / n,
        n_cases=n, k=k, per_case=per_case)


def grounding_rate(per_claim: list[tuple[str, list[Hit]]],
                   *, min_similarity: float) -> dict[str, Any]:
    """Live proxy metric: the share of this script's claims for which retrieval
    returned at least one chunk above the similarity floor.

    Uses raw cosine, not the fused rank score -- the fused score is only
    meaningful within a single query's candidate ordering and cannot be compared
    against a fixed threshold.
    """
    if not per_claim:
        return {"grounded_claims": 0, "total_claims": 0, "grounding_rate": None,
                "min_similarity": min_similarity, "ungrounded_claim_ids": []}

    ungrounded = []
    grounded = 0
    for claim_id, hits in per_claim:
        best = max((h.dense_score for h in hits), default=0.0)
        if best >= min_similarity:
            grounded += 1
        else:
            ungrounded.append({"claim_id": claim_id, "best_similarity": round(best, 4)})

    return {"grounded_claims": grounded,
            "total_claims": len(per_claim),
            "grounding_rate": round(grounded / len(per_claim), 4),
            "min_similarity": min_similarity,
            "ungrounded_claim_ids": ungrounded}
