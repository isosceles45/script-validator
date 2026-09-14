"""Stage 3: verify each claim against its retrieved excerpts, and turn the
verdicts into the Product Claim Validity score.

The score is computed deterministically from the verdicts rather than asked of an
LLM. Two reasons: an LLM asked for a holistic 1-10 is not reproducible run to
run, and a penalty table can be shown to a brand team and argued with -- which is
what makes the number defensible.
"""
from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any

from ..providers import LLM
from .claims import Claim
from .prompts import JSON_ONLY, VERIFIER_SCHEMA, VERIFIER_SYSTEM, VERIFIER_USER
from .retriever import Retrieved, format_evidence

log = logging.getLogger(__name__)

VERDICTS = {"supported", "partially_supported", "contradicted", "unverifiable"}

# Penalty applied to a starting score of 10, by verdict and claim risk.
#
# The high-risk `unverifiable` cell is the one that matters most and the one
# that is easiest to get wrong. "The manual is silent" is a mild documentation
# gap for "contains mango butter" and a publication blocker for "dermatologist
# proven to cure acne" -- because the real-world rule is that you cannot publish
# a medical or safety claim you cannot evidence. Silence is not exoneration. It
# sits just below an outright contradiction: still catastrophic, but a
# contradiction is worse because the manual actively refutes it.
PENALTY: dict[str, dict[str, float]] = {
    "contradicted":        {"high": 5.0, "medium": 3.0, "low": 1.5},
    "partially_supported": {"high": 1.2, "medium": 0.7, "low": 0.35},
    "unverifiable":        {"high": 4.0, "medium": 0.9, "low": 0.4},
    "supported":           {"high": 0.0, "medium": 0.0, "low": 0.0},
}


@dataclass
class Verdict:
    claim: Claim
    verdict: str
    confidence: float
    rationale: str
    evidence_chunk_ids: list[str]
    manual_quote: str | None
    suggested_fix: str | None
    citations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {**self.claim.as_dict(),
                "verdict": self.verdict,
                "confidence": round(self.confidence, 3),
                "rationale": self.rationale,
                "manual_quote": self.manual_quote,
                "citations": self.citations,
                "evidence_chunk_ids": self.evidence_chunk_ids,
                "suggested_fix": self.suggested_fix}


def _verify_one(llm: LLM, item: Retrieved) -> Verdict:
    claim = item.claim
    if not item.hits:
        # No evidence retrieved at all: skip the LLM call entirely. Asking a model
        # to verify against nothing invites it to fall back on world knowledge,
        # which is exactly the failure this pipeline exists to prevent.
        return Verdict(claim=claim, verdict="unverifiable", confidence=1.0,
                       rationale="No manual excerpts were retrieved for this claim; "
                                 "the product catalogue does not appear to cover it.",
                       evidence_chunk_ids=[], manual_quote=None,
                       suggested_fix=None, citations=[])

    raw = llm.json(
        system=VERIFIER_SYSTEM,
        user=VERIFIER_USER.format(
            claim_id=claim.id, claim_text=claim.text,
            claim_quote=claim.quote or claim.text, claim_type=claim.type,
            evidence=format_evidence(item.hits)),
        schema_hint=f"{VERIFIER_SCHEMA}\n\n{JSON_ONLY}",
        stage="claim_verification",
    )

    verdict = str(raw.get("verdict", "unverifiable")).lower().strip()
    if verdict not in VERDICTS:
        log.warning("verifier returned unknown verdict, treating as unverifiable",
                    extra={"claim_id": claim.id, "raw_verdict": verdict})
        verdict = "unverifiable"

    used_ids = [str(i) for i in (raw.get("evidence_chunk_ids") or [])]
    by_id = {h.chunk_id: h for h in item.hits}
    # Only cite excerpts that were actually in the evidence set; a model
    # occasionally invents a plausible-looking chunk id.
    cited = [by_id[i] for i in used_ids if i in by_id]
    if not cited and verdict in {"supported", "partially_supported", "contradicted"}:
        cited = item.hits[:1]

    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    return Verdict(
        claim=claim,
        verdict=verdict,
        confidence=min(max(confidence, 0.0), 1.0),
        rationale=str(raw.get("rationale") or "").strip(),
        evidence_chunk_ids=[h.chunk_id for h in cited],
        manual_quote=(raw.get("manual_quote") or None),
        suggested_fix=(raw.get("suggested_fix") or None),
        citations=[h.citation() for h in cited],
    )


def verify_claims(llm: LLM, retrieved: list[Retrieved], *,
                  max_workers: int = 6) -> list[Verdict]:
    """Claims are verified concurrently -- they are independent, and serialising
    them makes latency scale linearly with script length."""
    if not retrieved:
        return []
    verdicts: list[Verdict | None] = [None] * len(retrieved)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_verify_one, llm, item): idx
                   for idx, item in enumerate(retrieved)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                verdicts[idx] = future.result()
            except Exception as exc:
                # One failed verification must not lose the other findings; it is
                # recorded as unverifiable with the error kept for the log.
                claim = retrieved[idx].claim
                log.exception("claim verification failed", extra={"claim_id": claim.id})
                verdicts[idx] = Verdict(
                    claim=claim, verdict="unverifiable", confidence=0.0,
                    rationale=f"Verification failed: {str(exc)[:200]}",
                    evidence_chunk_ids=[], manual_quote=None,
                    suggested_fix=None, citations=[])
    result = [v for v in verdicts if v is not None]
    log.info("claims verified", extra=tally(result))
    return result


def tally(verdicts: list[Verdict]) -> dict[str, int]:
    counts = {v: 0 for v in VERDICTS}
    for verdict in verdicts:
        counts[verdict.verdict] += 1
    return {"total": len(verdicts), **counts}


def claim_validity_score(verdicts: list[Verdict]) -> float | None:
    """Returns None when the script makes no factual claims at all.

    None is deliberately not 10: a purely emotional script has not *earned* a
    perfect claim-accuracy score, it simply has nothing to check. The aggregator
    drops the axis and renormalises the remaining weights rather than rewarding
    a script for making no claims.
    """
    if not verdicts:
        return None
    score = 10.0
    for verdict in verdicts:
        score -= PENALTY[verdict.verdict][verdict.claim.risk]
    return round(min(max(score, 0.0), 10.0), 2)
