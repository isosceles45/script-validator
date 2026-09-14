"""Offline provider doubles.

These exist so the whole pipeline -- ingestion, retrieval, verification,
aggregation, eval, HTTP layer -- is testable in CI with no API key and no
network. The embedder is a deterministic hashed bag-of-words, which is weak
semantically but genuinely ranks lexically-overlapping text higher, so retrieval
assertions are meaningful rather than vacuous.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from app.providers.base import Usage

DIM = 256


class FakeEmbedder:
    def __init__(self, dim: int = DIM) -> None:
        self.name = "fake-hash-embedder"
        self.dim = dim
        self.usage = Usage()
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], *, stage: str = "embed") -> list[list[float]]:
        self.calls.append(list(texts))
        self.usage.add(stage, embed_tok=sum(len(t.split()) for t in texts))
        out = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                vector[digest[0] % self.dim] += 1.0
            out.append(vector)
        return out


class ScriptedLLM:
    """Returns a canned payload per pipeline stage. `responses` maps stage name to
    either a dict or a callable taking the user prompt."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.name = "fake-scripted-llm"
        self.usage = Usage()
        self.responses = responses
        self.seen: list[tuple[str, str]] = []

    def json(self, *, system: str, user: str, stage: str, schema_hint: str = "") -> Any:
        self.seen.append((stage, user))
        self.usage.add(stage, in_tok=len(user) // 4, out_tok=32)
        payload = self.responses.get(stage)
        if payload is None:
            raise AssertionError(f"ScriptedLLM has no response for stage {stage!r}")
        return payload(user) if callable(payload) else payload


def default_responses(claims: list[dict[str, Any]] | None = None,
                      verdict: str = "supported") -> dict[str, Any]:
    claims = claims if claims is not None else [{
        "id": "c1", "text": "contains tea tree leaf water",
        "quote": "It contains tea tree leaf water.", "product": "Tea Tree Pore Ampoule",
        "type": "ingredient", "risk": "medium"}]
    return {
        "claim_extraction": {"product_mentions": ["Tea Tree Pore Ampoule"],
                             "claims": claims, "non_factual_lines": ["Hey besties!"]},
        "claim_verification": {
            "verdict": verdict, "confidence": 0.9,
            "rationale": "The manual states this directly.",
            "evidence_chunk_ids": [], "manual_quote": "Contains tea tree leaf water.",
            "suggested_fix": None if verdict == "supported" else "Softened copy."},
        "brief_alignment": {
            "score": 8, "justification": "Hits the brief.",
            "dimensions": {"objective": 8, "audience": 8, "tone_and_voice": 7,
                           "key_message": 8, "mandatories_and_format": 9},
            "missing_mandatories": [], "strengths": ["On-audience"], "gaps": []},
        "message_quality": {
            "score": 7, "justification": "Solid craft.",
            "dimensions": {"hook": 7, "clarity": 8, "structure_and_flow": 7,
                           "persuasiveness": 6, "call_to_action": 7,
                           "brand_voice_fit": 8},
            "strengths": ["Clear open"],
            "improvements": [{"issue": "Weak CTA", "rewrite": "Shop it today."}]},
        "overall_feedback": {"feedback": "Good script, tighten the CTA.",
                             "top_actions": ["Tighten the CTA"]},
    }
