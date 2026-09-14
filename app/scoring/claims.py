"""Stage 1: extract atomic, checkable claims from the script."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..providers import LLM
from .prompts import (CLAIM_EXTRACTOR_SCHEMA, CLAIM_EXTRACTOR_SYSTEM,
                      CLAIM_EXTRACTOR_USER, JSON_ONLY)

log = logging.getLogger(__name__)

VALID_RISK = {"high", "medium", "low"}
VALID_TYPE = {"ingredient", "efficacy", "safety", "usage", "spec", "comparative",
              "certification"}


@dataclass
class Claim:
    id: str
    text: str
    quote: str = ""
    product: str | None = None
    type: str = "efficacy"
    risk: str = "medium"

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "quote": self.quote,
                "product": self.product, "type": self.type, "risk": self.risk}


@dataclass
class ExtractionResult:
    claims: list[Claim] = field(default_factory=list)
    product_mentions: list[str] = field(default_factory=list)
    non_factual_lines: list[str] = field(default_factory=list)


def extract_claims(llm: LLM, *, brief: str, script: str) -> ExtractionResult:
    raw = llm.json(
        system=CLAIM_EXTRACTOR_SYSTEM,
        user=CLAIM_EXTRACTOR_USER.format(brief=brief, script=script),
        schema_hint=f"{CLAIM_EXTRACTOR_SCHEMA}\n\n{JSON_ONLY}",
        stage="claim_extraction",
    )

    claims: list[Claim] = []
    for i, item in enumerate(raw.get("claims") or [], start=1):
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        risk = str(item.get("risk", "medium")).lower()
        ctype = str(item.get("type", "efficacy")).lower()
        claims.append(Claim(
            # Re-issue ids rather than trusting the model's: downstream joins
            # (verdict -> claim -> evidence) break silently on a duplicate id.
            id=f"c{i}",
            text=text,
            quote=(item.get("quote") or "").strip(),
            product=(item.get("product") or None),
            type=ctype if ctype in VALID_TYPE else "efficacy",
            risk=risk if risk in VALID_RISK else "medium",
        ))

    result = ExtractionResult(
        claims=claims,
        product_mentions=[str(p) for p in (raw.get("product_mentions") or []) if p],
        non_factual_lines=[str(line) for line in (raw.get("non_factual_lines") or []) if line],
    )
    log.info("claims extracted",
             extra={"n_claims": len(claims),
                    "risk_mix": {r: sum(1 for c in claims if c.risk == r)
                                 for r in VALID_RISK},
                    "products": result.product_mentions})
    return result
