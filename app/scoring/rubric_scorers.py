"""Stages 4 and 5: brief alignment and marketing message quality.

Neither needs retrieval -- brief alignment is a brief-vs-script comparison and
message quality is a craft judgement. Running them through the vector store
would add latency and a spurious grounding signal for no accuracy gain.
"""
from __future__ import annotations

import logging
from typing import Any

from ..providers import LLM
from .prompts import (BRIEF_ALIGNMENT_SCHEMA, BRIEF_ALIGNMENT_SYSTEM,
                      BRIEF_ALIGNMENT_USER, JSON_ONLY, MESSAGE_QUALITY_SCHEMA,
                      MESSAGE_QUALITY_SYSTEM, MESSAGE_QUALITY_USER)

log = logging.getLogger(__name__)


def _clamp_score(value: Any, default: float = 5.0) -> float:
    try:
        return round(min(max(float(value), 0.0), 10.0), 2)
    except (TypeError, ValueError):
        return default


def _clean_dimensions(raw: Any, expected: tuple[str, ...]) -> dict[str, float]:
    raw = raw if isinstance(raw, dict) else {}
    return {key: _clamp_score(raw.get(key)) for key in expected}


BRIEF_DIMENSIONS = ("objective", "audience", "tone_and_voice", "key_message",
                    "mandatories_and_format")
MESSAGE_DIMENSIONS = ("hook", "clarity", "structure_and_flow", "persuasiveness",
                      "call_to_action", "brand_voice_fit")


def score_brief_alignment(llm: LLM, *, brief: str, script: str) -> dict[str, Any]:
    raw = llm.json(
        system=BRIEF_ALIGNMENT_SYSTEM,
        user=BRIEF_ALIGNMENT_USER.format(brief=brief, script=script),
        schema_hint=f"{BRIEF_ALIGNMENT_SCHEMA}\n\n{JSON_ONLY}",
        stage="brief_alignment",
    )
    dimensions = _clean_dimensions(raw.get("dimensions"), BRIEF_DIMENSIONS)
    # Fall back to the dimension mean if the headline score is missing or junk,
    # rather than silently defaulting a whole axis to 5.
    score = _clamp_score(raw.get("score"),
                         default=round(sum(dimensions.values()) / len(dimensions), 2))
    result = {
        "score": score,
        "justification": str(raw.get("justification") or "").strip(),
        "dimensions": dimensions,
        "missing_mandatories": [str(x) for x in (raw.get("missing_mandatories") or [])],
        "strengths": [str(x) for x in (raw.get("strengths") or [])],
        "gaps": [str(x) for x in (raw.get("gaps") or [])],
    }
    log.info("brief alignment scored", extra={"score": score, "dimensions": dimensions})
    return result


def score_message_quality(llm: LLM, *, brief: str, script: str) -> dict[str, Any]:
    raw = llm.json(
        system=MESSAGE_QUALITY_SYSTEM,
        user=MESSAGE_QUALITY_USER.format(brief=brief, script=script),
        schema_hint=f"{MESSAGE_QUALITY_SCHEMA}\n\n{JSON_ONLY}",
        stage="message_quality",
    )
    dimensions = _clean_dimensions(raw.get("dimensions"), MESSAGE_DIMENSIONS)
    score = _clamp_score(raw.get("score"),
                         default=round(sum(dimensions.values()) / len(dimensions), 2))

    improvements = []
    for item in (raw.get("improvements") or []):
        if isinstance(item, dict):
            improvements.append({"issue": str(item.get("issue") or ""),
                                 "rewrite": str(item.get("rewrite") or "")})
        elif item:
            improvements.append({"issue": str(item), "rewrite": ""})

    result = {
        "score": score,
        "justification": str(raw.get("justification") or "").strip(),
        "dimensions": dimensions,
        "strengths": [str(x) for x in (raw.get("strengths") or [])],
        "improvements": improvements,
    }
    log.info("message quality scored", extra={"score": score, "dimensions": dimensions})
    return result
