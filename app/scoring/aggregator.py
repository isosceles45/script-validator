"""Stage 6: combine the three axes and write the reviewer note."""
from __future__ import annotations

import logging
from typing import Any

from ..providers import LLM
from .prompts import FEEDBACK_SCHEMA, FEEDBACK_SYSTEM, FEEDBACK_USER, JSON_ONLY
from .verifier import Verdict

log = logging.getLogger(__name__)

# Two conditions block publication regardless of how well a script scores
# elsewhere, so the verdict label is capped independently of the weighted average:
# a claim the manuals actively refute, and a high-risk claim (medical, safety,
# absolute superlative) the manuals cannot substantiate. The second matters
# because a manual is silent on almost every false medical claim ever written --
# treating that silence as a minor gap is how "cures acne" ships.
BLOCKING_VERDICTS = {"contradicted"}


def is_blocking(verdict) -> bool:
    return (verdict.verdict in BLOCKING_VERDICTS
            or (verdict.verdict == "unverifiable" and verdict.claim.risk == "high"))


def combine_scores(*, brief_score: float, message_score: float,
                   claim_score: float | None,
                   weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Weighted mean, renormalised when an axis is not applicable.

    Claim validity is None for scripts that make no factual claims. Treating that
    as 10 would reward claim-free copy; treating it as 0 would punish it. Dropping
    the axis and redistributing its weight is the only defensible option.
    """
    parts: dict[str, float] = {"brief_alignment": brief_score,
                               "message_quality": message_score}
    if claim_score is not None:
        parts["claim_validity"] = claim_score

    active = {k: weights[k] for k in parts}
    total_weight = sum(active.values()) or 1.0
    normalised = {k: w / total_weight for k, w in active.items()}
    overall = sum(parts[k] * normalised[k] for k in parts)
    return round(overall, 2), {k: round(v, 4) for k, v in normalised.items()}


def verdict_label(overall: float, verdicts: list[Verdict]) -> str:
    if any(is_blocking(v) for v in verdicts):
        return "needs_revision_blocking"
    if overall >= 8.0:
        return "approved"
    if overall >= 6.0:
        return "approved_with_edits"
    return "needs_revision"


def _format_claim_findings(verdicts: list[Verdict]) -> str:
    if not verdicts:
        return "(the script makes no checkable product claims)"
    order = {"contradicted": 0, "partially_supported": 1, "unverifiable": 2,
             "supported": 3}
    lines = []
    for v in sorted(verdicts, key=lambda v: (order[v.verdict], v.claim.risk != "high")):
        line = (f"- [{v.verdict.upper()}] ({v.claim.risk} risk) \"{v.claim.text}\"\n"
                f"    rationale: {v.rationale}")
        if v.citations:
            line += f"\n    source: {'; '.join(v.citations)}"
        if v.manual_quote:
            line += f"\n    manual says: \"{v.manual_quote}\""
        if v.suggested_fix:
            line += f"\n    suggested replacement: \"{v.suggested_fix}\""
        lines.append(line)
    return "\n".join(lines)


def _format_bullets(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    rendered = []
    for item in items:
        if isinstance(item, dict):
            rendered.append(f"- {item.get('issue', '')} -> {item.get('rewrite', '')}")
        else:
            rendered.append(f"- {item}")
    return f"{title}:\n" + "\n".join(rendered)


def write_feedback(llm: LLM, *, scores: dict[str, Any], verdict: str,
                   verdicts: list[Verdict], brief_result: dict[str, Any],
                   message_result: dict[str, Any]) -> dict[str, Any]:
    brief_findings = "\n".join(filter(None, [
        brief_result.get("justification", ""),
        _format_bullets("Missing mandatories", brief_result.get("missing_mandatories", [])),
        _format_bullets("Gaps", brief_result.get("gaps", [])),
        _format_bullets("Strengths", brief_result.get("strengths", [])),
    ])) or "(none recorded)"

    message_findings = "\n".join(filter(None, [
        message_result.get("justification", ""),
        _format_bullets("Improvements", message_result.get("improvements", [])),
        _format_bullets("Strengths", message_result.get("strengths", [])),
    ])) or "(none recorded)"

    claim_score = scores.get("claim_validity")
    raw = llm.json(
        system=FEEDBACK_SYSTEM,
        user=FEEDBACK_USER.format(
            brief_score=scores["brief_alignment"],
            message_score=scores["message_quality"],
            claim_score=f"{claim_score}/10" if claim_score is not None
                        else "not applicable (no checkable claims)",
            overall_score=scores["overall"],
            verdict=verdict,
            claim_findings=_format_claim_findings(verdicts),
            brief_findings=brief_findings,
            message_findings=message_findings),
        schema_hint=f"{FEEDBACK_SCHEMA}\n\n{JSON_ONLY}",
        stage="overall_feedback",
    )
    return {"feedback": str(raw.get("feedback") or "").strip(),
            "top_actions": [str(a) for a in (raw.get("top_actions") or [])]}
