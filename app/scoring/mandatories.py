"""Deterministic mandatory checking.

A campaign brief's mandatories split cleanly in two. Some are settled by looking
for a string -- a hashtag, a product name, a required disclaimer. Some need
interpretation -- "show the texture on camera", "end with a clear CTA".

Asking an LLM to do the first kind is the same mistake as asking it for a
holistic claim-validity score: it usually works, and when it fails it fails
invisibly and confidently. Observed in practice on this corpus: gpt-4o-mini
reported "did not say the full product name" for a script whose opening line was
"I want to talk about the TFS Tea Tree Pore Ampoule", costing 5 points on the
mandatories dimension.

So literal mandatories are settled here, in code, and the result is handed to the
brief-alignment scorer as fact. Judgement mandatories are left to the model.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from ..providers import LLM
from .prompts import (JSON_ONLY, MANDATORY_EXTRACTOR_SCHEMA,
                      MANDATORY_EXTRACTOR_SYSTEM, MANDATORY_EXTRACTOR_USER)

log = logging.getLogger(__name__)

LITERAL, JUDGEMENT = "literal", "judgement"


def normalize(text: str) -> str:
    """Fold the differences that should not decide a mandatory: case, unicode
    punctuation (creators' apostrophes and dashes come from phone keyboards),
    and whitespace runs. Hash and handle characters are kept -- "#PoreCheck" and
    "PoreCheck" are genuinely different mandatories."""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―]", "-", text)
    text = re.sub(r"[^\w\s#@'\"./-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Mandatory:
    text: str
    kind: str
    tokens: list[str]
    present: bool | None = None      # None => judgement, not settled here
    matched_token: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "tokens": self.tokens,
                "present": self.present, "matched_token": self.matched_token}


def extract_mandatories(llm: LLM, *, brief: str) -> list[Mandatory]:
    raw = llm.json(
        system=MANDATORY_EXTRACTOR_SYSTEM,
        user=MANDATORY_EXTRACTOR_USER.format(brief=brief),
        schema_hint=f"{MANDATORY_EXTRACTOR_SCHEMA}\n\n{JSON_ONLY}",
        stage="mandatory_extraction",
    )
    out: list[Mandatory] = []
    for item in (raw.get("mandatories") or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        kind = str(item.get("kind", JUDGEMENT)).lower()
        tokens = [str(t).strip() for t in (item.get("tokens") or []) if str(t).strip()]
        # A "literal" mandatory with no token to search for cannot be checked;
        # treat it as judgement rather than silently passing or failing it.
        if kind != LITERAL or not tokens:
            kind, tokens = JUDGEMENT, []
        out.append(Mandatory(text=text, kind=kind, tokens=tokens))
    return out


def check(script: str, mandatories: list[Mandatory]) -> list[Mandatory]:
    """Settle every literal mandatory against the script. A mandatory with
    several tokens passes if ANY of them appears -- multiple tokens are spelling
    variants of one requirement, not a list of separate requirements."""
    haystack = normalize(script)
    for mandatory in mandatories:
        if mandatory.kind != LITERAL:
            continue
        mandatory.present = False
        for token in mandatory.tokens:
            needle = normalize(token)
            if needle and needle in haystack:
                mandatory.present = True
                mandatory.matched_token = token
                break
    return mandatories


def format_for_prompt(mandatories: list[Mandatory]) -> str:
    if not mandatories:
        return "(the brief states no explicit mandatories)"
    lines = []
    for m in mandatories:
        if m.kind == LITERAL:
            state = "PRESENT" if m.present else "ABSENT"
            detail = (f' (found "{m.matched_token}")' if m.present
                      else f' (searched for: {", ".join(repr(t) for t in m.tokens)})')
            lines.append(f"- [VERIFIED] {state}: {m.text}{detail}")
        else:
            lines.append(f"- [JUDGEMENT] {m.text}")
    return "\n".join(lines)


def summary(mandatories: list[Mandatory]) -> dict[str, Any]:
    literal = [m for m in mandatories if m.kind == LITERAL]
    return {
        "checked": [m.as_dict() for m in mandatories],
        "n_literal": len(literal),
        "n_judgement": len(mandatories) - len(literal),
        "verified_missing": [m.text for m in literal if m.present is False],
        "verified_present": [m.text for m in literal if m.present is True],
    }


def run(llm: LLM, *, brief: str, script: str) -> tuple[list[Mandatory], str]:
    mandatories = check(script, extract_mandatories(llm, brief=brief))
    log.info("mandatory check", extra=summary(mandatories))
    return mandatories, format_for_prompt(mandatories)
