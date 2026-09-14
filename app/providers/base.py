"""Provider abstraction.

The pipeline only ever talks to `LLM` and `Embedder`. Swapping OpenAI for Gemini
is a config change, not a code change -- which also makes the two directly
comparable on the same golden eval set.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class Usage:
    """Token accounting, aggregated per run so cost-per-run is reportable."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)

    def add(self, stage: str, in_tok: int = 0, out_tok: int = 0, embed_tok: int = 0) -> None:
        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.embed_tokens += embed_tok
        self.by_stage[stage] = self.by_stage.get(stage, 0) + in_tok + out_tok + embed_tok

    def as_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "embed_tokens": self.embed_tokens,
                "by_stage": self.by_stage}


class LLM(Protocol):
    name: str
    usage: Usage

    def json(self, *, system: str, user: str, stage: str, schema_hint: str = "") -> Any: ...


class Embedder(Protocol):
    name: str
    dim: int
    usage: Usage

    def embed(self, texts: list[str], *, stage: str = "embed") -> list[list[float]]: ...


class ProviderError(RuntimeError):
    pass


def parse_json(raw: str) -> Any:
    """Models occasionally wrap JSON in prose or fences despite JSON mode.
    Parse defensively rather than failing a whole run on a stray backtick."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost brace/bracket span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ProviderError(f"model did not return parseable JSON: {raw[:400]!r}")


# Client errors that will never succeed on retry: a bad key, a revoked model, a
# malformed request. 429 is excluded -- rate limiting IS transient and is the
# main reason this wrapper exists.
_PERMANENT_STATUS = (400, 401, 403, 404, 405, 422)


def is_permanent(exc: Exception) -> bool:
    """Provider SDKs surface HTTP status differently (openai sets .status_code,
    google-genai sets .code and embeds the status in the message), so check the
    structured attributes first and fall back to the text."""
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value in _PERMANENT_STATUS:
            return True
    text = str(exc)[:200]
    return any(str(code) in text for code in _PERMANENT_STATUS)


def with_retries(fn, *, attempts: int = 4, base_delay: float = 1.0, stage: str = ""):
    """Exponential backoff over transient provider failures (rate limits, 5xx).
    A single flaky call should not lose a whole scoring run.

    Permanent failures are raised immediately. Retrying a retired model name
    four times per manual turns a clear configuration error into a slow one.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # provider SDKs raise their own types
            last = exc
            if is_permanent(exc):
                log.error("provider call failed permanently, not retrying",
                          extra={"stage": stage, "error": str(exc)[:300]})
                raise ProviderError(f"provider rejected the call ({stage}): {exc}") from exc
            if attempt == attempts - 1:
                break
            delay = base_delay * (2 ** attempt)
            log.warning("provider call failed, retrying",
                        extra={"stage": stage, "attempt": attempt + 1, "delay_s": delay,
                               "error": str(exc)[:300]})
            time.sleep(delay)
    raise ProviderError(f"provider call failed after {attempts} attempts ({stage}): {last}") from last
