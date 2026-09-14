from __future__ import annotations

import logging
from typing import Any

from .base import Embedder, LLM, Usage, parse_json, with_retries

log = logging.getLogger(__name__)


class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.name = model
        self.usage = Usage()

    def json(self, *, system: str, user: str, stage: str, schema_hint: str = "") -> Any:
        from google.genai import types
        prompt = f"{user}\n\n{schema_hint}".strip()

        def _call():
            return self._client.models.generate_content(
                model=self.name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

        resp = with_retries(_call, stage=stage)
        meta = getattr(resp, "usage_metadata", None)
        self.usage.add(stage,
                       in_tok=getattr(meta, "prompt_token_count", 0) or 0,
                       out_tok=getattr(meta, "candidates_token_count", 0) or 0)
        return parse_json(resp.text or "")


class GeminiEmbedder(Embedder):
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self.name = model
        self.dim = 768
        self.usage = Usage()

    def embed(self, texts: list[str], *, stage: str = "embed") -> list[list[float]]:
        if not texts:
            return []
        from google.genai import types
        # Gemini's free tier caps batch size well below OpenAI's.
        task = "RETRIEVAL_QUERY" if stage.startswith("query") else "RETRIEVAL_DOCUMENT"
        out: list[list[float]] = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            resp = with_retries(
                lambda b=batch: self._client.models.embed_content(
                    model=self.name, contents=b,
                    config=types.EmbedContentConfig(task_type=task)),
                stage=stage)
            out.extend(list(e.values) for e in resp.embeddings)
            self.usage.add(stage, embed_tok=sum(len(t.split()) for t in batch))
        self.dim = len(out[0])
        return out
