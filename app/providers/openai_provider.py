from __future__ import annotations

import logging
from typing import Any

from .base import Embedder, LLM, Usage, parse_json, with_retries

log = logging.getLogger(__name__)

_EMBED_DIMS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536,
               "text-embedding-ada-002": 1536}


class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.name = model
        self.usage = Usage()

    def json(self, *, system: str, user: str, stage: str, schema_hint: str = "") -> Any:
        prompt = f"{user}\n\n{schema_hint}".strip()

        def _call():
            return self._client.chat.completions.create(
                model=self.name,
                temperature=0,  # scoring must be reproducible run-to-run
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
            )

        resp = with_retries(_call, stage=stage)
        u = resp.usage
        self.usage.add(stage, in_tok=getattr(u, "prompt_tokens", 0) or 0,
                       out_tok=getattr(u, "completion_tokens", 0) or 0)
        return parse_json(resp.choices[0].message.content or "")


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.name = model
        self.dim = _EMBED_DIMS.get(model, 3072)
        self.usage = Usage()

    def embed(self, texts: list[str], *, stage: str = "embed") -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        # Batched to stay under per-request input limits on large manuals.
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            resp = with_retries(
                lambda b=batch: self._client.embeddings.create(model=self.name, input=b),
                stage=stage)
            out.extend(d.embedding for d in resp.data)
            self.usage.add(stage, embed_tok=getattr(resp.usage, "total_tokens", 0) or 0)
        self.dim = len(out[0])
        return out
