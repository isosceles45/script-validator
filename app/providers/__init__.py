"""Provider factory. `PROVIDER=openai|gemini` selects the whole stack."""
from __future__ import annotations

from ..config import Settings
from .base import Embedder, LLM, ProviderError, Usage


def get_llm(settings: Settings) -> LLM:
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ProviderError("PROVIDER=openai but OPENAI_API_KEY is not set")
        from .openai_provider import OpenAILLM
        return OpenAILLM(settings.openai_api_key, settings.openai_llm_model)
    if settings.provider == "gemini":
        if not settings.google_api_key:
            raise ProviderError("PROVIDER=gemini but GOOGLE_API_KEY is not set")
        from .gemini_provider import GeminiLLM
        return GeminiLLM(settings.google_api_key, settings.gemini_llm_model)
    raise ProviderError(f"unknown PROVIDER={settings.provider!r} (expected 'openai' or 'gemini')")


def get_embedder(settings: Settings) -> Embedder:
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ProviderError("PROVIDER=openai but OPENAI_API_KEY is not set")
        from .openai_provider import OpenAIEmbedder
        return OpenAIEmbedder(settings.openai_api_key, settings.openai_embed_model)
    if settings.provider == "gemini":
        if not settings.google_api_key:
            raise ProviderError("PROVIDER=gemini but GOOGLE_API_KEY is not set")
        from .gemini_provider import GeminiEmbedder
        return GeminiEmbedder(settings.google_api_key, settings.gemini_embed_model)
    raise ProviderError(f"unknown PROVIDER={settings.provider!r} (expected 'openai' or 'gemini')")


__all__ = ["get_llm", "get_embedder", "LLM", "Embedder", "Usage", "ProviderError"]
