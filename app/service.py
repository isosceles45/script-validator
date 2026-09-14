"""Dependency wiring shared by the API, the CLI and the MCP server."""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import Settings, get_settings
from .providers import Embedder, LLM, get_embedder, get_llm
from .store.run_store import build_run_store
from .store.vector_store import VectorStore

log = logging.getLogger(__name__)


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(get_settings().db_path)


@lru_cache
def get_run_store():
    return build_run_store(get_settings())


def new_providers(settings: Settings | None = None) -> tuple[LLM, Embedder]:
    """Fresh provider instances per run.

    Deliberately not cached: `Usage` accumulates on the instance, and a shared
    singleton would report the process's lifetime token count as this run's cost.
    Per-run cost accounting is worth more than the microseconds saved reusing a
    client object.
    """
    settings = settings or get_settings()
    return get_llm(settings), get_embedder(settings)


def health() -> dict:
    settings = get_settings()
    store = get_vector_store()
    stats = store.stats()
    key_present = bool(settings.openai_api_key if settings.provider == "openai"
                       else settings.google_api_key)
    ready = stats["n_chunks"] > 0 and key_present
    return {
        "status": "ok" if ready else "degraded",
        "provider": settings.provider,
        "llm_model": settings.model_name(),
        "embed_model": settings.embed_model_name(),
        "api_key_configured": key_present,
        "corpus": {"n_manuals": stats["n_manuals"], "n_chunks": stats["n_chunks"]},
        "reason": None if ready else (
            "no manuals ingested -- run `python -m app.cli ingest`"
            if stats["n_chunks"] == 0 else
            f"no API key configured for provider {settings.provider!r}"),
    }
