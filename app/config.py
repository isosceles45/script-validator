"""Central configuration. Everything tunable lives here so a run is reproducible
from (config snapshot + prompt version + model name), all of which get logged."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Bumped whenever a prompt template changes, so score drift is traceable per run.
PROMPT_VERSION = "2026-09-15.1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider: str = "openai"

    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-large"

    google_api_key: str = ""
    gemini_llm_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "text-embedding-004"

    data_dir: Path = Path("./data")
    db_path: Path = Path("./data/store.sqlite3")

    top_k: int = 5
    chunk_tokens: int = 500
    chunk_overlap: float = 0.15
    min_chunk_chars: int = 40
    min_similarity: float = 0.25

    w_brief: float = 0.30
    w_message: float = 0.25
    w_claims: float = 0.45

    golden_sample_size: int = 8
    log_level: str = "INFO"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def manuals_dir(self) -> Path:
        return self.data_dir / "manuals"

    @property
    def golden_path(self) -> Path:
        return self.data_dir / "eval" / "golden.yaml"

    @property
    def weights(self) -> dict[str, float]:
        return {"brief_alignment": self.w_brief,
                "message_quality": self.w_message,
                "claim_validity": self.w_claims}

    def model_name(self) -> str:
        return self.openai_llm_model if self.provider == "openai" else self.gemini_llm_model

    def embed_model_name(self) -> str:
        return self.openai_embed_model if self.provider == "openai" else self.gemini_embed_model


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.runs_dir.mkdir(parents=True, exist_ok=True)
    s.manuals_dir.mkdir(parents=True, exist_ok=True)
    (s.data_dir / "eval").mkdir(parents=True, exist_ok=True)
    return s
