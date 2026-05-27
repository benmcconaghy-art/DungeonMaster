"""Application configuration via pydantic-settings.

Single source of truth for runtime settings. Values are read from process
environment variables; in dev a ``.env`` file at the repo root is also
consulted (see ``.env.example``). The production deployment passes them via
the ``EnvironmentFile=`` line in ``deploy/dungeon-master.service``.

Defaults are suitable for local development. Production deployments must
set ``VLLM_BASE_URL``, ``FLUX_BASE_URL``, and ``SESSION_SECRET`` at minimum.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-level configuration.

    Override any field by setting an environment variable of the same name
    (case-insensitive); pydantic-settings handles the mapping.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    db_path: Path = Field(
        default=Path("/var/lib/dungeon-master/dm.db"),
        description="Filesystem path to the SQLite database file.",
    )
    image_storage_path: Path = Field(
        default=Path("/var/lib/dungeon-master/images"),
        description="Directory holding generated PNGs served via X-Accel-Redirect.",
    )
    vllm_base_url: str = Field(
        default="http://localhost:8000",
        description="Root URL of the vLLM endpoint (OpenAI-compatible). Set via VLLM_BASE_URL.",
    )
    flux_base_url: str = Field(
        default="http://localhost:11437",
        description="Root URL of the FLUX.1 image service. Set via FLUX_BASE_URL.",
    )
    redis_url: str = Field(
        default="redis://127.0.0.1:6379/0",
        description="Redis connection URL used for pub/sub and the image queue.",
    )
    session_secret: str = Field(
        default="dev-only-not-secret-replace-in-production",
        description="Server-side secret used to sign session cookies.",
        min_length=16,
    )
    embedding_base_url: str | None = Field(
        default=None,
        description=(
            "Optional URL of an OpenAI-compatible embedding endpoint, e.g."
            " 'http://YOUR_HOST:11436/v1' (Ollama). When"
            " unset, the local sentence-transformers backend is used."
        ),
    )
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description=(
            "Model id passed to /v1/embeddings (or sentence-transformers"
            " load_model). Default is 1024-dimensional, matching the spec §5"
            " convention. Changing this invalidates existing world_facts rows"
            " — they live in the old model's vector space."
        ),
    )
    embedding_dim: int = Field(
        default=1024,
        description=(
            "Expected output dimension. The embedder asserts at startup that"
            " its actual output matches this; mismatches fail the worker rather"
            " than silently writing inconsistent BLOBs."
        ),
        ge=1,
    )

    @property
    def db_url(self) -> str:
        """SQLAlchemy async URL derived from ``db_path``."""
        return f"sqlite+aiosqlite:///{self.db_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the ``.env`` file is parsed exactly once. Tests that need to
    override values should call ``get_settings.cache_clear()`` first.
    """

    return Settings()
