"""Application configuration via pydantic-settings.

Single source of truth for runtime settings. Values are read from process
environment variables; in dev a ``.env`` file at the repo root is also
consulted. The production deployment passes them via the
``EnvironmentFile=`` line in ``deploy/dungeon-master.service``.

Defaults match the production layout described in spec §13:
``/var/lib/dungeon-master/dm.db`` for the database file,
``/var/lib/dungeon-master/images/`` for generated PNGs, the internal vLLM
and FLUX endpoints on ``svrai01``, and the local Redis instance.
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
        default="http://svrai01.mcconaghygroup.internal:8000",
        description="Root URL of the internal vLLM endpoint (Nemotron 3 Super).",
    )
    flux_base_url: str = Field(
        default="http://svrai01.mcconaghygroup.internal:11437",
        description="Root URL of the FLUX.1 [dev] / Kontext image service.",
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
