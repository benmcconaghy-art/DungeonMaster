"""Alembic environment — wired to our async engine and ORM metadata."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from alembic import context
from app.config import get_settings

# Importing models registers their tables on Base.metadata so autogenerate
# can see them. Empty in Phase 0; populated as Phase 1 lands the schema.
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.session import create_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _db_url() -> str:
    """Resolve the database URL from app settings (single source of truth)."""

    return get_settings().db_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emits SQL without a live engine."""

    context.configure(
        url=_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    """Configure Alembic's context against a live (sync) connection.

    Called via ``connection.run_sync(...)`` from the async migration
    runner — Alembic's migration loop is synchronous, but the underlying
    engine here is the async aiosqlite one (which does run our WAL pragma
    hook on connect; the same migration that creates the schema therefore
    sees ``foreign_keys=ON`` and the rest of the spec §5 pragmas).
    """

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # render_as_batch lets SQLite get ALTER TABLE-equivalent behaviour
        # via batch operations; required for any future column drop / type
        # change because SQLite cannot ALTER COLUMN in place.
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def _run_migrations_online_async() -> None:
    """Run migrations in 'online' mode using our async engine."""

    engine: AsyncEngine = create_engine(_db_url())
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(_run_migrations_online_async())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
