"""Async SQLAlchemy engine, session factory, and the WAL pragma hook.

Wires every new SQLite connection (raw ``sqlite3.Connection`` underneath
``aiosqlite``) with the pragmas mandated by spec §5 and AGENTS.md
invariant #3. Without WAL the writer blocks every reader and multi-player
concurrency breaks; the rest of the pragmas are tuned for our 2-4 player
workload.

The ``connect`` event fires once per physical connection, so each pragma
runs exactly when it needs to — on connection setup, before any user
statement executes against that connection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

# Per spec §5. Run as separate statements; ``executescript`` would
# implicitly commit and bypass the connection's transaction state, so
# we drive the cursor directly.
_PRAGMAS: tuple[str, ...] = (
    "journal_mode = WAL",
    "synchronous = NORMAL",
    "foreign_keys = ON",
    "busy_timeout = 5000",
    "cache_size = -65536",
    "temp_store = MEMORY",
    "mmap_size = 268435456",
)


def _apply_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
    """Connection-event hook: apply WAL pragmas on every new connection.

    Registered only on engines created by :func:`create_engine`, all of
    which are SQLite — no dialect check needed. Uses the DBAPI cursor
    directly so the pragmas run before SQLAlchemy wraps the connection
    in a transaction (``journal_mode`` cannot be changed inside one).

    Note: under aiosqlite the object passed in is an
    ``AsyncAdapt_aiosqlite_connection`` adapter, not a raw
    ``sqlite3.Connection`` — its ``cursor()`` and ``cursor.execute()``
    quack the same so we use them directly without an isinstance check.
    """

    cursor = dbapi_connection.cursor()
    try:
        for pragma in _PRAGMAS:
            cursor.execute(f"PRAGMA {pragma};")
    finally:
        cursor.close()


def create_engine(db_url: str | None = None) -> AsyncEngine:
    """Build a fresh async engine with the WAL pragma hook attached.

    Most callers want :data:`engine` (the process-wide singleton). This
    factory exists so tests can spin up an in-memory engine with the same
    pragma machinery.
    """

    url = db_url or get_settings().db_url
    eng = create_async_engine(url, future=True)
    event.listen(eng.sync_engine, "connect", _apply_pragmas)
    return eng


engine: AsyncEngine = create_engine()
SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yield a session, close it after the request."""

    async with SessionLocal() as session:
        yield session
