"""Pytest fixtures shared across the suite.

Phase 0 / 1 ships two fixtures:

* ``db_session`` — an in-memory SQLite ``AsyncSession`` for unit tests of
  pure DB code. The schema is created fresh per test from
  ``Base.metadata`` and the engine is disposed after.

* ``client`` — an httpx ASGI client where the FastAPI ``get_db``
  dependency is overridden to yield sessions from the same in-memory
  engine. Auth handlers, ``/health``, and any future router that uses
  ``get_db`` all see the test database.

In-memory SQLite cannot run in actual WAL mode (the journal mode silently
falls back to ``memory``) but the rest of the pragmas — ``foreign_keys``,
``busy_timeout``, etc. — apply, which is what catches the bugs unit tests
need to catch. Migration tests use a file-backed temp DB.
"""

from __future__ import annotations

import os

# Set sentinel settings before app modules load. Tests should never write
# to the production DB path or speak to real upstream services.
os.environ.setdefault("DB_PATH", ":memory-sentinel:")
os.environ.setdefault("SESSION_SECRET", "test-secret-not-used-but-min-length-ok")

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.base import Base
from app.db.session import create_engine
from app.deps import get_db
from app.main import app as fastapi_app


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Yield an ``AsyncSession`` against a fresh in-memory SQLite database.

    The schema (``Base.metadata``) is created at fixture setup. Phase 0
    has no models registered, so this currently materialises an empty
    schema; Phase 1 onward populates it.
    """

    engine = create_engine("sqlite+aiosqlite:///:memory:")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """HTTP client bound directly to the FastAPI app via ASGI transport.

    Overrides the ``get_db`` dependency so every handler — auth, /health,
    future routers — sees a fresh in-memory engine. ``pytest`` never
    touches the production SQLite file. Avoids spinning up a real server.
    """

    test_engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    test_factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with test_factory() as session:
            yield session

    fastapi_app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=fastapi_app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac
    finally:
        fastapi_app.dependency_overrides.pop(get_db, None)
        await test_engine.dispose()
