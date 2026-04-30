"""Pytest fixtures shared across the suite.

Phase 0 ships a single fixture (``db_session``) modelled on the sketch in
``.claude/agents/test-writer.md``: an in-memory SQLite engine wired with
the same WAL pragma hook the production engine uses, the schema created
fresh for each test, and an ``AsyncSession`` yielded for the duration.

In-memory SQLite cannot run in actual WAL mode (the journal mode silently
falls back to ``memory``) but the rest of the pragmas — ``foreign_keys``,
``busy_timeout``, etc. — apply, which is what catches the bugs unit tests
need to catch. Migration tests will use a file-backed DB when they land.

The HTTP ``client`` fixture replaces the app's ``SessionLocal`` dependency
with one bound to the in-memory engine for the duration of a test, so
running ``pytest`` from a clean checkout never touches the production
``/var/lib/dungeon-master/dm.db`` path.
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
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    """HTTP client bound directly to the FastAPI app via ASGI transport.

    Swaps the app's production ``SessionLocal`` for a sessionmaker bound
    to an in-memory engine, so ``pytest`` never touches the production
    SQLite file. Avoids spinning up a real server.
    """

    test_engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    test_factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)
    monkeypatch.setattr("app.main.SessionLocal", test_factory)

    transport = ASGITransport(app=fastapi_app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac
    finally:
        await test_engine.dispose()
