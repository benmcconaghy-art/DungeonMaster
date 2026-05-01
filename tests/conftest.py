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
from app.ratelimit import reset_for_tests as _reset_ratelimit


@pytest.fixture(autouse=True)
def _fresh_ratelimit_counters() -> None:
    """Reset rate-limit counters before every test.

    The MemoryStorage / FixedWindowRateLimiter are module singletons —
    without this fixture, a test that hits ``/api/auth/login`` 50 times
    would leave a populated counter that the next test inherits (the
    testserver IP is the same for every case). Resetting per test
    keeps the limit window deterministic per case and prevents bleed.
    """

    _reset_ratelimit()


@pytest.fixture(autouse=True)
def _stub_opening_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 6.8 Bug 3: ``create_session`` schedules an opening DM turn
    as a background task that calls ``run_dm_turn`` (vLLM + pubsub).
    For tests that don't care about that path, stub ``run_dm_turn`` to
    a no-op coroutine so the scheduled task fires and resolves
    immediately without real network I/O. Tests that *do* exercise
    the auto-greeting wiring patch this attribute to their own
    capture (the per-test patch wins over this autouse one).

    Autouse so every fixture topology — the ``client`` fixture, the
    WS-test ``ws_setup`` fixture, integration scaffolding — picks it
    up without each having to remember.
    """

    async def _stub_run_dm_turn(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr("app.api.sessions.run_dm_turn", _stub_run_dm_turn)


@pytest.fixture
async def db_session(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncSession]:
    """Yield an ``AsyncSession`` against a fresh in-memory SQLite database.

    The schema (``Base.metadata``) is created at fixture setup.
    Monkey-patches ``app.orchestrator.dm.SessionLocal`` to the same
    in-memory factory so the orchestrator's fire-and-forget post-turn
    tasks (fact extractor, session-summary regen) — which open their own
    sessions via ``SessionLocal()`` rather than the one yielded here —
    don't try to write to the production DB path during a unit test.
    """

    engine = create_engine("sqlite+aiosqlite:///:memory:")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    monkeypatch.setattr("app.orchestrator.dm.SessionLocal", factory)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    """HTTP client bound directly to the FastAPI app via ASGI transport.

    Overrides the ``get_db`` dependency so every handler — auth, /health,
    future routers — sees a fresh in-memory engine. ``pytest`` never
    touches the production SQLite file.
    """

    test_engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    test_factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with test_factory() as session:
            yield session

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    # The orchestrator's post-turn fact-extractor / session-summary tasks
    # open their own session via SessionLocal(); send them at the test
    # engine too so they don't create stray ``:memory-sentinel:`` files.
    # The background tasks fire-and-forget; tests don't await them, but
    # we still want them pointing at a real schema in case they're
    # observed by a future test.
    monkeypatch.setattr("app.orchestrator.dm.SessionLocal", test_factory)

    transport = ASGITransport(app=fastapi_app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac
    finally:
        fastapi_app.dependency_overrides.pop(get_db, None)
        await test_engine.dispose()
