"""End-to-end multiplayer smoke against the real stack.

One simulated WebSocket client in a session against the real Nemotron
endpoint and the real Valkey instance. Validates that Phase 4's WS
hub orchestrator broadcast path round-trips through Valkey:

  WS pc_action → take_turn → DmEvent → bridge → ws.ServerMessage →
  Pubsub.publish → channel session:{id} → Pubsub.subscribe →
  WS forward → client receive

If any link breaks, the test fails. This is the integration smoke;
the deterministic per-step coverage (multi-client broadcast, whisper
isolation, initiative gating, reconnect) lives in
``tests/api/test_ws.py`` against an in-memory FakePubsub.

Why single-client and not multi-client here: FastAPI's :class:`TestClient`
runs each instance in its own anyio :class:`~anyio.from_thread.BlockingPortal`
which owns its own event loop. The Pubsub singleton's redis-py async
client is bound to whichever loop first built it; two TestClient
instances cannot share that singleton without a "got Future attached
to a different loop" error. Spinning up a real uvicorn server in a
thread and connecting via the ``websockets`` library would work, but
is heavy for the smoke we need. The unit suite already covers the
multi-client semantics with a deterministic in-memory pubsub.

Run with::

    uv run pytest -m integration tests/integration/test_multiplayer.py

Skipped if vLLM or Valkey is unreachable.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.db import models
from app.db.base import Base
from app.db.session import create_engine
from app.deps import get_db
from app.main import app as fastapi_app
from app.realtime import presence as presence_module
from app.realtime import pubsub as pubsub_module

pytestmark = pytest.mark.integration


_VALID_PW = "correct horse battery staple"
_PER_TURN_TIMEOUT_S = 90.0


@pytest.fixture(scope="module")
def vllm_reachable() -> bool:
    settings = get_settings()
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as c:
            response = c.get(f"{settings.vllm_base_url}/v1/models")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"vLLM unreachable at {settings.vllm_base_url}: {exc}")
    return True


@pytest.fixture(scope="module")
def valkey_reachable() -> bool:
    settings = get_settings()

    async def _ping() -> None:
        from app.realtime.pubsub import Pubsub

        instance = Pubsub(url=settings.redis_url)
        try:
            await instance.health()
        finally:
            await instance.aclose()

    try:
        asyncio.run(_ping())
    except Exception as exc:  # DmPubsubError or transport
        pytest.skip(f"Valkey unreachable at {settings.redis_url}: {exc}")
    return True


@pytest.fixture
def integration_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, async_sessionmaker[AsyncSession]]]:
    """File-backed SQLite, real Valkey, real vLLM. Single TestClient
    so the WS handler and the orchestrator share the TestClient's
    event loop with the lazily-built Pubsub singleton.
    """

    db_path = tmp_path / "multiplayer.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    monkeypatch.setattr("app.api.ws.SessionLocal", factory)
    monkeypatch.setattr("app.orchestrator.dm.SessionLocal", factory)

    # Drop any singleton from a previous test so this run gets a fresh
    # Pubsub bound to the current TestClient's loop.
    asyncio.run(pubsub_module.reset_for_tests())
    presence_module.reset_for_tests()

    client = TestClient(fastapi_app)
    try:
        yield client, factory
    finally:
        fastapi_app.dependency_overrides.pop(get_db, None)
        asyncio.run(pubsub_module.reset_for_tests())
        presence_module.reset_for_tests()
        asyncio.run(engine.dispose())


def _register(client: TestClient, *, username: str) -> dict[str, Any]:
    response = client.post(
        "/api/auth/register",
        json={"username": username, "password": _VALID_PW},
    )
    assert response.status_code == 201, response.text
    return response.json()  # type: ignore[no-any-return]


def _create_campaign(client: TestClient) -> dict[str, Any]:
    r = client.post("/api/campaigns", json={"name": "Multi"})
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


def _create_character(client: TestClient, campaign_id: str, name: str) -> dict[str, Any]:
    r = client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": name,
            "race": "Human",
            "class_name": "Fighter",
            "alignment": "neutral",
            "abilities": {
                "str": 14,
                "int": 10,
                "wis": 10,
                "dex": 12,
                "con": 14,
                "cha": 10,
            },
        },
    )
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


def _create_session(client: TestClient, campaign_id: str) -> dict[str, Any]:
    r = client.post(f"/api/campaigns/{campaign_id}/sessions")
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


def _drain_until(
    ws: Any,
    expected: str,
    *,
    max_frames: int = 50,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Drain frames until ``expected`` appears."""

    deadline = time.monotonic() + timeout
    for _ in range(max_frames):
        if time.monotonic() > deadline:
            raise AssertionError(f"timeout waiting for frame type={expected!r}")
        raw = ws.receive_text()
        parsed = json.loads(raw)
        if parsed.get("type") == expected:
            return parsed  # type: ignore[no-any-return]
    raise AssertionError(f"never received frame type={expected!r} after {max_frames} frames")


def _drain_any(
    ws: Any,
    *,
    accept: set[str],
    max_frames: int = 100,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Drain frames until one whose type is in ``accept`` arrives."""

    deadline = time.monotonic() + timeout
    for _ in range(max_frames):
        if time.monotonic() > deadline:
            raise AssertionError(f"timeout waiting for one of {accept!r}")
        raw = ws.receive_text()
        parsed = json.loads(raw)
        if parsed.get("type") in accept:
            return parsed  # type: ignore[no-any-return]
    raise AssertionError(f"none of {accept!r} arrived within {max_frames} frames")


def test_real_nemotron_via_ws_round_trips_through_valkey(
    integration_app: tuple[TestClient, async_sessionmaker[AsyncSession]],
    vllm_reachable: bool,
    valkey_reachable: bool,
) -> None:
    """Player submits a non-combat ``pc_action`` over WebSocket; the
    real orchestrator runs against real Nemotron, broadcasts each
    event through real Valkey, and the same WS connection receives at
    least one ``narration_chunk`` followed by a terminal frame
    (``narration_complete`` on the happy path or ``dm_error`` with one
    of the documented Phase-2 reasons).

    Asserts:
      - Snapshot delivered on connect.
      - pc_action echo reaches the originating client (Valkey doesn't
        filter by origin; the client de-dupes).
      - At least one narration_chunk OR a typed terminal frame arrives
        within ``_PER_TURN_TIMEOUT_S`` (Nemotron has run-to-run variance
        but always responds eventually unless something is broken).
      - The player message persists in ``session_messages`` regardless
        of how the turn ended (it's persisted before the LLM is hit).
    """

    client, factory = integration_app

    _register(client, username="alice")
    campaign = _create_campaign(client)
    char = _create_character(client, campaign["id"], name="Tav")
    session = _create_session(client, campaign["id"])

    with client.websocket_connect(f"/ws/session/{session['id']}") as ws:
        snapshot = _drain_until(ws, "snapshot")
        assert snapshot["session_id"] == session["id"]

        ws.send_text(
            json.dumps(
                {
                    "type": "pc_action",
                    "character_id": char["id"],
                    "content": ("I take in the keep's main courtyard. What stands out?"),
                    "kind": "other",
                }
            )
        )

        echo = _drain_until(ws, "pc_action")
        assert echo["content"].startswith("I take")

        # Wait for at least one orchestrator-emitted frame: a chunk, a
        # complete, or a typed error.
        accept = {"narration_chunk", "narration_complete", "dm_error"}
        first = _drain_any(ws, accept=accept, timeout=_PER_TURN_TIMEOUT_S)
        assert first["type"] in accept
        if first["type"] == "dm_error":
            # Documented Phase-2 outcomes are still acceptable here —
            # the test's purpose is round-tripping, not orchestrator
            # quality. iteration_cap and empty_completion are the two
            # known typed errors a clean stream can produce.
            assert first["reason"] in {
                "iteration_cap",
                "empty_completion",
                "stream_error",
                "stream_failed",
                "runaway_token",
            }, f"unexpected dm_error reason: {first['reason']!r}"

    async def _check() -> None:
        async with factory() as db:
            from sqlalchemy import select

            rows = list(
                (
                    await db.scalars(
                        select(models.SessionMessage)
                        .where(models.SessionMessage.session_id == session["id"])
                        .where(models.SessionMessage.sender_kind == "player")
                    )
                ).all()
            )
            assert rows, "no player message persisted"
            assert "courtyard" in rows[0].content

    asyncio.run(_check())
