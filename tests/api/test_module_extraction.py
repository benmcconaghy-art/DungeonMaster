"""Integration tests for POST /api/sessions/{id}/extract-module.

Tests the extraction pipeline: prompt building, LLM call (mocked),
JSON validation, retry on ValidationError, and Module row insertion.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.base import Base
from app.db.models import Campaign, CampaignMember, Module
from app.db.models import Session as DmSession
from app.db.session import create_engine
from app.deps import get_db
from app.main import app as fastapi_app

_VALID_PW = "correct horse battery staple"

_VALID_MODULE_JSON: dict[str, Any] = {
    "format_version": "1.0",
    "synopsis": "A band of goblins troubles a border keep.",
    "tone": "gritty",
    "image_style": "dark fantasy",
    "image_negative_prompt": "modern objects",
    "level_range": [1, 3],
    "estimated_sessions": 4,
    "starting_hook": "The party is hired to investigate goblin raids.",
    "starting_location_symbol": "loc_keep",
    "locations": [
        {"symbol": "loc_keep", "name": "The Keep", "description": "A stone border fortress."}
    ],
    "npcs": [
        {
            "symbol": "npc_commander",
            "name": "Commander Aldric",
            "description": "Battle-scarred veteran.",
            "motivation": "Defend the keep.",
            "starting_location_symbol": "loc_keep",
        }
    ],
    "encounters": [],
    "plot_beats": [
        {
            "symbol": "beat_arrival",
            "title": "Arrival",
            "trigger_hint": "When the party arrives at the keep.",
            "outcome": "Party is hired.",
        }
    ],
    "secrets": [],
    "endings": [
        {
            "symbol": "end_victory",
            "trigger": "Goblins defeated.",
            "outcome": "Keep is safe.",
        }
    ],
    "world_facts": [
        {"fact": "The keep guards the northern pass.", "tags": ["keep"], "importance": 6}
    ],
}


@pytest_asyncio.fixture
async def client_db(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncClient, AsyncSession]]:
    """Shared-engine client + db fixture."""
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    monkeypatch.setattr("app.orchestrator.dm.SessionLocal", factory)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as s:
            yield s

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=fastapi_app)
    async with factory() as db:
        try:
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                yield ac, db
        finally:
            fastapi_app.dependency_overrides.pop(get_db, None)

    await engine.dispose()


async def _setup_ended_session(
    client: AsyncClient, db: AsyncSession
) -> tuple[str, str, str]:
    """Register alice, create a campaign + ended session. Returns (user_id, campaign_id, session_id)."""

    resp = await client.post(
        "/api/auth/register", json={"username": "alice", "password": _VALID_PW}
    )
    user_id = resp.json()["id"]

    campaign = Campaign(name="Test Campaign", owner_id=user_id)
    db.add(campaign)
    await db.flush()

    db.add(CampaignMember(campaign_id=campaign.id, user_id=user_id, role="owner"))
    await db.flush()

    session = DmSession(campaign_id=campaign.id, ended_at="2026-05-01T12:00:00.000Z")
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return user_id, campaign.id, session.id


@pytest.mark.asyncio
async def test_extract_module_happy_path(
    client_db: tuple[AsyncClient, AsyncSession],
) -> None:
    """LLM returns valid JSON; module row is inserted."""

    client, db = client_db
    _, _, session_id = await _setup_ended_session(client, db)

    mock_complete = AsyncMock(return_value=json.dumps(_VALID_MODULE_JSON))
    with patch("app.llm.client.DmClient.complete", mock_complete):
        resp = await client.post(f"/api/sessions/{session_id}/extract-module")

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "module_id" in body
    assert body["synopsis"] == _VALID_MODULE_JSON["synopsis"]

    # Verify the Module row was inserted.
    module = await db.get(Module, body["module_id"])
    assert module is not None
    assert module.source_session_id == session_id
    assert module.public is False
    assert module.content["synopsis"] == _VALID_MODULE_JSON["synopsis"]


@pytest.mark.asyncio
async def test_extract_module_retries_on_invalid_json(
    client_db: tuple[AsyncClient, AsyncSession],
) -> None:
    """First LLM response is malformed; second is valid. Module is created on retry."""

    client, db = client_db
    _, _, session_id = await _setup_ended_session(client, db)

    calls = 0

    async def _flaky(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "this is not valid json {"
        return json.dumps(_VALID_MODULE_JSON)

    with patch("app.llm.client.DmClient.complete", side_effect=_flaky):
        resp = await client.post(f"/api/sessions/{session_id}/extract-module")

    assert resp.status_code == 201, resp.text
    assert calls == 2  # one failure, one success

    module = await db.get(Module, resp.json()["module_id"])
    assert module is not None


@pytest.mark.asyncio
async def test_extract_module_all_retries_exhausted(
    client_db: tuple[AsyncClient, AsyncSession],
) -> None:
    """All 3 retries return invalid JSON; endpoint returns 422."""

    client, db = client_db
    _, _, session_id = await _setup_ended_session(client, db)

    mock_complete = AsyncMock(return_value="not json at all")
    with patch("app.llm.client.DmClient.complete", mock_complete):
        resp = await client.post(f"/api/sessions/{session_id}/extract-module")

    assert resp.status_code == 422
    assert mock_complete.call_count == 3  # all retries attempted

    # No module row should have been inserted.
    mods = list((await db.scalars(select(Module))).all())
    assert mods == []


@pytest.mark.asyncio
async def test_extract_module_requires_ended_session(
    client_db: tuple[AsyncClient, AsyncSession],
) -> None:
    """Active (not ended) sessions return 422."""

    client, db = client_db
    resp = await client.post(
        "/api/auth/register", json={"username": "alice", "password": _VALID_PW}
    )
    user_id = resp.json()["id"]

    campaign = Campaign(name="Active Campaign", owner_id=user_id)
    db.add(campaign)
    await db.flush()
    db.add(CampaignMember(campaign_id=campaign.id, user_id=user_id, role="owner"))
    await db.flush()

    # Active session: no ended_at.
    session = DmSession(campaign_id=campaign.id)
    db.add(session)
    await db.commit()
    await db.refresh(session)

    resp = await client.post(f"/api/sessions/{session.id}/extract-module")
    assert resp.status_code == 422
    assert "ended" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_extract_module_requires_owner(
    client_db: tuple[AsyncClient, AsyncSession],
) -> None:
    """Non-owner members get 403."""

    client, db = client_db

    # Register alice as owner.
    await client.post(
        "/api/auth/register", json={"username": "alice", "password": _VALID_PW}
    )
    owner_id = (await client.post("/api/auth/login", data={"username": "alice", "password": _VALID_PW})).json().get("id") or ""

    # Use db to get the actual owner id.
    from app.db.models import User
    alice = (await db.scalars(select(User).where(User.username == "alice"))).first()
    assert alice is not None
    owner_id = alice.id

    campaign = Campaign(name="Owner Campaign", owner_id=owner_id)
    db.add(campaign)
    await db.flush()
    db.add(CampaignMember(campaign_id=campaign.id, user_id=owner_id, role="owner"))
    await db.flush()

    session = DmSession(campaign_id=campaign.id, ended_at="2026-05-01T12:00:00.000Z")
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # Register bob and add as player.
    await client.post("/api/auth/logout")
    await client.post(
        "/api/auth/register", json={"username": "bob", "password": _VALID_PW}
    )
    bob = (await db.scalars(select(User).where(User.username == "bob"))).first()
    assert bob is not None
    db.add(CampaignMember(campaign_id=campaign.id, user_id=bob.id, role="player"))
    await db.commit()

    resp = await client.post(f"/api/sessions/{session.id}/extract-module")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_extract_module_requires_auth(client: AsyncClient) -> None:
    resp = await client.post("/api/sessions/any-id/extract-module")
    assert resp.status_code == 401
