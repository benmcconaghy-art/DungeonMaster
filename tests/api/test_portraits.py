"""Tests for the portrait endpoints (Step 7).

POST /api/characters/{id}/portrait and POST /api/npcs/{id}/portrait
both enqueue a job onto the shared Valkey queue. We swap the queue
client singleton for an in-memory fake so the test asserts on what
landed on the queue without needing a real Valkey.

The router lives at ``app.api.portraits``; auth and membership
checks reuse the same primitives as the rest of the API surface.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from httpx import AsyncClient

from app.images.portrait import reset_for_tests as reset_queue_client
from app.images.portrait import set_queue_client_for_tests

_VALID_PW = "correct horse battery staple"


class _FakeQueueClient:
    """Captures rpush calls; same shape as the redis async client
    surface that ``push_job`` actually uses."""

    def __init__(self) -> None:
        self.pushed: list[tuple[str, bytes]] = []

    async def rpush(self, key: str, value: bytes) -> int:
        self.pushed.append((key, value))
        return len(self.pushed)

    async def aclose(self) -> None:
        return None


@pytest.fixture
async def fake_queue() -> Any:
    """Yield a captured queue client + ensure cleanup runs even if a
    test raises. Restores the singleton between tests so the next one
    builds fresh."""

    fake = _FakeQueueClient()
    set_queue_client_for_tests(cast(Any, fake))
    try:
        yield fake
    finally:
        await reset_queue_client()


async def _register_and_login(client: AsyncClient, username: str = "alice") -> str:
    response = await client.post(
        "/api/auth/register",
        json={"username": username, "password": _VALID_PW},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]  # type: ignore[no-any-return]


async def _create_campaign(client: AsyncClient, name: str = "Borderlands") -> str:
    response = await client.post("/api/campaigns", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]  # type: ignore[no-any-return]


async def _create_character(client: AsyncClient, campaign_id: str) -> str:
    """Roll up a fixed character so portrait tests have a real PC id
    to address. Reuses the same shape the chargen endpoint expects."""

    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Brunhild",
            "race": "Human",
            "class_name": "Fighter",
            "alignment": "lawful",
            "method": "classic",
            "abilities": {
                "str": 14,
                "int": 10,
                "wis": 10,
                "dex": 12,
                "con": 12,
                "cha": 10,
            },
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# POST /api/characters/{id}/portrait
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_character_portrait_enqueues_job(
    client: AsyncClient, fake_queue: _FakeQueueClient
) -> None:
    """Happy path: campaign owner requests a portrait, gets 202 +
    image_id, and one job lands on the queue with subject_character_id
    pointing at the PC."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client)
    character_id = await _create_character(client, campaign_id)

    response = await client.post(
        f"/api/characters/{character_id}/portrait",
        json={},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["image_id"]
    assert "Brunhild" in body["prompt"]

    # One job pushed; the FK target is the character row.
    assert len(fake_queue.pushed) == 1
    payload = json.loads(fake_queue.pushed[0][1])
    assert payload["id"] == body["image_id"]
    assert payload["subject_character_id"] == character_id
    assert payload["subject_npc_id"] is None
    assert payload["kind"] == "npc"  # spec §8 portrait kind


@pytest.mark.asyncio
async def test_character_portrait_explicit_prompt_overrides_auto(
    client: AsyncClient, fake_queue: _FakeQueueClient
) -> None:
    """A caller-supplied prompt is used verbatim — no auto-composition.
    UI lets the player tune 'more rugged, scar across left eye' and
    expects the FLUX request to use that prompt directly."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client)
    character_id = await _create_character(client, campaign_id)

    response = await client.post(
        f"/api/characters/{character_id}/portrait",
        json={"prompt": "moody portrait, scar across left eye"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["prompt"] == "moody portrait, scar across left eye"

    payload = json.loads(fake_queue.pushed[0][1])
    assert payload["prompt"] == "moody portrait, scar across left eye"


@pytest.mark.asyncio
async def test_character_portrait_session_id_threads_through(
    client: AsyncClient, fake_queue: _FakeQueueClient
) -> None:
    """The optional session_id ends up on the queued job so the worker
    knows where to broadcast image_ready."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client)
    character_id = await _create_character(client, campaign_id)

    response = await client.post(
        f"/api/characters/{character_id}/portrait",
        json={"session_id": "sess-123"},
    )
    assert response.status_code == 202
    payload = json.loads(fake_queue.pushed[0][1])
    assert payload["session_id"] == "sess-123"


@pytest.mark.asyncio
async def test_character_portrait_404_for_unknown_id(
    client: AsyncClient, fake_queue: _FakeQueueClient
) -> None:
    await _register_and_login(client)
    response = await client.post(
        "/api/characters/does-not-exist/portrait",
        json={},
    )
    assert response.status_code == 404
    assert fake_queue.pushed == []


@pytest.mark.asyncio
async def test_character_portrait_403_for_non_member(
    client: AsyncClient, fake_queue: _FakeQueueClient
) -> None:
    """Authorisation: another logged-in user who isn't a member of
    the character's campaign gets 403, not 200 — so portrait requests
    are scoped to the campaign membership."""

    # Owner sets up campaign + character.
    await _register_and_login(client, "owner")
    campaign_id = await _create_campaign(client)
    character_id = await _create_character(client, campaign_id)

    # Switch to a fresh non-member (logout via cookie reset by re-registering).
    await client.post("/api/auth/logout")
    await _register_and_login(client, "intruder")

    response = await client.post(
        f"/api/characters/{character_id}/portrait",
        json={},
    )
    assert response.status_code == 403
    assert fake_queue.pushed == []


@pytest.mark.asyncio
async def test_character_portrait_requires_auth(
    client: AsyncClient, fake_queue: _FakeQueueClient
) -> None:
    """Anonymous request → 401. The endpoint is behind require_user
    via the Annotated dependency on every handler."""

    response = await client.post(
        "/api/characters/anything/portrait",
        json={},
    )
    assert response.status_code == 401
    assert fake_queue.pushed == []


# ---------------------------------------------------------------------------
# POST /api/npcs/{id}/portrait
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_npc_portrait_enqueues_job(client: AsyncClient, fake_queue: _FakeQueueClient) -> None:
    """NPC portrait endpoint mirrors the character one but links via
    subject_npc_id. We seed the NPC directly through the test DB
    fixture rather than via spawn_npc (which is exercised in the
    handler tests)."""

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.db.models import Npc
    from app.deps import get_db
    from app.main import app

    await _register_and_login(client)
    campaign_id = await _create_campaign(client)

    # Reach into the same overridden session factory the test fixture
    # set up so the NPC we insert is visible to the handler.
    db_factory = app.dependency_overrides[get_db]
    async for db in db_factory():
        npc = Npc(campaign_id=campaign_id, name="Castellan Thorvald")
        db.add(npc)
        await db.flush()
        await db.commit()
        npc_id = npc.id
        break
    assert isinstance(db, AsyncSession)

    response = await client.post(
        f"/api/npcs/{npc_id}/portrait",
        json={},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert "Castellan Thorvald" in body["prompt"]

    payload = json.loads(fake_queue.pushed[0][1])
    assert payload["subject_npc_id"] == npc_id
    assert payload["subject_character_id"] is None
