"""Tests for campaign + character + session CRUD endpoints (Phase 2)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

_VALID_PW = "correct horse battery staple"


async def _register_and_login(client: AsyncClient, username: str = "alice") -> str:
    """Helper: register a fresh user, return the user id."""

    response = await client.post(
        "/api/auth/register",
        json={"username": username, "password": _VALID_PW},
    )
    assert response.status_code == 201
    return response.json()["id"]  # type: ignore[no-any-return]


async def _create_campaign(client: AsyncClient, name: str = "Borderlands") -> str:
    response = await client.post("/api/campaigns", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]  # type: ignore[no-any-return]


@pytest.mark.asyncio
async def test_create_campaign_authenticated(client: AsyncClient) -> None:
    await _register_and_login(client)
    campaign_id = await _create_campaign(client, "The Borderlands")
    assert campaign_id


@pytest.mark.asyncio
async def test_create_campaign_requires_auth(client: AsyncClient) -> None:
    response = await client.post("/api/campaigns", json={"name": "x"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_roll_up_fighter(client: AsyncClient) -> None:
    """The full chargen path the integration test relies on."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client)
    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Borin Stoneforge",
            "race": "Dwarf",
            "class_name": "Fighter",
            "alignment": "lawful",
            "method": "classic",
            "seed": 42,
            "abilities": {
                "str": 16,
                "int": 10,
                "wis": 12,
                "dex": 12,
                "con": 14,
                "cha": 9,
            },
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["name"] == "Borin Stoneforge"
    assert body["race"] == "Dwarf"
    assert body["class_name"] == "Fighter"
    assert body["level"] == 1
    assert body["hp_current"] == body["hp_max"]  # Phase 2 starts at full HP
    assert body["str_score"] == 16


@pytest.mark.asyncio
async def test_roll_character_rejects_incompatible_race_class(client: AsyncClient) -> None:
    await _register_and_login(client)
    campaign_id = await _create_campaign(client)
    # Dwarves can't be Magic-Users in BFRPG core.
    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Bad Take",
            "race": "Dwarf",
            "class_name": "Magic-User",
            "alignment": "lawful",
            "abilities": {
                "str": 12,
                "int": 12,
                "wis": 10,
                "dex": 10,
                "con": 14,
                "cha": 9,
            },
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_roll_character_requires_membership(client: AsyncClient) -> None:
    """User A creates a campaign; user B can't roll a character into it."""

    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Alice's Game")
    await client.post("/api/auth/logout")

    await _register_and_login(client, "bob")
    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Sneaky",
            "race": "Human",
            "class_name": "Thief",
            "alignment": "neutral",
        },
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_and_end_session(client: AsyncClient) -> None:
    await _register_and_login(client)
    campaign_id = await _create_campaign(client)

    create = await client.post(f"/api/campaigns/{campaign_id}/sessions")
    assert create.status_code == 201, create.text
    snapshot = create.json()
    assert snapshot["is_active"] is True
    assert snapshot["ended_at"] is None
    session_id = snapshot["id"]

    snap_get = await client.get(f"/api/sessions/{session_id}")
    assert snap_get.status_code == 200
    assert snap_get.json()["id"] == session_id

    end = await client.post(f"/api/sessions/{session_id}/end")
    assert end.status_code == 200
    ended = end.json()
    assert ended["is_active"] is False
    assert ended["ended_at"] is not None

    # Idempotent — calling end again is a no-op.
    end_again = await client.post(f"/api/sessions/{session_id}/end")
    assert end_again.status_code == 200


@pytest.mark.asyncio
async def test_list_messages_empty_session(client: AsyncClient) -> None:
    await _register_and_login(client)
    campaign_id = await _create_campaign(client)
    session = (await client.post(f"/api/campaigns/{campaign_id}/sessions")).json()
    response = await client.get(f"/api/sessions/{session['id']}/messages")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_session_membership_enforced(client: AsyncClient) -> None:
    """A user not in the campaign can't see the session at all."""

    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Alice's Game")
    session = (await client.post(f"/api/campaigns/{campaign_id}/sessions")).json()
    await client.post("/api/auth/logout")

    await _register_and_login(client, "bob")
    response = await client.get(f"/api/sessions/{session['id']}")
    assert response.status_code == 403
