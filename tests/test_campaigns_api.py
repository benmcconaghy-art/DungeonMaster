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


# ---------------------------------------------------------------------------
# Phase 6: dashboard endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_campaigns_returns_member_campaigns(client: AsyncClient) -> None:
    """``GET /api/campaigns`` returns campaigns the user belongs to —
    both as owner and as a redeemed-invite player."""

    await _register_and_login(client, "alice")
    own_id = await _create_campaign(client, "Alice's Hold")
    listed = await client.get("/api/campaigns")
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    assert rows[0]["id"] == own_id
    assert rows[0]["owner_id"]
    # No sessions yet → no last_played, not active.
    assert rows[0]["last_played_at"] is None
    assert rows[0]["has_active_session"] is False


@pytest.mark.asyncio
async def test_list_campaigns_requires_auth(client: AsyncClient) -> None:
    response = await client.get("/api/campaigns")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_campaigns_orders_by_last_played(client: AsyncClient) -> None:
    """Campaigns with the most recent session activity sort first.
    The first row is flagged ``most_recent=True``; never-played
    campaigns sort to the end with ``most_recent=False``."""

    await _register_and_login(client, "alice")
    older = await _create_campaign(client, "Older Camp")
    newer = await _create_campaign(client, "Newer Camp")
    quiet = await _create_campaign(client, "Quiet Camp")

    # Older: a closed session.
    s1 = (await client.post(f"/api/campaigns/{older}/sessions")).json()
    await client.post(f"/api/sessions/{s1['id']}/end")
    # Newer: a more recent active session.
    await client.post(f"/api/campaigns/{newer}/sessions")
    # Quiet: no sessions.

    rows = (await client.get("/api/campaigns")).json()
    ids_in_order = [r["id"] for r in rows]
    # Newer-active first, older-closed second, quiet last.
    assert ids_in_order[0] == newer
    assert ids_in_order[1] == older
    assert ids_in_order[2] == quiet
    assert rows[0]["most_recent"] is True
    assert rows[1]["most_recent"] is False
    assert rows[2]["most_recent"] is False
    assert rows[0]["has_active_session"] is True
    assert rows[1]["has_active_session"] is False


@pytest.mark.asyncio
async def test_get_campaign_includes_characters_and_members(
    client: AsyncClient,
) -> None:
    """``GET /api/campaigns/{id}`` returns the full detail composite
    the dashboard card needs."""

    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Detail Test")
    await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Vela",
            "race": "Human",
            "class_name": "Cleric",
            "alignment": "lawful",
            "abilities": {"str": 11, "int": 10, "wis": 15, "dex": 12, "con": 13, "cha": 14},
        },
    )
    await client.post(f"/api/campaigns/{campaign_id}/sessions")

    detail = (await client.get(f"/api/campaigns/{campaign_id}")).json()
    assert detail["id"] == campaign_id
    assert detail["active_session_id"]  # the active session shows up
    assert any(m["username"] == "alice" for m in detail["members"])
    members_with_classes = [m for m in detail["members"] if m["character_classes"]]
    assert any("Cleric" in m["character_classes"] for m in members_with_classes)
    assert len(detail["characters"]) == 1
    assert detail["characters"][0]["name"] == "Vela"
    assert detail["characters"][0]["is_mine"] is True
    assert len(detail["recent_sessions"]) == 1


@pytest.mark.asyncio
async def test_get_campaign_rejects_non_member(client: AsyncClient) -> None:
    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Private")
    await client.post("/api/auth/logout")

    await _register_and_login(client, "bob")
    response = await client.get(f"/api/campaigns/{campaign_id}")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_campaign_404_when_unknown(client: AsyncClient) -> None:
    await _register_and_login(client, "alice")
    response = await client.get("/api/campaigns/does-not-exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_invite_then_join_round_trip(client: AsyncClient) -> None:
    """Owner mints an invite; a different user redeems it and becomes
    a player member."""

    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Invite Test")

    invite = await client.post(f"/api/campaigns/{campaign_id}/invite")
    assert invite.status_code == 201
    code = invite.json()["code"]
    assert code
    assert invite.json()["expires_in_seconds"] >= 24 * 3600
    # Phase 7: invite_id is exposed so a UI can drive a manage-invites view.
    assert "invite_id" in invite.json()

    await client.post("/api/auth/logout")
    await _register_and_login(client, "bob")

    join = await client.post("/api/campaigns/join", json={"code": code})
    assert join.status_code == 200
    assert join.json()["campaign_id"] == campaign_id

    # Bob now sees the campaign in his list.
    listed = (await client.get("/api/campaigns")).json()
    assert any(row["id"] == campaign_id for row in listed)

    # Phase 7: invites are single-use. The same code re-presented after
    # a successful redeem returns 400, not the Phase 6 idempotent 200 —
    # the audit row tracks who consumed the code.
    re_redeem = await client.post("/api/campaigns/join", json={"code": code})
    assert re_redeem.status_code == 400
    assert "already been used" in re_redeem.json()["detail"]


@pytest.mark.asyncio
async def test_invite_owner_only(client: AsyncClient) -> None:
    """A non-owner member cannot mint invites — owner privilege only."""

    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Owner-Only Invite")
    invite = await client.post(f"/api/campaigns/{campaign_id}/invite")
    code = invite.json()["code"]

    await client.post("/api/auth/logout")
    await _register_and_login(client, "bob")
    await client.post("/api/campaigns/join", json={"code": code})

    # Bob is now a player; minting should fail.
    bob_invite = await client.post(f"/api/campaigns/{campaign_id}/invite")
    assert bob_invite.status_code == 403


@pytest.mark.asyncio
async def test_invite_rejects_garbage(client: AsyncClient) -> None:
    await _register_and_login(client, "bob")
    response = await client.post("/api/campaigns/join", json={"code": "not-a-real-token"})
    assert response.status_code == 400
