"""Phase 6 tests for the character-sheet detail + notes endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

_VALID_PW = "correct horse battery staple"


async def _register_and_login(client: AsyncClient, username: str = "alice") -> str:
    response = await client.post(
        "/api/auth/register",
        json={"username": username, "password": _VALID_PW},
    )
    assert response.status_code == 201
    return response.json()["id"]  # type: ignore[no-any-return]


async def _create_campaign(client: AsyncClient, name: str = "Test") -> str:
    response = await client.post("/api/campaigns", json={"name": name})
    assert response.status_code == 201
    return response.json()["id"]  # type: ignore[no-any-return]


async def _create_character(
    client: AsyncClient,
    campaign_id: str,
    *,
    name: str = "Vela",
    class_name: str = "Cleric",
) -> str:
    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": name,
            "race": "Human",
            "class_name": class_name,
            "alignment": "lawful",
            "abilities": {
                "str": 11,
                "int": 10,
                "wis": 15,
                "dex": 12,
                "con": 13,
                "cha": 14,
            },
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]  # type: ignore[no-any-return]


@pytest.mark.asyncio
async def test_get_character_returns_full_detail(client: AsyncClient) -> None:
    """The detail endpoint returns abilities-with-modifiers, saves
    resolved from class+level, status, and the editable notes
    (empty string by default)."""

    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Detail Test")
    character_id = await _create_character(client, campaign_id)

    response = await client.get(f"/api/characters/{character_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == character_id
    assert body["name"] == "Vela"
    assert body["status"] == "alive"
    assert body["is_mine"] is True
    assert body["is_spellcaster"] is True  # Cleric

    # Abilities carry computed modifiers — Wis 15 → +1.
    assert body["abilities"]["wis"]["score"] == 15
    assert body["abilities"]["wis"]["modifier"] == 1
    # Str 11 → 0 modifier in BFRPG.
    assert body["abilities"]["str"]["modifier"] == 0

    # Save table for Cleric Lvl 1 includes all five kinds.
    save_kinds = {s["kind"] for s in body["saves"]}
    assert "death_ray" in save_kinds
    assert "spells" in save_kinds
    # Targets are positive integers.
    assert all(isinstance(s["target"], int) and s["target"] > 0 for s in body["saves"])

    assert body["notes"] == ""
    assert body["inventory"] == []
    assert body["spells"] == []


@pytest.mark.asyncio
async def test_get_character_visible_to_table_member(client: AsyncClient) -> None:
    """A campaign member can see another player's character sheet —
    the table shares its information. Editing is gated separately."""

    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Shared Table")
    character_id = await _create_character(client, campaign_id, name="Alice's PC")
    invite_code = (await client.post(f"/api/campaigns/{campaign_id}/invite")).json()["code"]
    await client.post("/api/auth/logout")

    await _register_and_login(client, "bob")
    await client.post("/api/campaigns/join", json={"code": invite_code})

    response = await client.get(f"/api/characters/{character_id}")
    assert response.status_code == 200
    assert response.json()["is_mine"] is False


@pytest.mark.asyncio
async def test_get_character_rejects_non_member(client: AsyncClient) -> None:
    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Private Table")
    character_id = await _create_character(client, campaign_id)
    await client.post("/api/auth/logout")

    await _register_and_login(client, "bob")
    response = await client.get(f"/api/characters/{character_id}")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_character_404_when_unknown(client: AsyncClient) -> None:
    await _register_and_login(client, "alice")
    response = await client.get("/api/characters/nope")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_notes_owner_only(client: AsyncClient) -> None:
    """The owner can PATCH their notes; another table member can read
    the sheet but cannot edit notes — surfaces 403 cleanly."""

    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Notes Test")
    character_id = await _create_character(client, campaign_id)

    response = await client.patch(
        f"/api/characters/{character_id}/notes",
        json={"notes": "Brann owes me 4gp."},
    )
    assert response.status_code == 200
    assert response.json()["notes"] == "Brann owes me 4gp."

    # Re-read confirms persistence.
    detail = await client.get(f"/api/characters/{character_id}")
    assert detail.json()["notes"] == "Brann owes me 4gp."

    # Non-owner same-campaign user gets 403.
    invite_code = (await client.post(f"/api/campaigns/{campaign_id}/invite")).json()["code"]
    await client.post("/api/auth/logout")
    await _register_and_login(client, "bob")
    await client.post("/api/campaigns/join", json={"code": invite_code})

    forbidden = await client.patch(
        f"/api/characters/{character_id}/notes",
        json={"notes": "evil rewrite"},
    )
    assert forbidden.status_code == 403


@pytest.mark.asyncio
async def test_update_notes_rejects_oversize(client: AsyncClient) -> None:
    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Oversize Test")
    character_id = await _create_character(client, campaign_id)

    response = await client.patch(
        f"/api/characters/{character_id}/notes",
        json={"notes": "x" * 6000},
    )
    assert response.status_code == 422  # Pydantic max_length


@pytest.mark.asyncio
async def test_get_character_non_spellcaster_hides_spells(client: AsyncClient) -> None:
    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Fighter Test")
    character_id = await _create_character(client, campaign_id, name="Brann", class_name="Fighter")

    body = (await client.get(f"/api/characters/{character_id}")).json()
    assert body["is_spellcaster"] is False
    # Saves still resolve for Fighter.
    assert any(s["kind"] == "death_ray" for s in body["saves"])
