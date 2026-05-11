"""Tests for the Phase 6.5 chargen UI surface.

Covers:

- ``POST /api/chargen/roll-abilities`` — auth, validation, seed
  determinism, classic vs heroic distribution range.
- ``GET /campaigns/{id}/chargen`` — membership gating and the page
  shell renders with rolled abilities + race + class data.
- A smoke walk that mirrors what the page does: roll → commit via
  the existing characters endpoint → land on the sheet route.
"""

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


# ---------------------------------------------------------------------------
# POST /api/chargen/roll-abilities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_roll_abilities_requires_auth(client: AsyncClient) -> None:
    response = await client.post("/api/chargen/roll-abilities", json={})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_roll_abilities_classic_returns_six_scores(client: AsyncClient) -> None:
    await _register_and_login(client)

    response = await client.post("/api/chargen/roll-abilities", json={"method": "classic"})
    assert response.status_code == 200
    body = response.json()
    abilities = body["abilities"]
    assert set(abilities) == {"str", "int", "wis", "dex", "con", "cha"}
    for key, entry in abilities.items():
        assert 3 <= entry["score"] <= 18, f"{key} out of 3d6 range: {entry['score']}"
        # Modifier mirrors BFRPG ability_modifier — at score 3 it's -3, at 18 it's +3.
        assert -3 <= entry["modifier"] <= 3


@pytest.mark.asyncio
async def test_roll_abilities_heroic_returns_six_scores(client: AsyncClient) -> None:
    await _register_and_login(client)

    response = await client.post("/api/chargen/roll-abilities", json={"method": "heroic"})
    assert response.status_code == 200
    abilities = response.json()["abilities"]
    for entry in abilities.values():
        assert 3 <= entry["score"] <= 18


@pytest.mark.asyncio
async def test_roll_abilities_seeded_is_deterministic(client: AsyncClient) -> None:
    """Same seed + method → same roll. Lets us reason about test
    fixtures without having to monkeypatch the engine."""

    await _register_and_login(client)

    first = (
        await client.post("/api/chargen/roll-abilities", json={"method": "classic", "seed": 42})
    ).json()
    second = (
        await client.post("/api/chargen/roll-abilities", json={"method": "classic", "seed": 42})
    ).json()
    assert first == second


@pytest.mark.asyncio
async def test_roll_abilities_default_method_is_classic(client: AsyncClient) -> None:
    """Omitting ``method`` defaults to classic — no validation error."""

    await _register_and_login(client)
    response = await client.post("/api/chargen/roll-abilities", json={})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_roll_abilities_unknown_method_rejected(client: AsyncClient) -> None:
    await _register_and_login(client)
    response = await client.post("/api/chargen/roll-abilities", json={"method": "point-buy"})
    # Pydantic Literal validation → 422.
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /campaigns/{id}/chargen
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chargen_view_renders_for_member(client: AsyncClient) -> None:
    await _register_and_login(client)
    campaign_id = await _create_campaign(client)

    response = await client.get(f"/campaigns/{campaign_id}/chargen")
    assert response.status_code == 200
    body = response.text
    # Banner copy and the four core BFRPG races render into the page.
    assert "Roll a new character" in body or "A new traveller" in body
    for race in ("Human", "Dwarf", "Elf", "Halfling"):
        assert race in body
    for cls in ("Fighter", "Cleric", "Magic-User", "Thief"):
        assert cls in body
    # The injected JSON island carries the campaign id so the commit
    # POST goes to the right scope.
    assert campaign_id in body


@pytest.mark.asyncio
async def test_chargen_view_rejects_non_member(client: AsyncClient) -> None:
    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Alice's table")
    await client.post("/api/auth/logout")

    await _register_and_login(client, "mallory")
    response = await client.get(f"/campaigns/{campaign_id}/chargen")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_chargen_view_404_for_missing_campaign(client: AsyncClient) -> None:
    await _register_and_login(client)
    response = await client.get("/campaigns/does-not-exist/chargen")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Smoke: roll → commit → sheet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_roll_then_commit_creates_character(client: AsyncClient) -> None:
    """Mirrors the form's lifecycle: hit the chargen page, ask for a
    fresh roll, then post the assembled payload to the existing
    create endpoint. No new commit endpoint — the chargen UI reuses
    POST /api/campaigns/{id}/characters."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client, "Smoke")

    # Page shell loads.
    page = await client.get(f"/campaigns/{campaign_id}/chargen")
    assert page.status_code == 200

    # Roll a known set so we can pick a race that satisfies the
    # requirements (Dwarf needs CON 9; this set has it).
    rolled = await client.post("/api/chargen/roll-abilities", json={"method": "classic", "seed": 7})
    assert rolled.status_code == 200
    abilities = {k: v["score"] for k, v in rolled.json()["abilities"].items()}

    # Commit through the existing endpoint with abilities the player
    # decided to keep. Use Human + Fighter so we don't fight a seeded
    # ability requirement on a different race.
    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Borin Stoneward",
            "race": "Human",
            "class_name": "Fighter",
            "alignment": "lawful",
            "abilities": abilities,
        },
    )
    assert response.status_code == 201, response.text
    character = response.json()
    assert character["name"] == "Borin Stoneward"
    assert character["race"] == "Human"
    assert character["class_name"] == "Fighter"

    # The sheet view route resolves and serves a 200.
    sheet = await client.get(f"/characters/{character['id']}")
    assert sheet.status_code == 200


# ---------------------------------------------------------------------------
# Phase 6.13: pronouns + description on chargen
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chargen_accepts_pronouns_and_description(client: AsyncClient) -> None:
    """POST /api/campaigns/{id}/characters with pronouns and description →
    201, both fields echoed back in the response."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client, "Presentation Test")

    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Mira Ashvale",
            "race": "Human",
            "class_name": "Magic-User",
            "alignment": "neutral",
            "abilities": {
                "str": 9,
                "int": 15,
                "wis": 12,
                "dex": 13,
                "con": 10,
                "cha": 11,
            },
            "pronouns": "she/her",
            "description": "Dark braided hair",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["pronouns"] == "she/her"
    assert body["description"] == "Dark braided hair"


@pytest.mark.asyncio
async def test_chargen_without_presentation_fields_gives_nulls(client: AsyncClient) -> None:
    """POST without pronouns/description → 201 with both fields None."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client, "Null Presentation")

    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Borin",
            "race": "Human",
            "class_name": "Fighter",
            "alignment": "lawful",
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
    body = response.json()
    assert body["pronouns"] is None
    assert body["description"] is None


@pytest.mark.asyncio
async def test_chargen_null_presentation_fields_explicit(client: AsyncClient) -> None:
    """POST with explicit null for pronouns/description → 201, both still None."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client, "Explicit Null")

    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Dagna",
            "race": "Dwarf",
            "class_name": "Fighter",
            "alignment": "lawful",
            "abilities": {
                "str": 14,
                "int": 10,
                "wis": 10,
                "dex": 12,
                "con": 12,
                "cha": 10,
            },
            "pronouns": None,
            "description": None,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["pronouns"] is None
    assert body["description"] is None


@pytest.mark.asyncio
async def test_chargen_description_over_500_chars_rejected(client: AsyncClient) -> None:
    """description > 500 chars → 422 Unprocessable Entity."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client, "Oversize Desc")

    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Tomas",
            "race": "Human",
            "class_name": "Thief",
            "alignment": "neutral",
            "abilities": {
                "str": 10,
                "int": 13,
                "wis": 10,
                "dex": 14,
                "con": 11,
                "cha": 12,
            },
            "description": "x" * 501,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chargen_pronouns_over_40_chars_rejected(client: AsyncClient) -> None:
    """pronouns > 40 chars → 422 Unprocessable Entity."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client, "Oversize Pronouns")

    response = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": "Zena",
            "race": "Human",
            "class_name": "Cleric",
            "alignment": "lawful",
            "abilities": {
                "str": 10,
                "int": 11,
                "wis": 15,
                "dex": 10,
                "con": 12,
                "cha": 13,
            },
            "pronouns": "z" * 41,
        },
    )
    assert response.status_code == 422
