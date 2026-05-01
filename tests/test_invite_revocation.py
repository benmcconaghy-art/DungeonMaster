"""Phase 7 invite-revocation lifecycle tests.

Exercises the contract added in Phase 7:

* mint → row inserted, signed token includes ``invite_id``
* GET /api/campaigns/{id}/invites — owner sees full audit, classifies
  state correctly (active / used / revoked / expired)
* DELETE /api/campaigns/invites/{id} — owner revokes; idempotent;
  non-owner forbidden; never-existed → 404
* redeem path: revoked / expired / never-existed / already-used all
  return 400 with distinct messages; happy path marks the row used
* legacy grace: a Phase 6 stateless token (no ``invite_id``) is
  accepted while ``_LEGACY_GRACE_END`` lies in the future, rejected
  after.

The legacy-grace tests monkeypatch ``_LEGACY_GRACE_END`` so they're
deterministic regardless of wall clock.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from app.api import campaigns as campaigns_module
from app.db import models

_VALID_PW = "correct horse battery staple"


async def _register_login(client: AsyncClient, username: str) -> str:
    r = await client.post(
        "/api/auth/register",
        json={"username": username, "password": _VALID_PW},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]  # type: ignore[no-any-return]


async def _create_campaign(client: AsyncClient, name: str = "Phase 7") -> str:
    r = await client.post("/api/campaigns", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]  # type: ignore[no-any-return]


async def _mint(client: AsyncClient, campaign_id: str) -> dict[str, Any]:
    r = await client.post(f"/api/campaigns/{campaign_id}/invite")
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Mint shape + list endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mint_returns_invite_id_and_inserts_row(client: AsyncClient) -> None:
    """Mint returns ``invite_id`` and persists a row with the matching id."""

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)

    minted = await _mint(client, campaign_id)
    assert "invite_id" in minted
    assert minted["expires_in_seconds"] >= 24 * 3600

    listed = await client.get(f"/api/campaigns/{campaign_id}/invites")
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    assert rows[0]["invite_id"] == minted["invite_id"]
    assert rows[0]["state"] == "active"
    assert rows[0]["used_by"] is None
    assert rows[0]["revoked_at"] is None


@pytest.mark.asyncio
async def test_list_invites_owner_only(client: AsyncClient) -> None:
    """A non-owner member can't list a campaign's invites."""

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    minted = await _mint(client, campaign_id)
    await client.post("/api/auth/logout")

    await _register_login(client, "bob")
    join = await client.post("/api/campaigns/join", json={"code": minted["code"]})
    assert join.status_code == 200

    # Bob is now a player, but list-invites is owner-only.
    forbidden = await client.get(f"/api/campaigns/{campaign_id}/invites")
    assert forbidden.status_code == 403


@pytest.mark.asyncio
async def test_list_invites_classifies_state_correctly(client: AsyncClient) -> None:
    """``state`` reports active / used / revoked correctly across rows."""

    alice_id = await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)

    # Three invites: leave one active, revoke one, redeem one.
    active = await _mint(client, campaign_id)
    revoked = await _mint(client, campaign_id)
    used = await _mint(client, campaign_id)

    revoke_resp = await client.delete(f"/api/campaigns/invites/{revoked['invite_id']}")
    assert revoke_resp.status_code == 204

    await client.post("/api/auth/logout")
    bob_id = await _register_login(client, "bob")
    join = await client.post("/api/campaigns/join", json={"code": used["code"]})
    assert join.status_code == 200
    await client.post("/api/auth/logout")

    # Back to alice for the list.
    await client.post(
        "/api/auth/login",
        json={"username": "alice", "password": _VALID_PW},
    )
    listed = (await client.get(f"/api/campaigns/{campaign_id}/invites")).json()
    by_id = {row["invite_id"]: row for row in listed}
    assert by_id[active["invite_id"]]["state"] == "active"
    assert by_id[revoked["invite_id"]]["state"] == "revoked"
    assert by_id[used["invite_id"]]["state"] == "used"
    assert by_id[used["invite_id"]]["used_by"] == bob_id
    assert by_id[active["invite_id"]]["created_by"] == alice_id


# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revoke_invite_idempotent(client: AsyncClient) -> None:
    """Two DELETE calls in a row both succeed (204), and the row stays
    revoked at the timestamp from the first call."""

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    minted = await _mint(client, campaign_id)

    r1 = await client.delete(f"/api/campaigns/invites/{minted['invite_id']}")
    assert r1.status_code == 204

    listed = (await client.get(f"/api/campaigns/{campaign_id}/invites")).json()
    first_revoked_at = listed[0]["revoked_at"]
    assert first_revoked_at is not None

    r2 = await client.delete(f"/api/campaigns/invites/{minted['invite_id']}")
    assert r2.status_code == 204

    listed = (await client.get(f"/api/campaigns/{campaign_id}/invites")).json()
    assert listed[0]["revoked_at"] == first_revoked_at


@pytest.mark.asyncio
async def test_revoke_invite_non_owner_forbidden(client: AsyncClient) -> None:
    """A campaign player (non-owner) cannot revoke. Even an unrelated
    user (no membership at all) gets 403, not 404 — the distinction
    being that the invite *exists* but the caller isn't authorised."""

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    minted = await _mint(client, campaign_id)
    await client.post("/api/auth/logout")

    await _register_login(client, "bob")
    join = await client.post("/api/campaigns/join", json={"code": minted["code"]})
    assert join.status_code == 200

    # Bob is now a player, not the owner — revoke must 403.
    # Mint a fresh one so we don't try to revoke the already-used invite.
    await client.post("/api/auth/logout")
    await client.post("/api/auth/login", json={"username": "alice", "password": _VALID_PW})
    fresh = await _mint(client, campaign_id)
    await client.post("/api/auth/logout")

    await client.post("/api/auth/login", json={"username": "bob", "password": _VALID_PW})
    forbidden = await client.delete(f"/api/campaigns/invites/{fresh['invite_id']}")
    assert forbidden.status_code == 403


@pytest.mark.asyncio
async def test_revoke_never_existed_returns_404(client: AsyncClient) -> None:
    """Revoking an invite that doesn't exist at all returns 404."""

    await _register_login(client, "alice")

    response = await client.delete("/api/campaigns/invites/01234567-89ab-7def-8000-000000000000")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Redeem failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_join_revoked_invite_returns_400(client: AsyncClient) -> None:
    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    minted = await _mint(client, campaign_id)
    revoke_resp = await client.delete(f"/api/campaigns/invites/{minted['invite_id']}")
    assert revoke_resp.status_code == 204
    await client.post("/api/auth/logout")

    await _register_login(client, "bob")
    join = await client.post("/api/campaigns/join", json={"code": minted["code"]})
    assert join.status_code == 400
    assert "revoked" in join.json()["detail"]


@pytest.mark.asyncio
async def test_join_invite_with_unknown_id_returns_400(client: AsyncClient) -> None:
    """A token whose signature is valid but whose ``invite_id`` doesn't
    correspond to a row returns 400. Simulated by signing a token by
    hand with a never-existed id — the signer accepts it (signature
    valid) but the redeem path rejects on the row lookup."""

    from itsdangerous import URLSafeTimedSerializer

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    minted = await _mint(client, campaign_id)

    forged = URLSafeTimedSerializer(
        campaigns_module.get_settings().session_secret,
        salt=campaigns_module._INVITE_SALT,
    ).dumps(
        {
            "invite_id": "01234567-89ab-7def-8000-aaaaaaaaaaaa",
            "campaign_id": campaign_id,
        }
    )

    await client.post("/api/auth/logout")
    await _register_login(client, "bob")
    forged_resp = await client.post("/api/campaigns/join", json={"code": forged})
    assert forged_resp.status_code == 400
    # Sanity: the genuine, fresh-from-mint code on the same campaign
    # still works — proves the rejection above isn't a false positive.
    fresh_resp = await client.post("/api/campaigns/join", json={"code": minted["code"]})
    assert fresh_resp.status_code == 200


@pytest.mark.asyncio
async def test_join_used_invite_returns_400(client: AsyncClient) -> None:
    """The same code presented twice — the second redemption is 400."""

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    minted = await _mint(client, campaign_id)
    await client.post("/api/auth/logout")

    await _register_login(client, "bob")
    first = await client.post("/api/campaigns/join", json={"code": minted["code"]})
    assert first.status_code == 200

    # Logout + register a third user; the SAME code should still 400.
    await client.post("/api/auth/logout")
    await _register_login(client, "carol")
    second = await client.post("/api/campaigns/join", json={"code": minted["code"]})
    assert second.status_code == 400
    assert "already been used" in second.json()["detail"]


@pytest.mark.asyncio
async def test_join_expired_invite_returns_400(client: AsyncClient) -> None:
    """An invite whose ``expires_at`` lies in the past is rejected."""

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    minted = await _mint(client, campaign_id)

    # Reach into the DB and rewrite the row's expires_at to a past
    # timestamp. This simulates the "7 days have passed" path without
    # waiting in real time.
    from app.deps import get_db
    from app.main import app as fastapi_app

    db_dep = fastapi_app.dependency_overrides[get_db]
    async for db in db_dep():
        invite = await db.get(models.CampaignInvite, minted["invite_id"])
        invite.expires_at = "2020-01-01T00:00:00.000Z"
        await db.commit()
        break

    await client.post("/api/auth/logout")
    await _register_login(client, "bob")
    join = await client.post("/api/campaigns/join", json={"code": minted["code"]})
    assert join.status_code == 400
    assert "expired" in join.json()["detail"]


# ---------------------------------------------------------------------------
# Legacy grace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_token_within_grace_accepted(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Phase 6 stateless token (no ``invite_id``) is honoured while
    ``_LEGACY_GRACE_END`` is in the future. Adds the user to the
    campaign and logs a deprecation warning."""

    monkeypatch.setattr(campaigns_module, "_LEGACY_GRACE_END", "2099-01-01T00:00:00.000Z")

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    await client.post("/api/auth/logout")

    # Mint a Phase 6-shaped token directly (no DB row).
    from itsdangerous import URLSafeTimedSerializer

    legacy = URLSafeTimedSerializer(
        campaigns_module.get_settings().session_secret,
        salt=campaigns_module._INVITE_SALT,
    ).dumps({"campaign_id": campaign_id, "by": "any-user-id"})

    await _register_login(client, "bob")
    join = await client.post("/api/campaigns/join", json={"code": legacy})
    assert join.status_code == 200
    assert join.json()["campaign_id"] == campaign_id


@pytest.mark.asyncio
async def test_legacy_token_after_grace_rejected(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the cutoff the legacy path returns 400."""

    monkeypatch.setattr(campaigns_module, "_LEGACY_GRACE_END", "2020-01-01T00:00:00.000Z")

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    await client.post("/api/auth/logout")

    from itsdangerous import URLSafeTimedSerializer

    legacy = URLSafeTimedSerializer(
        campaigns_module.get_settings().session_secret,
        salt=campaigns_module._INVITE_SALT,
    ).dumps({"campaign_id": campaign_id, "by": "any-user-id"})

    await _register_login(client, "bob")
    join = await client.post("/api/campaigns/join", json={"code": legacy})
    assert join.status_code == 400
    assert "deprecated" in join.json()["detail"]


# ---------------------------------------------------------------------------
# DB-level invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_used_invite_records_redeemer_and_timestamp(
    client: AsyncClient,
) -> None:
    """After a successful redemption the row's ``used_by`` matches the
    redeeming user id and ``used_at`` is non-null."""

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    minted = await _mint(client, campaign_id)
    await client.post("/api/auth/logout")

    bob_id = await _register_login(client, "bob")
    join = await client.post("/api/campaigns/join", json={"code": minted["code"]})
    assert join.status_code == 200

    # Read back via the list endpoint (owner-only) for an end-to-end check.
    await client.post("/api/auth/logout")
    await client.post("/api/auth/login", json={"username": "alice", "password": _VALID_PW})
    rows = (await client.get(f"/api/campaigns/{campaign_id}/invites")).json()
    assert rows[0]["used_by"] == bob_id
    assert rows[0]["used_at"] is not None
    assert rows[0]["state"] == "used"


@pytest.mark.asyncio
async def test_revoking_used_invite_is_allowed_and_changes_state(
    client: AsyncClient,
) -> None:
    """Revoking a used invite is allowed; the audit trail keeps both
    facts visible (``used_by`` stays set; ``revoked_at`` is set;
    ``state`` is "revoked" because revocation dominates the
    classification)."""

    await _register_login(client, "alice")
    campaign_id = await _create_campaign(client)
    minted = await _mint(client, campaign_id)
    await client.post("/api/auth/logout")

    await _register_login(client, "bob")
    join = await client.post("/api/campaigns/join", json={"code": minted["code"]})
    assert join.status_code == 200
    await client.post("/api/auth/logout")

    await client.post("/api/auth/login", json={"username": "alice", "password": _VALID_PW})
    revoke = await client.delete(f"/api/campaigns/invites/{minted['invite_id']}")
    assert revoke.status_code == 204
    rows = (await client.get(f"/api/campaigns/{campaign_id}/invites")).json()
    assert rows[0]["state"] == "revoked"
    assert rows[0]["used_by"] is not None  # audit preserved
    assert rows[0]["revoked_at"] is not None
