"""Auth-flow tests: register → login → /me, plus failure modes."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

# Match the validator in ``app.api.auth.RegisterRequest``: 8+ chars.
_VALID_PW = "correct horse battery staple"


@pytest.mark.asyncio
async def test_register_returns_user_and_sets_session(client: AsyncClient) -> None:
    """A successful register returns a UserResponse and authenticates the
    caller — no separate login round-trip required."""

    response = await client.post(
        "/api/auth/register",
        json={"username": "alice", "email": "alice@example.com", "password": _VALID_PW},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["username"] == "alice"
    assert body["email"] == "alice@example.com"
    assert body["is_admin"] is False
    assert "id" in body and "created_at" in body
    # The register response should not leak the password hash.
    assert "pwd_hash" not in body

    me = await client.get("/api/me")
    assert me.status_code == 200
    assert me.json()["id"] == body["id"]


@pytest.mark.asyncio
async def test_register_login_logout_round_trip(client: AsyncClient) -> None:
    register = await client.post(
        "/api/auth/register",
        json={"username": "bob", "password": _VALID_PW},
    )
    assert register.status_code == 201

    logout = await client.post("/api/auth/logout")
    assert logout.status_code == 204

    me_after_logout = await client.get("/api/me")
    assert me_after_logout.status_code == 401

    login = await client.post(
        "/api/auth/login",
        json={"username": "bob", "password": _VALID_PW},
    )
    assert login.status_code == 200
    assert login.json()["username"] == "bob"

    me_after_login = await client.get("/api/me")
    assert me_after_login.status_code == 200


@pytest.mark.asyncio
async def test_login_with_wrong_password_returns_401(client: AsyncClient) -> None:
    await client.post(
        "/api/auth/register",
        json={"username": "carol", "password": _VALID_PW},
    )
    await client.post("/api/auth/logout")

    response = await client.post(
        "/api/auth/login",
        json={"username": "carol", "password": "definitely wrong"},
    )
    assert response.status_code == 401
    # Response body must not differentiate "no such user" from "wrong password"
    # — same message either way (account-enumeration defence).
    assert response.json() == {"detail": "invalid username or password"}


@pytest.mark.asyncio
async def test_login_unknown_user_returns_401(client: AsyncClient) -> None:
    """Same shape as wrong-password: the error message and status must
    not let an attacker tell the two apart."""

    response = await client.post(
        "/api/auth/login",
        json={"username": "nonexistent", "password": _VALID_PW},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid username or password"}


@pytest.mark.asyncio
async def test_login_is_case_insensitive_on_username(client: AsyncClient) -> None:
    """Usernames collapse to a single identity regardless of case (matches
    the COLLATE NOCASE intent in spec §5)."""

    await client.post(
        "/api/auth/register",
        json={"username": "dave", "password": _VALID_PW},
    )
    await client.post("/api/auth/logout")

    response = await client.post(
        "/api/auth/login",
        json={"username": "DAVE", "password": _VALID_PW},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_duplicate_username_returns_409(client: AsyncClient) -> None:
    first = await client.post(
        "/api/auth/register",
        json={"username": "eve", "password": _VALID_PW},
    )
    assert first.status_code == 201

    second = await client.post(
        "/api/auth/register",
        json={"username": "eve", "password": "anothersafepass"},
    )
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_duplicate_username_is_case_insensitive(client: AsyncClient) -> None:
    """Registering ``frank`` then ``FRANK`` must collide."""

    first = await client.post(
        "/api/auth/register",
        json={"username": "frank", "password": _VALID_PW},
    )
    assert first.status_code == 201

    second = await client.post(
        "/api/auth/register",
        json={"username": "FRANK", "password": _VALID_PW},
    )
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_short_password_rejected_at_validation(client: AsyncClient) -> None:
    response = await client.post(
        "/api/auth/register",
        json={"username": "shortpw", "password": "abc"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_logout_when_not_signed_in_is_noop(client: AsyncClient) -> None:
    """Logout is idempotent — calling it without a session must not 500."""

    response = await client.post("/api/auth/logout")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_me_without_session_returns_401(client: AsyncClient) -> None:
    response = await client.get("/api/me")
    assert response.status_code == 401
    assert response.json() == {"detail": "not authenticated"}
