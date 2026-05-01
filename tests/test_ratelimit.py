"""Rate-limiting tests for the auth-shaped endpoints (Phase 7).

Covers the three contracts the brief calls out:

* The limit applies — N+1th request is rejected with 429.
* The 429 carries a Retry-After header and a clear human message,
  not the raw "60 per 1 minute" limits-DSL string.
* The ``reset_for_tests`` autouse fixture genuinely resets state
  between tests (otherwise the first test in the suite would
  permanently break the ones that follow).

The module-level limit constants are imported and locally overridden
to small values via ``monkeypatch`` so a test can trip the limit in
a handful of requests rather than 60. The endpoint contract is the
same; only the threshold differs.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from limits import parse

from app import ratelimit

_VALID_PW = "correct horse battery staple"


# ---------------------------------------------------------------------------
# Trip the limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_rate_limit_returns_429_with_retry_after(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 4th login attempt with a 3/minute limit should be 429."""

    # Shrink the parsed login limit to 3/minute so the test trips
    # within a handful of requests instead of 60. The endpoint
    # contract — limit applies, 429 returned, Retry-After header,
    # human message — is what's under test, not the calibration.
    monkeypatch.setattr(ratelimit, "_LOGIN_ITEM", parse("3/minute"))

    # Pre-create the user so the auth path fails on credentials
    # rather than on user-not-found, which is irrelevant here.
    await client.post(
        "/api/auth/register",
        json={"username": "rl_login", "password": _VALID_PW},
    )
    await client.post("/api/auth/logout")

    bad_login = {"username": "rl_login", "password": "wrong-password-x"}

    # First three attempts: 401 (bad password) but NOT rate-limited.
    for _ in range(3):
        r = await client.post("/api/auth/login", json=bad_login)
        assert r.status_code == 401, r.text

    # Fourth attempt trips the limit.
    r = await client.post("/api/auth/login", json=bad_login)
    assert r.status_code == 429
    assert r.json()["detail"] == ratelimit.HUMAN_MESSAGE
    # Retry-After is in seconds, equal to the limit window (60s for /minute).
    assert r.headers["retry-after"] == "60"


@pytest.mark.asyncio
async def test_register_rate_limit_minute_limit_trips(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The register endpoint enforces the per-minute limit."""

    # Shrink to 2/minute; second 'fresh' attempt is the third hit
    # because each register attempt counts whether or not it succeeds.
    monkeypatch.setattr(
        ratelimit,
        "_REGISTER_ITEMS",
        (parse("2/minute"), parse("100/hour")),
    )

    r1 = await client.post(
        "/api/auth/register",
        json={"username": "rl_a", "password": _VALID_PW},
    )
    assert r1.status_code == 201

    # Purge the just-set session so the second register isn't an
    # already-signed-in collision; rate limiting is per IP regardless
    # of session state, which is the bit being tested.
    await client.post("/api/auth/logout")

    r2 = await client.post(
        "/api/auth/register",
        json={"username": "rl_b", "password": _VALID_PW},
    )
    assert r2.status_code == 201

    await client.post("/api/auth/logout")

    r3 = await client.post(
        "/api/auth/register",
        json={"username": "rl_c", "password": _VALID_PW},
    )
    assert r3.status_code == 429
    assert r3.json()["detail"] == ratelimit.HUMAN_MESSAGE
    assert r3.headers["retry-after"] == "60"


@pytest.mark.asyncio
async def test_register_rate_limit_hour_limit_trips_independently(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hour limit can trip even when the minute limit hasn't.

    With a 100/minute + 2/hour combination, the third request fails
    on the hour budget — proving both limits are checked, not just
    the first.
    """

    monkeypatch.setattr(
        ratelimit,
        "_REGISTER_ITEMS",
        (parse("100/minute"), parse("2/hour")),
    )

    r1 = await client.post(
        "/api/auth/register",
        json={"username": "rl_h1", "password": _VALID_PW},
    )
    assert r1.status_code == 201
    await client.post("/api/auth/logout")

    r2 = await client.post(
        "/api/auth/register",
        json={"username": "rl_h2", "password": _VALID_PW},
    )
    assert r2.status_code == 201
    await client.post("/api/auth/logout")

    r3 = await client.post(
        "/api/auth/register",
        json={"username": "rl_h3", "password": _VALID_PW},
    )
    assert r3.status_code == 429
    # Retry-After should reflect the hour window since that's what tripped.
    assert r3.headers["retry-after"] == "3600"


@pytest.mark.asyncio
async def test_join_rate_limit_trips(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The campaign-join endpoint enforces its own limit."""

    monkeypatch.setattr(ratelimit, "_JOIN_ITEM", parse("2/minute"))

    # Need an authenticated user since join requires a session.
    await client.post(
        "/api/auth/register",
        json={"username": "rl_join", "password": _VALID_PW},
    )

    bad_payload = {"code": "garbage-code-not-a-real-token"}
    # First two attempts: 400 (invalid invite) — limit not yet tripped.
    for _ in range(2):
        r = await client.post("/api/campaigns/join", json=bad_payload)
        assert r.status_code == 400, r.text

    # Third attempt trips.
    r = await client.post("/api/campaigns/join", json=bad_payload)
    assert r.status_code == 429
    assert r.json()["detail"] == ratelimit.HUMAN_MESSAGE


# ---------------------------------------------------------------------------
# Counters reset between tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_counters_reset_between_tests_part_one(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trip the login limit; the next test in this pair must still
    have a clean slate."""

    monkeypatch.setattr(ratelimit, "_LOGIN_ITEM", parse("1/minute"))
    await client.post(
        "/api/auth/register",
        json={"username": "rl_reset_a", "password": _VALID_PW},
    )
    await client.post("/api/auth/logout")

    r1 = await client.post(
        "/api/auth/login",
        json={"username": "rl_reset_a", "password": _VALID_PW},
    )
    assert r1.status_code == 200

    r2 = await client.post(
        "/api/auth/login",
        json={"username": "rl_reset_a", "password": _VALID_PW},
    )
    # After hitting once, the 1/minute limit trips on the very
    # next call.
    assert r2.status_code == 429


@pytest.mark.asyncio
async def test_counters_reset_between_tests_part_two(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The autouse ``_fresh_ratelimit_counters`` fixture must have
    flushed state from the previous test — otherwise the first
    request below would see the prior test's exhausted counter
    and 429 immediately."""

    monkeypatch.setattr(ratelimit, "_LOGIN_ITEM", parse("1/minute"))
    await client.post(
        "/api/auth/register",
        json={"username": "rl_reset_b", "password": _VALID_PW},
    )
    await client.post("/api/auth/logout")

    r1 = await client.post(
        "/api/auth/login",
        json={"username": "rl_reset_b", "password": _VALID_PW},
    )
    # If counters didn't reset, this would already be 429.
    assert r1.status_code == 200, r1.text


# ---------------------------------------------------------------------------
# Production storage backend lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_production_storage_backend_handles_full_lifecycle(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production-default storage backend handles the request
    lifecycle end-to-end without app-startup async context setup.

    Phase 7 playthrough surfaced two related bugs against the prior
    Valkey-backed configuration: the ``limits`` library's async-redis
    backend pulled in coredis as a hard prerequisite (ConfigurationError
    when missing), and coredis uses an async-context-managed connection
    pool that the lazy module-singleton ``_ensure_limiter`` couldn't
    enter (RuntimeError: connection pool not initialized).

    Both bugs trace to a storage backend that needs lifecycle hooks the
    module doesn't provide; in-process memory storage doesn't. This
    test exercises the real production storage URI through the real
    HTTP middleware path with no monkeypatching of the backend itself
    — so a regression that re-introduced a lifecycle-managed backend
    would fail at CI rather than only on a real Valkey-equipped deploy.
    """

    # The contract under test is the storage URI itself; pin it without
    # patching. Limit calibration is shrunk so the test trips quickly.
    assert ratelimit._STORAGE_URI == "async+memory://"
    monkeypatch.setattr(ratelimit, "_LOGIN_ITEM", parse("2/minute"))

    await client.post(
        "/api/auth/register",
        json={"username": "rl_lifecycle", "password": _VALID_PW},
    )
    await client.post("/api/auth/logout")

    bad = {"username": "rl_lifecycle", "password": "wrong-pw-x"}

    # Two attempts hit the auth check (401, not 429) — the limiter
    # increments cleanly per request.
    for _ in range(2):
        r = await client.post("/api/auth/login", json=bad)
        assert r.status_code == 401, r.text

    # Third trips the limit. Proves the storage backend's hit/check
    # cycle survives a complete request — ASGI lifespan never set up
    # an async context manager for it.
    r = await client.post("/api/auth/login", json=bad)
    assert r.status_code == 429
    assert r.headers["retry-after"] == "60"
    assert r.json()["detail"] == ratelimit.HUMAN_MESSAGE


# ---------------------------------------------------------------------------
# X-Forwarded-For respect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_xff_separates_rate_limit_keys(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two distinct X-Forwarded-For values get distinct counter
    buckets — exhausting one IP's budget doesn't penalise the other.

    The trusted-LAN deployment puts nginx in front of the app, so
    XFF is the authoritative client identifier for rate-limit keying.
    """

    monkeypatch.setattr(ratelimit, "_LOGIN_ITEM", parse("1/minute"))

    await client.post(
        "/api/auth/register",
        json={"username": "rl_xff", "password": _VALID_PW},
    )
    await client.post("/api/auth/logout")

    bad = {"username": "rl_xff", "password": "wrong-pw-x"}

    # First IP burns its single shot.
    r1 = await client.post(
        "/api/auth/login",
        json=bad,
        headers={"X-Forwarded-For": "10.0.0.1"},
    )
    assert r1.status_code == 401

    r2 = await client.post(
        "/api/auth/login",
        json=bad,
        headers={"X-Forwarded-For": "10.0.0.1"},
    )
    assert r2.status_code == 429

    # Second IP has its own bucket — should still get the credential
    # 401 rather than the rate-limit 429.
    r3 = await client.post(
        "/api/auth/login",
        json=bad,
        headers={"X-Forwarded-For": "10.0.0.2"},
    )
    assert r3.status_code == 401
