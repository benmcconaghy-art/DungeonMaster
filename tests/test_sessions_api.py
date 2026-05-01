"""Phase 6.8 Bug 3: ``POST /api/campaigns/{id}/sessions`` auto-dispatches
an opening DM turn.

The conftest's ``client`` fixture stubs ``run_dm_turn`` to a no-op
coroutine so most tests don't fire real network I/O. These tests
override that stub to capture the scheduled call without hitting the
LLM or pubsub.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from httpx import AsyncClient

_VALID_PW = "correct horse battery staple"


async def _register(client: AsyncClient, username: str = "alice") -> str:
    r = await client.post("/api/auth/register", json={"username": username, "password": _VALID_PW})
    assert r.status_code == 201, r.text
    return r.json()["id"]  # type: ignore[no-any-return]


async def _create_campaign(client: AsyncClient, name: str = "Borderlands") -> str:
    r = await client.post("/api/campaigns", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]  # type: ignore[no-any-return]


@pytest.mark.asyncio
async def test_create_session_schedules_opening_turn(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creating a session must dispatch an opening DM turn so the player
    lands on a setting-itself scene rather than the placeholder."""

    # Capture run_dm_turn invocations from the schedule helper.
    captured: list[dict[str, Any]] = []

    async def _capture(**kwargs: Any) -> None:
        captured.append(kwargs)

    # Patch the binding inside app.api.sessions where run_dm_turn was
    # imported — that's the reference _schedule_opening_turn calls.
    # This overrides the conftest's no-op stub for this test.
    monkeypatch.setattr("app.api.sessions.run_dm_turn", _capture)

    user_id = await _register(client)
    campaign_id = await _create_campaign(client)

    response = await client.post(f"/api/campaigns/{campaign_id}/sessions")
    assert response.status_code == 201, response.text
    session_id = response.json()["id"]

    # Yield control so the background task has a chance to execute.
    await asyncio.sleep(0)

    assert len(captured) == 1, "exactly one opening turn should have been scheduled"
    call = captured[0]
    assert call["session_id"] == session_id
    assert call["sender_user_id"] == user_id
    assert call["sender_character_id"] is None
    assert call["opening"] is True
    # The directive must read as a stage direction, not as user input.
    assert "Session begins" in call["content"]


@pytest.mark.asyncio
async def test_create_session_does_not_block_on_opening_turn(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTTP response must return promptly even if the opening turn
    is slow (the LLM call could take 10-30s in production). The handler
    schedules a background task and returns immediately."""

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow(**kwargs: Any) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr("app.api.sessions.run_dm_turn", _slow)

    await _register(client)
    campaign_id = await _create_campaign(client)

    try:
        # The HTTP request must complete before run_dm_turn finishes.
        response = await asyncio.wait_for(
            client.post(f"/api/campaigns/{campaign_id}/sessions"), timeout=2.0
        )
        assert response.status_code == 201
        # And the slow background task must have started by now.
        await asyncio.wait_for(started.wait(), timeout=1.0)
    finally:
        # Let the captured task finish so it doesn't leak past the test.
        release.set()
        await asyncio.sleep(0)
