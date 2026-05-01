"""Tests for the bootstrap routes (``/`` and ``/health``)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient) -> None:
    """Happy path: DB is reachable, status and db both ``ok``."""

    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "db": "ok"}


@pytest.mark.asyncio
async def test_root_renders_landing_page(client: AsyncClient) -> None:
    """The Jinja2-rendered home page (Phase 6 design refresh) shows the
    project wordmark and sign-in / register affordances when the
    visitor is anonymous."""

    response = await client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "Dungeon Master" in body
    # Anonymous visit shows both auth entry points.
    assert "Sign in" in body
    assert "Register" in body
