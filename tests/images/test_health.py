"""Tests for ``app.images.health`` — the FLUX status rendezvous.

The watchdog (in the imageworker process) writes ``image:status`` and
the orchestrator (in the FastAPI process) reads it. The contract is
the JSON payload shape and the missing-key default behaviour — both
are pinned here.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
import redis.asyncio as redis_async

from app.images.health import HEALTH_KEY, read_status, write_status


class _FakeKvClient:
    """Implements just ``get`` / ``set`` against an in-memory dict —
    everything ``app.images.health`` actually calls."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def set(self, key: str, value: bytes) -> bool:
        self._store[key] = value
        return True

    async def get(self, key: str) -> bytes | None:
        return self._store.get(key)


def _client() -> redis_async.Redis:
    return cast(redis_async.Redis, _FakeKvClient())


@pytest.mark.asyncio
async def test_write_then_read_round_trip() -> None:
    """Status + since timestamp round-trip through the JSON wire shape."""

    client = _client()
    await write_status(client, "degraded", since_iso="2026-05-01T12:34:56Z")

    status, since = await read_status(client)
    assert status == "degraded"
    assert since == "2026-05-01T12:34:56Z"


@pytest.mark.asyncio
async def test_read_defaults_to_ok_when_key_missing() -> None:
    """The orchestrator reads on every turn; a never-written key
    (worker hasn't started yet, or the deployment doesn't run a
    worker) must not silently flip the DM into degraded mode."""

    status, since = await read_status(_client())
    assert status == "ok"
    assert since is None


@pytest.mark.asyncio
async def test_read_defaults_to_ok_on_unparseable_payload() -> None:
    """If somebody pokes the key with redis-cli or a future schema
    change lands without a migration, fail open rather than fail
    closed — degrading the DM on garbage data is the worse outcome."""

    fake = _FakeKvClient()
    fake._store[HEALTH_KEY] = b"not-json"
    status, since = await read_status(cast(Any, fake))
    assert status == "ok"
    assert since is None


@pytest.mark.asyncio
async def test_read_defaults_to_ok_on_unknown_status_value() -> None:
    """Forwards-compatibility: a future watchdog might write a third
    state ('initializing' say) — older readers should treat any
    non-{ok,degraded} value as 'don't know, assume ok'."""

    fake = _FakeKvClient()
    fake._store[HEALTH_KEY] = json.dumps(
        {"status": "initializing", "since": "2026-05-01T00:00:00Z"}
    ).encode()
    status, _since = await read_status(cast(Any, fake))
    assert status == "ok"


@pytest.mark.asyncio
async def test_write_status_payload_shape() -> None:
    """Pin the JSON shape so a subtle key rename (e.g. ``state`` vs
    ``status``) doesn't quietly break cross-process readers in the
    middle of a deployment."""

    fake = _FakeKvClient()
    await write_status(cast(Any, fake), "ok", since_iso="2026-05-01T12:00:00Z")
    raw = fake._store[HEALTH_KEY]
    assert json.loads(raw) == {
        "status": "ok",
        "since": "2026-05-01T12:00:00Z",
    }
