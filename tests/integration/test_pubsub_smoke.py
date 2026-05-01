"""End-to-end smoke against a live Valkey instance.

Spec §13 names Valkey as a hard dependency. This test confirms that
``app.realtime.pubsub.Pubsub`` round-trips a typed
:class:`~app.realtime.messages.NarrationChunk` through real Valkey
publish/subscribe. Skipped if Valkey isn't reachable, the same shape
as the vLLM integration tests skip when their endpoint is down.

Run with::

    uv run pytest -m integration tests/integration/test_pubsub_smoke.py

The test deliberately uses a fresh, randomly-named session id so it
doesn't collide with any other run; no cleanup needed because nothing
is persisted across publishes.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from app.realtime.messages import NarrationChunk
from app.realtime.pubsub import DmPubsubError, Pubsub

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def reachable_pubsub() -> AsyncIterator[Pubsub]:
    """Yield a live ``Pubsub`` instance bound to localhost Valkey.

    Skips the test if the ping fails (Valkey not running, wrong port,
    auth required) so a misconfigured dev box doesn't fail loudly.
    """

    instance = Pubsub()
    try:
        await instance.health()
    except DmPubsubError as exc:
        await instance.aclose()
        pytest.skip(f"Valkey unreachable: {exc}")
    try:
        yield instance
    finally:
        await instance.aclose()


@pytest.mark.asyncio
async def test_publish_subscribe_round_trip(reachable_pubsub: Pubsub) -> None:
    """Publish a NarrationChunk and read it back via subscribe()."""

    pubsub = reachable_pubsub
    session_id = f"test-{uuid.uuid4()}"
    received: list[NarrationChunk] = []

    async def reader() -> None:
        async for msg in pubsub.subscribe(session_id):
            assert isinstance(msg, NarrationChunk)
            received.append(msg)
            return

    task = asyncio.create_task(reader())
    # Give .subscribe time to attach before the publish; otherwise the
    # publish-before-subscribe race drops the message.
    await asyncio.sleep(0.1)

    count = await pubsub.publish(session_id, NarrationChunk(stream_id="s-1", content="round trip"))
    assert count >= 1, "expected at least one subscriber attached"

    await asyncio.wait_for(task, timeout=2.0)
    assert len(received) == 1
    assert received[0].content == "round trip"


@pytest.mark.asyncio
async def test_isolated_channel_does_not_receive_other_session(
    reachable_pubsub: Pubsub,
) -> None:
    """Subscribers on session A do not see publishes to session B —
    confirms Valkey channel scoping matches the spec §9 contract."""

    pubsub = reachable_pubsub
    session_a = f"test-A-{uuid.uuid4()}"
    session_b = f"test-B-{uuid.uuid4()}"
    a_received: list[NarrationChunk] = []

    async def reader_a() -> None:
        async for msg in pubsub.subscribe(session_a):
            assert isinstance(msg, NarrationChunk)
            a_received.append(msg)
            return

    task = asyncio.create_task(reader_a())
    await asyncio.sleep(0.1)

    # Publish on B — should not reach A.
    await pubsub.publish(session_b, NarrationChunk(stream_id="s-1", content="for B"))
    await asyncio.sleep(0.1)
    assert a_received == []

    # Publish on A — does reach A. Required to prove the subscription
    # is alive and the prior negative was meaningful.
    await pubsub.publish(session_a, NarrationChunk(stream_id="s-1", content="for A"))
    await asyncio.wait_for(task, timeout=2.0)
    assert len(a_received) == 1
    assert a_received[0].content == "for A"
