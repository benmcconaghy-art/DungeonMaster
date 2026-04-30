"""Tests for ``app.images.queue`` — the image-job FIFO contract.

The queue crosses two processes (FastAPI app enqueues, imageworker
consumes) so the on-the-wire shape and FIFO semantics must be locked
down. Real Valkey is exercised in the Phase 5 integration test; here
we use an in-memory fake that implements the redis-py async surface
the queue helpers actually call (``rpush``, ``blpop``).
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any, cast

import pytest
import redis.asyncio as redis_async

from app.images.queue import (
    QUEUE_KEY,
    ImageJob,
    ImageJobDecodeError,
    pop_job,
    push_job,
)


class _FakeRedis:
    """Minimal in-memory stand-in for the bits of ``redis.Redis`` the
    queue helpers use. Backed by a ``deque`` so ``rpush`` / ``blpop``
    follow real FIFO semantics."""

    def __init__(self) -> None:
        self._lists: dict[str, deque[bytes]] = {}
        self._wakers: list[asyncio.Future[None]] = []

    async def rpush(self, key: str, value: bytes) -> int:
        self._lists.setdefault(key, deque()).append(value)
        # Wake any pending blpop callers.
        for waker in self._wakers:
            if not waker.done():
                waker.set_result(None)
        self._wakers.clear()
        return len(self._lists[key])

    async def blpop(
        self,
        key: str,
        *,
        timeout: float = 0.0,  # noqa: ASYNC109
    ) -> tuple[bytes, bytes] | None:
        loop = asyncio.get_running_loop()
        deadline = None if timeout <= 0 else loop.time() + timeout
        while True:
            queue = self._lists.get(key)
            if queue:
                value = queue.popleft()
                return key.encode(), value
            if deadline is not None and loop.time() >= deadline:
                return None
            waker: asyncio.Future[None] = loop.create_future()
            self._wakers.append(waker)
            wait_for = None if deadline is None else max(0.001, deadline - loop.time())
            try:
                await asyncio.wait_for(waker, timeout=wait_for)
            except TimeoutError:
                return None


def _client() -> redis_async.Redis:
    """Cast for the type checker — production code annotates as
    ``redis.Redis`` and our fake matches the shape the helpers touch."""

    return cast(redis_async.Redis, _FakeRedis())


def _make_job(**overrides: Any) -> ImageJob:
    base: dict[str, Any] = {
        "id": "img-1",
        "campaign_id": "camp-1",
        "session_id": "sess-1",
        "kind": "scene",
        "prompt": "an alchemist's tower at sunset",
    }
    base.update(overrides)
    return ImageJob.model_validate(base)


@pytest.mark.asyncio
async def test_push_then_pop_round_trips() -> None:
    """A pushed job comes back out through the helpers byte-identical
    after JSON deserialisation. ``blpop`` returns ``(key, payload)``
    so the helper has to discard the key — verify it does."""

    client = _client()
    job = _make_job()

    length = await push_job(client, job)
    assert length == 1

    popped = await pop_job(client, timeout=0.1)
    assert popped is not None
    assert popped.id == job.id
    assert popped.prompt == job.prompt
    assert popped.kind == "scene"


@pytest.mark.asyncio
async def test_queue_is_fifo() -> None:
    """``rpush`` + ``blpop`` are documented FIFO. A regression to
    LIFO would scramble the order DM-emitted jobs are processed in
    and break the placeholder/ready ordering on the client side."""

    client = _client()
    jobs = [_make_job(id=f"img-{i}", prompt=f"p{i}") for i in range(3)]
    for job in jobs:
        await push_job(client, job)

    seen: list[str] = []
    for _ in range(3):
        popped = await pop_job(client, timeout=0.1)
        assert popped is not None
        seen.append(popped.id)
    assert seen == ["img-0", "img-1", "img-2"]


@pytest.mark.asyncio
async def test_pop_returns_none_on_timeout() -> None:
    """Empty queue + finite timeout returns ``None`` so the worker
    loop can wake periodically and check cancellation flags between
    pops without burning CPU."""

    client = _client()
    popped = await pop_job(client, timeout=0.05)
    assert popped is None


@pytest.mark.asyncio
async def test_edit_job_carries_reference_image_id() -> None:
    """Kontext jobs need the source image id + edit instruction to
    survive the round-trip; the worker dispatches based on these
    fields being set."""

    client = _client()
    job = _make_job(
        kind="npc",
        reference_image_id="canon-1",
        edit_instruction="same character, torchlit crypt",
    )
    await push_job(client, job)
    popped = await pop_job(client, timeout=0.1)
    assert popped is not None
    assert popped.reference_image_id == "canon-1"
    assert popped.edit_instruction == "same character, torchlit crypt"


@pytest.mark.asyncio
async def test_subject_ids_round_trip() -> None:
    """The Step 7 portrait flow sets ``subject_character_id`` so the
    worker knows to update ``characters.canonical_image_id`` after
    persisting."""

    client = _client()
    job = _make_job(subject_character_id="char-1")
    await push_job(client, job)
    popped = await pop_job(client, timeout=0.1)
    assert popped is not None
    assert popped.subject_character_id == "char-1"
    assert popped.subject_npc_id is None


@pytest.mark.asyncio
async def test_pop_raises_decode_error_on_garbage_payload() -> None:
    """A poison message (operator pushed garbage with redis-cli, or
    a schema-incompatible enqueuer) must fail loudly, not silently
    return a half-built ImageJob."""

    client = _FakeRedis()
    await client.rpush(QUEUE_KEY, b"not-json")
    with pytest.raises(ImageJobDecodeError):
        await pop_job(cast(Any, client), timeout=0.1)


@pytest.mark.asyncio
async def test_pop_raises_decode_error_on_wrong_shape() -> None:
    """JSON that parses but doesn't validate against ``ImageJob``
    (missing required fields, bad ``kind`` enum) raises the same
    decode error."""

    client = _FakeRedis()
    await client.rpush(QUEUE_KEY, json.dumps({"kind": "scene"}).encode())
    with pytest.raises(ImageJobDecodeError):
        await pop_job(cast(Any, client), timeout=0.1)


def test_image_kind_enum_is_pinned() -> None:
    """Adding a new kind has implications for the worker's per-kind
    parameter table — make a regression visible by pinning the set."""

    # Reading the literal type via __args__ is the most direct way
    # to assert the closed set without re-importing it.
    from typing import get_args

    from app.images.queue import ImageKind

    assert set(get_args(ImageKind)) == {"scene", "npc", "item", "map"}
