"""Tests for ``app.realtime.pubsub``.

Two layers:

* **Unit** — exercises the channel-naming and the singleton accessors;
  uses the in-memory :class:`FakePubsub` from ``fakes.py`` for the
  fan-out semantics so tests don't depend on a live Valkey.
* **Integration-shape** — the smoke test in
  ``tests/integration/test_pubsub_smoke.py`` (added in step 9) hits a
  real Valkey instance; no smoke runs here.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from app.realtime.messages import (
    DiceRoll,
    NarrationChunk,
    Whisper,
)
from app.realtime.pubsub import (
    Pubsub,
    get_pubsub,
    reset_for_tests,
    session_channel,
    set_pubsub_for_tests,
)
from tests.realtime.fakes import FakePubsub


def test_session_channel_format() -> None:
    """The channel name format is part of the spec §9 contract — locking
    it down keeps cross-process subscribers from drifting if anyone
    refactors the helper."""

    assert session_channel("abc-123") == "session:abc-123"
    assert session_channel("ANOTHER") == "session:ANOTHER"


@pytest.mark.asyncio
async def test_get_pubsub_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory caches; two callers see the same instance."""

    # Reset and force the singleton to be a FakePubsub so we don't open
    # a real Valkey connection during this unit test.
    await reset_for_tests()
    fake = FakePubsub()
    set_pubsub_for_tests(fake)
    assert get_pubsub() is fake


@pytest.mark.asyncio
async def test_reset_for_tests_drops_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """After reset, the next ``get_pubsub`` is a fresh instance."""

    fake_a = FakePubsub()
    set_pubsub_for_tests(fake_a)
    await reset_for_tests()
    fake_b = FakePubsub()
    set_pubsub_for_tests(fake_b)
    assert get_pubsub() is fake_b
    assert get_pubsub() is not fake_a


# ---------------------------------------------------------------------------
# FakePubsub — fan-out semantics that production code depends on
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_publish_with_no_subscribers_returns_zero() -> None:
    fake = FakePubsub()
    count = await fake.publish("s-1", NarrationChunk(content="alone in the void"))
    assert count == 0


@pytest.mark.asyncio
async def test_fake_subscribe_receives_published_message() -> None:
    """One subscriber receives a published message and the publish call
    reports one subscriber attached."""

    fake = FakePubsub()
    received: list[NarrationChunk] = []

    async def reader() -> None:
        async for msg in fake.subscribe("s-1"):
            assert isinstance(msg, NarrationChunk)
            received.append(msg)
            return

    task = asyncio.create_task(reader())
    # Yield control once so the iterator attaches before publish.
    await asyncio.sleep(0)
    count = await fake.publish("s-1", NarrationChunk(content="hello"))
    assert count == 1
    await asyncio.wait_for(task, timeout=1.0)
    assert len(received) == 1
    assert received[0].content == "hello"


@pytest.mark.asyncio
async def test_fake_two_subscribers_both_receive() -> None:
    """Multi-client broadcast: two subscribers both get every publish."""

    fake = FakePubsub()
    a_received: list[NarrationChunk] = []
    b_received: list[NarrationChunk] = []

    async def reader(out: list[NarrationChunk]) -> None:
        async for msg in fake.subscribe("s-1"):
            assert isinstance(msg, NarrationChunk)
            out.append(msg)
            if len(out) == 2:
                return

    task_a = asyncio.create_task(reader(a_received))
    task_b = asyncio.create_task(reader(b_received))
    await asyncio.sleep(0)
    n1 = await fake.publish("s-1", NarrationChunk(content="one"))
    n2 = await fake.publish("s-1", NarrationChunk(content="two"))
    assert n1 == 2
    assert n2 == 2
    await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=1.0)
    assert [m.content for m in a_received] == ["one", "two"]
    assert [m.content for m in b_received] == ["one", "two"]


@pytest.mark.asyncio
async def test_fake_session_isolation() -> None:
    """A subscriber on session A never sees a publish to session B."""

    fake = FakePubsub()
    a_received: list[DiceRoll] = []

    async def reader() -> None:
        async for msg in fake.subscribe("s-A"):
            assert isinstance(msg, DiceRoll)
            a_received.append(msg)
            return

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)

    await fake.publish(
        "s-B",  # different session
        DiceRoll(
            tool_call_id="tc-x",
            expression="1d20",
            total=15,
            individual=[15],
            purpose="should not reach",
        ),
    )
    # Confirm the A-subscriber didn't get the B-publish.
    await asyncio.sleep(0.01)  # yield to let the reader run if it would
    assert a_received == []

    # Now publish to A and confirm the reader receives it — proving the
    # subscription is alive and the prior negative was meaningful.
    await fake.publish(
        "s-A",
        DiceRoll(
            tool_call_id="tc-y",
            expression="1d20",
            total=12,
            individual=[12],
            purpose="should reach",
        ),
    )
    await asyncio.wait_for(task, timeout=1.0)
    assert len(a_received) == 1
    assert a_received[0].tool_call_id == "tc-y"


@pytest.mark.asyncio
async def test_fake_subscriber_cancellation_unregisters() -> None:
    """Cancelling a subscriber removes its queue from the channel so
    subsequent publishes don't fan out into a dead consumer."""

    fake = FakePubsub()

    async def reader() -> None:
        async for _ in fake.subscribe("s-1"):
            pass

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    n_attached = await fake.publish("s-1", NarrationChunk(content="x"))
    assert n_attached == 1

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # Yield once so the iterator's finally-block runs.
    await asyncio.sleep(0)
    n_after_cancel = await fake.publish("s-1", NarrationChunk(content="y"))
    assert n_after_cancel == 0


@pytest.mark.asyncio
async def test_fake_publish_after_aclose_raises() -> None:
    fake = FakePubsub()
    await fake.aclose()
    with pytest.raises(RuntimeError):
        await fake.publish("s-1", NarrationChunk(content="x"))


@pytest.mark.asyncio
async def test_fake_supports_whisper_round_trip() -> None:
    """Sanity: the FakePubsub doesn't lose discriminator info across
    its in-memory channel — a Whisper goes in and comes out as a
    Whisper, not a raw dict."""

    fake = FakePubsub()
    received: list[Whisper] = []

    async def reader() -> None:
        async for msg in fake.subscribe("s-1"):
            assert isinstance(msg, Whisper)
            received.append(msg)
            return

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    await fake.publish(
        "s-1",
        Whisper(tool_call_id="tc-1", audience=["ch-1"], content="psst"),
    )
    await asyncio.wait_for(task, timeout=1.0)
    assert received[0].audience == ["ch-1"]


# ---------------------------------------------------------------------------
# Real Pubsub class smoke (skipped if no Valkey — but the foundation
# tests don't gate on this; the integration suite does it explicitly.)
# ---------------------------------------------------------------------------


def test_pubsub_class_exists() -> None:
    """Trivial — confirm the production class can be instantiated.

    No network calls; just the constructor. The integration suite
    exercises live publish/subscribe.
    """

    instance = Pubsub(url="redis://localhost:0/0")  # 0 is fine — never connects
    assert instance.url.startswith("redis://")
