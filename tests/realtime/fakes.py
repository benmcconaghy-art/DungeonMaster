"""In-memory test doubles for the WS hub's external dependencies.

Real Valkey is in :mod:`tests.integration` (live integration tests).
Most unit tests want fast, deterministic publish/subscribe without a
broker in the loop — :class:`FakePubsub` is the canonical stand-in.

Surface match: anywhere production code calls ``get_pubsub()``, the
test installs a :class:`FakePubsub` via ``set_pubsub_for_tests`` (from
``app.realtime.pubsub``). Method shapes mirror the real client so a
type-check on the production caller passes against either.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from app.realtime.messages import ServerMessage


class FakePubsub:
    """In-memory fan-out: publish copies to every active subscriber's
    queue; the queue's ``get()`` drives the async iterator returned by
    ``subscribe``.

    Per-session queues so tests can publish on session A and assert the
    session-B subscriber doesn't receive — matches the real Valkey
    channel-isolation semantics.

    Two limitations relative to the real client (irrelevant for the
    cases the unit tests cover):

    * No back-pressure: ``asyncio.Queue`` is unbounded. The real
      client drops slow-subscriber messages eventually; tests should
      drain quickly enough that this doesn't matter.
    * No subscribe-then-late-attach race: every ``subscribe`` call
      gets its own queue, so messages published before the iterator
      is awaited are NOT delivered (just like the real client). Tests
      that want to assert one-shot delivery should ``await`` the
      iterator before publishing.
    """

    def __init__(self) -> None:
        # session_id -> list of subscriber queues (one per active
        # subscribe call). Publish iterates and ``put_nowait`` to each.
        self._channels: dict[str, list[asyncio.Queue[ServerMessage]]] = {}
        self._closed = False

    async def health(self) -> None:
        """Always healthy — the test pubsub never needs the network."""

    async def publish(self, session_id: str, message: ServerMessage) -> int:
        """Fan a copy out to every queue subscribed to ``session_id``.

        Returns the subscriber count, matching the real client's
        contract.
        """

        if self._closed:
            raise RuntimeError("FakePubsub is closed")
        queues = self._channels.get(session_id, [])
        for q in queues:
            q.put_nowait(message)
        return len(queues)

    async def subscribe(self, session_id: str) -> AsyncIterator[ServerMessage]:
        """Async iterator yielding messages published on ``session_id``.

        The iterator runs until the consumer cancels it. On exit the
        queue is detached from the channel list so a subsequent
        ``publish`` doesn't fan out into a dropped consumer.
        """

        queue: asyncio.Queue[ServerMessage] = asyncio.Queue()
        self._channels.setdefault(session_id, []).append(queue)
        try:
            while True:
                msg = await queue.get()
                yield msg
        finally:
            queues = self._channels.get(session_id, [])
            with contextlib.suppress(ValueError):
                queues.remove(queue)
            if not queues:
                self._channels.pop(session_id, None)

    async def aclose(self) -> None:
        """Mark closed and forget every queue. Subsequent publishes
        raise; subsequent subscribes raise."""

        self._closed = True
        self._channels.clear()


__all__ = ["FakePubsub"]
