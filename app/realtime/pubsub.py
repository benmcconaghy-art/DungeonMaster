"""Valkey/Redis pub/sub adapter for cross-connection fan-out.

Spec §9 names the channel ``session:{session_id}``; spec §13 names
Valkey (Redis-compatible fork; see ``deploy/bootstrap.sh``) as the
backend. The Python client (``redis-py``) treats both identically —
``redis://...`` URLs and the wire protocol are unchanged.

Two responsibilities the WS hub depends on:

1. **Publish** — a single ``publish(session_id, server_message)`` call
   serialises the Pydantic message to JSON bytes and pushes onto the
   channel. The orchestrator's broadcast hook calls this once per
   :class:`~app.orchestrator.dm.DmEvent` it emits.

2. **Subscribe** — a per-WS ``subscribe(session_id)`` returns an async
   iterator over decoded :class:`ServerMessage` instances arriving on
   the channel. The hub spawns one task per connection that fans
   incoming messages from the iterator out to its WebSocket.

The wrapper deliberately surfaces Valkey/Redis liveness errors to the
caller (``DmPubsubError``) rather than swallowing them with retries.
The hub fails the WS handshake on a publish failure so a misconfigured
or down KV store is obvious in development; spec §13 lists Valkey as
non-optional.

Test seam: tests can swap the singleton via :func:`reset_for_tests`
and inject a fake — the in-memory ``FakePubsub`` in
``tests/realtime/fakes.py`` is the canonical test double.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import redis.asyncio as redis
from pydantic import TypeAdapter

from app.config import get_settings
from app.realtime.messages import ServerMessage

log = logging.getLogger(__name__)


class DmPubsubError(RuntimeError):
    """Surface a Valkey publish/subscribe failure as a typed error.

    The hub treats this as fatal for the affected operation: a publish
    failure aborts the broadcast, a subscribe failure rejects the WS
    handshake. Don't paper over with retries; spec §13 has Valkey as
    a hard dependency on the trusted-LAN deployment.
    """


_SESSION_CHANNEL = "session:{session_id}"


def session_channel(session_id: str) -> str:
    """The Valkey channel name for one session's broadcast traffic."""

    return _SESSION_CHANNEL.format(session_id=session_id)


# TypeAdapter caches the discriminated-union deserialisation logic so we
# don't rebuild it on every received message. Validates inbound bytes
# against the ServerMessage union and returns the typed variant.
_SERVER_MSG_ADAPTER: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


class Pubsub:
    """Async wrapper around the redis-py client scoped to session pub/sub.

    Holds one long-lived ``Redis`` client. Subscriber tasks open their
    own ``pubsub()`` context inside :meth:`subscribe` so their lifecycle
    is independent from publishers.
    """

    def __init__(self, *, url: str | None = None) -> None:
        settings = get_settings()
        self._url = url or settings.redis_url
        # decode_responses=False — we serialise Pydantic models to bytes
        # ourselves and want bytes back on the wire so the round-trip is
        # explicit. The pubsub() iterator yields dicts whose 'data' field
        # is bytes; we model_validate_json those.
        self._client: redis.Redis = redis.Redis.from_url(self._url, decode_responses=False)

    @property
    def url(self) -> str:
        return self._url

    async def health(self) -> None:
        """Lightweight liveness check — used by app startup and tests.

        Raises :class:`DmPubsubError` on transport failure so the caller
        gets a typed failure to surface.
        """

        try:
            await self._client.ping()
        except Exception as exc:  # redis.RedisError + asyncio errors
            raise DmPubsubError(f"Valkey ping failed at {self._url}: {exc}") from exc

    async def publish(self, session_id: str, message: ServerMessage) -> int:
        """Publish a typed server message on the session channel.

        Returns the number of subscribers Valkey reports received the
        payload (0 if no listeners — fine, just means no clients are
        currently connected).
        """

        # ServerMessage is a discriminated union; the TypeAdapter handles
        # every variant uniformly without us reaching into a ``.model_dump_json``
        # accessor on the union type (which mypy can't resolve).
        try:
            payload = _SERVER_MSG_ADAPTER.dump_json(message)
        except Exception as exc:
            raise DmPubsubError(f"failed to serialise pubsub message: {exc}") from exc

        try:
            count = await self._client.publish(session_channel(session_id), payload)
        except Exception as exc:
            raise DmPubsubError(f"Valkey publish failed for session {session_id!r}: {exc}") from exc
        return int(count)

    async def subscribe(self, session_id: str) -> AsyncIterator[ServerMessage]:
        """Yield decoded :class:`ServerMessage` instances received on the
        session's channel.

        The iterator runs until the caller cancels it (e.g. the WS task
        exits). On cancellation, the underlying ``pubsub`` context is
        unsubscribed and the connection released to the redis-py pool.
        """

        pubsub = self._client.pubsub()
        try:
            await pubsub.subscribe(session_channel(session_id))
        except Exception as exc:
            await cast(Any, pubsub).aclose()
            raise DmPubsubError(
                f"Valkey subscribe failed for session {session_id!r}: {exc}"
            ) from exc

        try:
            async for raw in pubsub.listen():
                # The first frame after .subscribe() is a 'subscribe' ack
                # (no payload) we want to skip; only 'message' frames carry
                # what publishers sent.
                if raw.get("type") != "message":
                    continue
                data = raw.get("data")
                if data is None:
                    continue
                try:
                    decoded = data.encode("utf-8") if isinstance(data, str) else data
                    message = _SERVER_MSG_ADAPTER.validate_json(decoded)
                except Exception:
                    log.warning(
                        "pubsub: dropped unparseable message on %s",
                        session_channel(session_id),
                    )
                    continue
                yield message
        finally:
            try:
                await pubsub.unsubscribe(session_channel(session_id))
            except Exception:  # cleanup, best effort
                log.debug("pubsub: unsubscribe failed during teardown", exc_info=True)
            await cast(Any, pubsub).aclose()

    async def aclose(self) -> None:
        """Dispose of the underlying connection pool. Called from app
        lifespan teardown so the gunicorn worker shuts down cleanly."""

        await cast(Any, self._client).aclose()


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_singleton: Pubsub | None = None


def get_pubsub() -> Pubsub:
    """Return the process-wide :class:`Pubsub`, building on first call."""

    global _singleton
    if _singleton is None:
        _singleton = Pubsub()
    return _singleton


def set_pubsub_for_tests(instance: Any) -> None:
    """Replace the singleton with a test double.

    The double must implement the same ``health``/``publish``/
    ``subscribe``/``aclose`` surface; ``tests/realtime/fakes.py`` ships
    one keyed off an in-memory queue.
    """

    global _singleton
    _singleton = instance


async def reset_for_tests() -> None:
    """Tear down the singleton; subsequent ``get_pubsub`` builds fresh."""

    global _singleton
    if _singleton is not None:
        try:
            await _singleton.aclose()
        except Exception:
            log.debug("pubsub reset: aclose failed", exc_info=True)
        _singleton = None


__all__ = [
    "DmPubsubError",
    "Pubsub",
    "get_pubsub",
    "reset_for_tests",
    "session_channel",
    "set_pubsub_for_tests",
]
