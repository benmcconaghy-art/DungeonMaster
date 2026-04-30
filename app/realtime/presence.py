"""Presence tracking — who is currently connected to which session.

In-memory state mapping ``session_id`` → set of connected
``(user_id, character_id)`` tuples and the rendered display info each
needs (``username``, ``character_name``). The hub adds an entry when a
WebSocket connects and authorises, removes on disconnect; in both cases
it broadcasts a :class:`~app.realtime.messages.Presence` frame so every
connected client converges on the same view of the table.

Two operations the hub depends on:

* :meth:`PresenceRegistry.connect` and :meth:`disconnect` mutate the
  per-session set and return the new full roster as a sorted list of
  :class:`PresenceEntry` (so the server-emitted ``presence`` message is
  always self-contained — clients never need to track diffs).
* :meth:`roster` reads the current set without mutating, used inside
  the snapshot the hub sends each new client on connect.

Concurrency: Phase 4 runs a single gunicorn worker (spec §13), so the
registry is process-local and uncoordinated. AGENTS.md Follow-up #6
flags multi-worker fan-out for Phase 7+ if and when that ships;
nothing in this file relies on multi-process state today.

The registry intentionally does NOT hold WebSocket objects. The hub
owns those, keyed independently. Decoupling means a presence-roster
read doesn't pin live socket state, and tests can swap the registry
without touching the WS layer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.realtime.messages import PresenceEntry


@dataclass(frozen=True, slots=True)
class _ConnKey:
    """Hashable key per active WS attachment.

    A user with two browser tabs counts as two connections; a user with
    one tab playing two characters (rare in practice) also counts as
    two. Keying on ``(user_id, character_id, conn_id)`` keeps every
    connection visible without collapsing them prematurely.
    """

    user_id: str
    character_id: str | None
    conn_id: str


@dataclass
class _ConnInfo:
    """Display data we cache so presence broadcasts don't need a DB read."""

    username: str
    character_name: str | None


@dataclass
class _SessionState:
    """Per-session presence state."""

    connections: dict[_ConnKey, _ConnInfo] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class PresenceRegistry:
    """Process-local map of session_id → connected (user, character) set.

    Methods are coroutines because the underlying ``asyncio.Lock`` makes
    every mutation an awaitable; readers (``roster``) take the lock too
    so they observe a consistent view. The lock is per-session so two
    sessions don't serialise against each other.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        # Coarse lock guarding the outer dict; the per-session lock
        # protects within-session mutations. The outer lock is held only
        # for the dict[setdefault] pattern, never across an await.
        self._sessions_lock = asyncio.Lock()

    async def _state_for(self, session_id: str) -> _SessionState:
        """Lazily allocate the per-session state."""

        async with self._sessions_lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            return state

    async def connect(
        self,
        *,
        session_id: str,
        user_id: str,
        username: str,
        character_id: str | None,
        character_name: str | None,
        conn_id: str,
    ) -> list[PresenceEntry]:
        """Record a connection; return the new full roster.

        ``conn_id`` is whatever the hub uses to identify this specific
        WebSocket — typically a UUID per accepted connection. Two
        attachments from the same (user, character) but different
        ``conn_id`` are kept separately.
        """

        state = await self._state_for(session_id)
        async with state.lock:
            key = _ConnKey(user_id=user_id, character_id=character_id, conn_id=conn_id)
            state.connections[key] = _ConnInfo(username=username, character_name=character_name)
            return _render_roster(state.connections)

    async def disconnect(
        self,
        *,
        session_id: str,
        user_id: str,
        character_id: str | None,
        conn_id: str,
    ) -> list[PresenceEntry]:
        """Remove a connection; return the new full roster.

        Disconnecting an unknown ``conn_id`` is a no-op (idempotent —
        repeated disconnect from broken socket cleanup paths shouldn't
        raise). The roster reflects whatever's left; the hub broadcasts
        even when the call was a no-op so converging clients agree.
        """

        state = await self._state_for(session_id)
        async with state.lock:
            key = _ConnKey(user_id=user_id, character_id=character_id, conn_id=conn_id)
            state.connections.pop(key, None)
            return _render_roster(state.connections)

    async def roster(self, session_id: str) -> list[PresenceEntry]:
        """Return the current roster without mutating."""

        state = await self._state_for(session_id)
        async with state.lock:
            return _render_roster(state.connections)


def _render_roster(connections: dict[_ConnKey, _ConnInfo]) -> list[PresenceEntry]:
    """Render the connections dict to the wire shape, deterministically
    ordered.

    Sort by ``(username, character_name)`` so a client redrawing on
    every presence frame doesn't see entries shuffle on every join. We
    also collapse duplicate (user, character) appearances if any — two
    tabs of the same player on the same PC are equivalent for the
    presence roster the player sees.
    """

    seen: set[tuple[str, str | None]] = set()
    entries: list[PresenceEntry] = []
    for key, info in connections.items():
        identity = (key.user_id, key.character_id)
        if identity in seen:
            continue
        seen.add(identity)
        entries.append(
            PresenceEntry(
                user_id=key.user_id,
                username=info.username,
                character_id=key.character_id,
                character_name=info.character_name,
            )
        )
    entries.sort(key=lambda e: (e.username, e.character_name or "", e.character_id or ""))
    return entries


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_registry: PresenceRegistry | None = None


def get_presence_registry() -> PresenceRegistry:
    """Return the process-wide :class:`PresenceRegistry`, building on
    first call. Tests reset it via :func:`reset_for_tests`."""

    global _registry
    if _registry is None:
        _registry = PresenceRegistry()
    return _registry


def reset_for_tests() -> None:
    """Drop the singleton so each test starts with empty state."""

    global _registry
    _registry = None


__all__ = [
    "PresenceRegistry",
    "get_presence_registry",
    "reset_for_tests",
]
