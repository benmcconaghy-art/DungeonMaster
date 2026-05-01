"""Cross-cutting helpers for running DM turns from outside the WS hub.

The WS hub (``app/api/ws.py``) was the only entry point that needed a
per-session lock and a pubsub publisher around :func:`take_turn`. Phase
6.8 added a second entry point — the auto-greeting fired from
``POST /api/campaigns/{id}/sessions`` — which needs the same
serialisation (so the opening turn doesn't race a player who connects
quickly and sends an immediate action) and the same pubsub fan-out (so
the live narration reaches any connected client).

This module owns the per-session ``asyncio.Lock`` registry and the
``run_dm_turn`` helper that ties the lock + pubsub publish around a
``take_turn`` invocation. Both ws.py and the auto-greeting path call
in here.

The lock is process-local; spec §13's single-gunicorn-worker
deployment makes that correct today. The AGENTS.md follow-up
"Cross-worker shared state" already tracks the multi-worker upgrade
path (Valkey-backed distributed lock keyed by session_id).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from app.db.session import SessionLocal
from app.orchestrator.dm import take_turn
from app.realtime import messages as ws_msgs
from app.realtime.bridge import orchestrator_event_to_ws
from app.realtime.pubsub import DmPubsubError, get_pubsub

log = logging.getLogger(__name__)


# Per-session lock for the orchestrator. Two turn requests on the same
# session serialise; turn N+1 waits for turn N to finish before
# building its prompt. Used by the WS hub for player actions and by
# the auto-greeting path on session creation.
_session_turn_locks: dict[str, asyncio.Lock] = {}


def session_lock(session_id: str) -> asyncio.Lock:
    """Return (or lazily create) the per-session orchestrator lock."""

    lock = _session_turn_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_turn_locks[session_id] = lock
    return lock


async def run_dm_turn(
    *,
    session_id: str,
    sender_user_id: str,
    sender_character_id: str | None,
    content: str,
    opening: bool = False,
) -> None:
    """Run :func:`take_turn` and publish each yielded event to the
    session's Valkey channel.

    Holds the per-session orchestrator lock so concurrent turn
    requests on the same session serialise. Opens its own DB session
    so the caller doesn't have to manage SQLAlchemy lifetime around
    the streaming call. ``opening=True`` runs the auto-greeting path:
    no player message is persisted, the prompt instead carries a
    system pseudo-message instructing the DM to set the opening scene.

    Failures inside ``take_turn`` are caught and a generic
    ``orchestrator_crash`` ``dm_error`` is broadcast so the table sees
    something rather than silence. Pubsub failures stop the turn —
    without the broadcast the narration only reaches the originator's
    subscriber if Valkey came back; cleaner to surface and return.
    """

    pubsub = get_pubsub()
    lock = session_lock(session_id)
    async with lock, SessionLocal() as db:
        try:
            async for event in take_turn(
                db,
                session_id=session_id,
                sender_user_id=sender_user_id,
                sender_character_id=sender_character_id,
                content=content,
                opening=opening,
            ):
                ws_msg = orchestrator_event_to_ws(event)
                if ws_msg is None:
                    continue
                try:
                    await pubsub.publish(session_id, ws_msg)
                except DmPubsubError:
                    log.exception("dispatch: pubsub publish failed mid-turn")
                    return
        except Exception:
            log.exception("dispatch: take_turn raised an unexpected error")
            with contextlib.suppress(DmPubsubError):
                await pubsub.publish(
                    session_id,
                    ws_msgs.DmError(
                        reason="orchestrator_crash",
                        message="DM turn crashed; the table has been notified.",
                    ),
                )


__all__ = ["run_dm_turn", "session_lock"]
