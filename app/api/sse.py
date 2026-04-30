"""SSE bridge between the orchestrator's ``take_turn`` async generator and
the browser.

Phase 2 is single-player text-only, so a half-duplex stream is enough —
the action goes up as a query string on a ``GET /api/sessions/{id}/events``
request and the DM events come back as Server-Sent Events. Phase 4 will
upgrade to WebSocket for multi-player.

The endpoint:

  GET  /api/sessions/{session_id}/events?content=...&character_id=...

returns a ``text/event-stream`` response. Each ``DmEvent`` from
``take_turn`` is serialised as one SSE event whose ``event:`` line is the
event type and ``data:`` is the JSON payload (the event minus its
``type`` discriminator). The orchestrator yields ``WhisperEvent`` only
to the addressed audience; the bridge drops whisper events whose
audience doesn't include any of the requesting user's character ids.

A final ``event: turn_done`` is appended so the browser knows it can
close the EventSource.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.sessions import _require_session_membership
from app.db import models
from app.db.session import SessionLocal
from app.deps import CurrentUser, DbSession
from app.orchestrator.dm import DmError, WhisperEvent, take_turn

router = APIRouter(tags=["sessions"])


def _serialise_event(event_type: str, payload: dict[str, Any]) -> bytes:
    """Render one SSE frame: ``event: <type>\\ndata: <json>\\n\\n``.

    SSE frames end in a blank line; clients buffer until that's seen.
    Bytes (not str) so the StreamingResponse doesn't have to encode at
    every yield.
    """

    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


async def _user_character_ids(db: Any, session: models.Session, user_id: str) -> set[str]:
    """The acting user's character ids in the parent campaign — used to
    filter whispers."""

    statement = select(models.Character.id).where(
        models.Character.campaign_id == session.campaign_id,
        models.Character.user_id == user_id,
    )
    result = await db.execute(statement)
    return set(result.scalars())


async def _stream_events(
    *,
    session_id: str,
    sender_user_id: str,
    sender_character_id: str | None,
    content: str,
    visible_character_ids: set[str],
) -> AsyncIterator[bytes]:
    """The actual generator passed to ``StreamingResponse``.

    Opens its own ``AsyncSession`` so it doesn't compete for transaction
    state with FastAPI's per-request session (which already autobegan
    for the membership check at request entry). Each handler inside
    ``take_turn`` runs its own tight ``begin()`` block.
    """

    async with SessionLocal() as db:
        try:
            async for event in take_turn(
                db,
                session_id=session_id,
                sender_user_id=sender_user_id,
                sender_character_id=sender_character_id,
                content=content,
            ):
                # Whispers are filtered before serialisation: the bridge
                # only emits a whisper to clients in the addressed
                # audience. Phase 2 single-player means the requesting
                # user is the only client; the filter is forward
                # plumbing for Phase 4 multiplayer.
                if isinstance(event, WhisperEvent) and not any(
                    cid in visible_character_ids for cid in event.audience
                ):
                    continue

                payload = event.model_dump(mode="json", exclude={"type"})
                yield _serialise_event(event.type, payload)
        except Exception as exc:
            # The orchestrator already converts known failures into
            # ``dm_error`` events; anything that escapes is a bug we want
            # to surface to the client (and log) rather than dropping.
            err = DmError(reason="orchestrator_crash", message=str(exc))
            yield _serialise_event(err.type, err.model_dump(mode="json", exclude={"type"}))

        # Close-of-turn marker so the EventSource client knows to stop.
        yield _serialise_event("turn_done", {})


@router.get("/api/sessions/{session_id}/events")
async def session_events(
    session_id: str,
    user: CurrentUser,
    db: DbSession,
    content: str = Query(min_length=1, max_length=2000),
    character_id: str | None = Query(default=None),
) -> StreamingResponse:
    """Submit a player action and stream the DM's response as SSE.

    The membership check uses FastAPI's per-request session; the
    orchestrator runs against a fresh session opened inside the
    streaming generator (avoiding "transaction already begun" when the
    orchestrator opens its own ``async with db.begin():`` blocks).
    """

    session = await _require_session_membership(db, session_id, user)

    if character_id is not None:
        character = await db.get(models.Character, character_id)
        if character is None or character.campaign_id != session.campaign_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="character_id is not in this campaign",
            )
        if character.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="that character belongs to another player",
            )

    visible_character_ids = await _user_character_ids(db, session, user.id)

    return StreamingResponse(
        _stream_events(
            session_id=session_id,
            sender_user_id=user.id,
            sender_character_id=character_id,
            content=content,
            visible_character_ids=visible_character_ids,
        ),
        media_type="text/event-stream",
        headers={
            # Hint to nginx and proxies: don't buffer; flush per chunk.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
