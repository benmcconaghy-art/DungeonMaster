"""Session endpoints (Phase 2 surface).

Phase 2 needs:
  - POST   /api/campaigns/{id}/sessions  — start a new session
  - POST   /api/sessions/{id}/end        — close it out
  - GET    /api/sessions/{id}/messages   — list messages (chronological,
                                            paginated by ``before`` cursor)
  - GET    /api/sessions/{id}            — current snapshot
  - POST   /api/sessions/{id}/action     — submit a player action,
                                            returns DM events via SSE
                                            (added after the orchestrator
                                            handler module lands; lives
                                            in app/api/sse.py)
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db import models
from app.deps import CurrentUser, DbSession

router = APIRouter(tags=["sessions"])


# ---------- response models -------------------------------------------------


class SessionSnapshot(BaseModel):
    """Compact status of an in-progress session."""

    id: str
    campaign_id: str
    started_at: str
    ended_at: str | None
    summary: str | None
    current_location_id: str | None
    is_active: bool


class SessionMessageResponse(BaseModel):
    """One persisted utterance — player or DM."""

    id: str
    sender_kind: str
    sender_id: str | None
    content: str
    audience: list[str]
    created_at: str


def _session_to_snapshot(session: models.Session) -> SessionSnapshot:
    return SessionSnapshot(
        id=session.id,
        campaign_id=session.campaign_id,
        started_at=session.started_at,
        ended_at=session.ended_at,
        summary=session.summary,
        current_location_id=session.current_location_id,
        is_active=session.ended_at is None,
    )


def _message_to_response(message: models.SessionMessage) -> SessionMessageResponse:
    return SessionMessageResponse(
        id=message.id,
        sender_kind=message.sender_kind,
        sender_id=message.sender_id,
        content=message.content,
        audience=list(message.audience),
        created_at=message.created_at,
    )


async def _require_session_membership(
    db: DbSession,
    session_id: str,
    user: models.User,
) -> models.Session:
    """Resolve a session and verify the current user can access it.

    Raises 404 if the session doesn't exist; 403 if the user isn't a
    member of the parent campaign.
    """

    session = await db.get(models.Session, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found")
    membership = await db.get(models.CampaignMember, (session.campaign_id, user.id))
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="not a member of this campaign",
        )
    return session


# ---------- routes ---------------------------------------------------------


@router.post(
    "/api/campaigns/{campaign_id}/sessions",
    response_model=SessionSnapshot,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    campaign_id: str,
    user: CurrentUser,
    db: DbSession,
) -> SessionSnapshot:
    """Start a new session in ``campaign_id``."""

    campaign = await db.get(models.Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="campaign not found")
    membership = await db.get(models.CampaignMember, (campaign_id, user.id))
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="not a member of this campaign",
        )

    session = models.Session(campaign_id=campaign_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return _session_to_snapshot(session)


@router.get("/api/sessions/{session_id}", response_model=SessionSnapshot)
async def get_session(
    session_id: str,
    user: CurrentUser,
    db: DbSession,
) -> SessionSnapshot:
    """Return the current snapshot of a session."""

    session = await _require_session_membership(db, session_id, user)
    return _session_to_snapshot(session)


@router.post("/api/sessions/{session_id}/end", response_model=SessionSnapshot)
async def end_session(
    session_id: str,
    user: CurrentUser,
    db: DbSession,
) -> SessionSnapshot:
    """Close a session. Idempotent — calling on an already-ended session
    is a no-op rather than an error."""

    session = await _require_session_membership(db, session_id, user)
    if session.ended_at is None:
        session.ended_at = (
            datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        )
        await db.commit()
        await db.refresh(session)
    return _session_to_snapshot(session)


@router.get(
    "/api/sessions/{session_id}/messages",
    response_model=list[SessionMessageResponse],
)
async def list_messages(
    session_id: str,
    user: CurrentUser,
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=500),
    before: str | None = Query(
        default=None,
        description="Optional ISO timestamp; return messages strictly older than this.",
    ),
) -> list[SessionMessageResponse]:
    """Return messages for a session, newest-last.

    Implements the cursor pagination from spec §11 minimally — pass
    ``before=<created_at>`` to fetch the previous page.
    """

    await _require_session_membership(db, session_id, user)

    statement = select(models.SessionMessage).where(models.SessionMessage.session_id == session_id)
    if before is not None:
        statement = statement.where(models.SessionMessage.created_at < before)
    statement = statement.order_by(models.SessionMessage.created_at.desc()).limit(limit)

    result = await db.execute(statement)
    rows = list(result.scalars())
    rows.reverse()  # caller wants chronological

    # Filter whispers the requesting user shouldn't see. ``audience=[]``
    # means table-wide; otherwise the user's character_ids must be in the
    # audience list.
    user_character_ids = await _user_character_ids(db, session_id, user)
    return [
        _message_to_response(m)
        for m in rows
        if not m.audience or any(cid in user_character_ids for cid in m.audience)
    ]


async def _user_character_ids(
    db: DbSession,
    session_id: str,
    user: models.User,
) -> set[str]:
    """Set of character ids belonging to ``user`` in the parent campaign."""

    session = await db.get(models.Session, session_id)
    if session is None:
        return set()
    statement = select(models.Character.id).where(
        models.Character.campaign_id == session.campaign_id,
        models.Character.user_id == user.id,
    )
    result = await db.execute(statement)
    return set(result.scalars())


# ---------- player-action submission ----------------------------------------


class SubmitActionRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    character_id: str | None = Field(
        default=None,
        description="Acting character. Optional in single-player; required in multiplayer.",
    )


__all__ = [
    "SessionMessageResponse",
    "SessionSnapshot",
    "SubmitActionRequest",
    "router",
]
