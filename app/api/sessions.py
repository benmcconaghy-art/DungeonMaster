"""Session endpoints.

REST surface for session lifecycle and history. Realtime — player
actions, narration, dice, presence — runs over the WS hub at
``/ws/session/{session_id}`` (see ``app/api/ws.py``).

  - POST   /api/campaigns/{id}/sessions  — start a new session
  - POST   /api/sessions/{id}/end        — close it out
  - GET    /api/sessions/{id}/messages   — list messages (chronological,
                                            paginated by ``before`` cursor)
  - GET    /api/sessions/{id}            — current snapshot

Phase 6.8: ``create_session`` schedules a background opening DM turn
so the player lands on the table view with the scene already setting
itself instead of an indefinite "DM is preparing the scene…"
placeholder. The turn dispatches via the shared orchestrator helper
in ``app/orchestrator/dispatch.py`` (same lock + pubsub fan-out as
the WS hub uses for player actions). The persisted DM message lands
in ``session_messages`` regardless of whether a WS subscriber has
connected yet — the snapshot path covers late connections; live
streaming covers fast connections.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from app.db import models
from app.deps import CurrentUser, DbSession
from app.llm.client import DmClientError, get_dm_client
from app.llm.memory import regenerate_campaign_summary
from app.llm.modules import ModuleContent
from app.orchestrator.dispatch import run_dm_turn

log = logging.getLogger(__name__)

router = APIRouter(tags=["sessions"])

# Bootstrapping directive injected as the leading message of an
# auto-greeting turn. Persisted with sender_kind='system' so future
# prompts surface it as engine context, not as user input the DM
# should "answer". Phrased so the model treats it as a stage
# direction: set the scene, lean on the campaign + location +
# party context that the prompt builder already includes.
_OPENING_DIRECTIVE = (
    "[Session begins — set the opening scene for the party. Use the campaign,"
    " current location, and active PCs from the prompt context to ground the"
    " narration. End with a natural beat that invites the players to act.]"
)

# Module-level set keeps strong refs to in-flight opening tasks so
# asyncio's create_task can't drop them mid-flight when the only
# reference would have been a local that goes out of scope as the
# HTTP handler returns.
_OPENING_TASKS: set[asyncio.Task[Any]] = set()


def _schedule_opening_turn(session_id: str, *, sender_user_id: str) -> None:
    """Fire the auto-greeting in the background. The HTTP handler
    returns immediately so the browser proceeds to the table view; the
    DM turn streams through pubsub to whichever WS connects in time
    and persists the canonical DM message either way."""

    task = asyncio.create_task(
        run_dm_turn(
            session_id=session_id,
            sender_user_id=sender_user_id,
            sender_character_id=None,
            content=_OPENING_DIRECTIVE,
            opening=True,
        )
    )
    _OPENING_TASKS.add(task)
    task.add_done_callback(_OPENING_TASKS.discard)


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

    # Phase 6.8 Bug 3: kick off the opening DM turn in the background
    # so the player lands on a setting-itself scene instead of the
    # placeholder. The dispatch helper takes the per-session lock so
    # a fast first player action serialises behind the opening rather
    # than racing it.
    _schedule_opening_turn(session.id, sender_user_id=user.id)

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

        # Regenerate the campaign-level rolling summary now that this
        # session's contribution is final. Awaited (not fired-and-forgotten)
        # because "End session" is an explicit user action and the operator
        # expects the summary to be in place once the response returns.
        # The function manages its own transaction discipline.
        try:
            await regenerate_campaign_summary(db, campaign_id=session.campaign_id)
        except DmClientError:
            log.exception(
                "end_session: campaign summary regeneration failed for %s",
                session.campaign_id,
            )
            # Don't fail the close on a summary-regen blip; the session is
            # ended and the operator can re-trigger summarisation later.
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


# ---------------------------------------------------------------------------
# Module extraction
# ---------------------------------------------------------------------------

_EXTRACTION_MAX_RETRIES = 3

_EXTRACTION_SYSTEM = (
    "You are a TTRPG module author. You will be given a summary of a played session "
    "and must extract a reusable adventure module from it.\n\n"
    "Output ONLY a valid JSON object matching the ModuleContent schema below. "
    "No prose, no markdown fences, no explanation — pure JSON.\n\n"
    "ModuleContent schema fields:\n"
    '  format_version: "1.0"\n'
    "  synopsis: one-paragraph summary of the adventure\n"
    "  tone: one or two words (e.g. 'gritty', 'heroic', 'mysterious')\n"
    "  image_style: FLUX style prompt (e.g. 'dark fantasy ink illustration')\n"
    "  image_negative_prompt: FLUX negative prompt\n"
    "  level_range: [min_level, max_level] integers\n"
    "  estimated_sessions: integer\n"
    "  starting_hook: opening hook paragraph\n"
    "  starting_location_symbol: loc_ symbol of starting location\n"
    "  locations: [{symbol, name, description, parent_symbol?, image_role?, metadata?}]\n"
    "  npcs: [{symbol, name, description, motivation, starting_location_symbol, "
    "stats?, sample_dialogue?, image_role?, secrets?}]\n"
    "  encounters: [{symbol, name, trigger_hint, monsters: [{name, count, tactics?}], "
    "treasure_hint?}]\n"
    "  plot_beats: [{symbol, title, trigger_hint, outcome, leads_to?, dm_notes?}]\n"
    "  secrets: [{symbol, content, reveal_when, leads_to_beat?}]\n"
    "  endings: [{symbol, trigger, outcome}]\n"
    "  world_facts: [{fact, tags, importance}]\n\n"
    "Symbol naming: loc_ for locations, npc_ for NPCs, enc_ for encounters, "
    "beat_ for plot beats, sec_ for secrets, end_ for endings. All snake_case.\n\n"
    "Privacy: do NOT include player character names, dice results, or session-specific "
    "details. Distill only the reusable adventure structure."
)


def _build_extraction_prompt(
    *,
    campaign_name: str,
    locations: list[models.Location],
    npcs: list[models.Npc],
    messages: list[models.SessionMessage],
    characters: list[models.Character],
) -> list[dict[str, Any]]:
    """Build the prompt for module extraction from session data."""

    loc_block = "\n".join(
        f"  - {loc.name}: {loc.description or '(no description)'}" for loc in locations
    )
    npc_block = "\n".join(
        f"  - {npc.name}: {npc.description or '(no description)'}" for npc in npcs
    )

    # Filter to player+dm messages only; strip PC-specific attribution.
    transcript_lines: list[str] = []
    char_ids = {ch.id: ch.name for ch in characters}
    for msg in messages:
        if msg.sender_kind == "player":
            speaker = char_ids.get(msg.sender_id or "", "Player")
            transcript_lines.append(f"[{speaker}]: {msg.content}")
        elif msg.sender_kind == "dm":
            transcript_lines.append(f"[DM]: {msg.content}")
    # Cap transcript to avoid hitting token limits; keep last 200 exchanges.
    transcript = "\n".join(transcript_lines[-400:])

    user_content = (
        f"Campaign: {campaign_name}\n\n"
        f"Locations visited:\n{loc_block or '(none recorded)'}\n\n"
        f"NPCs encountered:\n{npc_block or '(none recorded)'}\n\n"
        f"Session transcript (excerpt):\n{transcript or '(no messages)'}\n\n"
        "Extract a reusable adventure module as valid JSON per the schema above."
    )

    return [
        {"role": "system", "content": _EXTRACTION_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _extract_json(raw: str) -> str:
    """Strip markdown fences and extract the JSON object from LLM output."""
    # Remove ```json ... ``` or ``` ... ``` fences.
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    # If the model still wrapped in a think block, grab after </think>.
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[-1].strip()
    return raw


class ExtractModuleResponse(BaseModel):
    module_id: str
    name: str
    synopsis: str


@router.post(
    "/api/sessions/{session_id}/extract-module",
    response_model=ExtractModuleResponse,
    status_code=status.HTTP_201_CREATED,
)
async def extract_module(
    session_id: str,
    user: CurrentUser,
    db: DbSession,
) -> ExtractModuleResponse:
    """Extract a reusable module from a completed session.

    Requires the session to be ended (ended_at IS NOT NULL) and the
    caller to be the campaign owner. Calls the LLM with reasoning_mode
    full, validates against ModuleContent, retries up to 3 times on
    ValidationError. On success inserts a Module row with
    author_id=user.id, source_session_id=session_id, public=False.
    """
    session = await db.get(models.Session, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found")

    # Ownership check: must be campaign owner.
    membership = await db.get(models.CampaignMember, (session.campaign_id, user.id))
    if membership is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="not a campaign member")
    if membership.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="only the campaign owner can extract a module",
        )

    if session.ended_at is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="session must be ended before extracting a module",
        )

    campaign = await db.get(models.Campaign, session.campaign_id)
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="campaign not found")

    # Load session data.
    messages = list(
        (
            await db.scalars(
                select(models.SessionMessage)
                .where(models.SessionMessage.session_id == session_id)
                .where(models.SessionMessage.sender_kind.in_(["player", "dm"]))
                .order_by(models.SessionMessage.created_at)
            )
        ).all()
    )
    locations = list(
        (
            await db.scalars(
                select(models.Location).where(models.Location.campaign_id == session.campaign_id)
            )
        ).all()
    )
    npcs = list(
        (
            await db.scalars(
                select(models.Npc).where(models.Npc.campaign_id == session.campaign_id)
            )
        ).all()
    )
    characters = list(
        (
            await db.scalars(
                select(models.Character).where(
                    models.Character.campaign_id == session.campaign_id
                )
            )
        ).all()
    )

    prompt = _build_extraction_prompt(
        campaign_name=campaign.name,
        locations=locations,
        npcs=npcs,
        messages=messages,
        characters=characters,
    )

    client = get_dm_client()
    last_error: Exception | None = None
    corrective: str | None = None

    for attempt in range(1, _EXTRACTION_MAX_RETRIES + 1):
        if corrective:
            # Append the corrective message for retry attempts.
            prompt = list(prompt) + [{"role": "user", "content": corrective}]

        try:
            raw = await client.complete(
                prompt,
                response_format={"type": "json_object"},
                reasoning_mode="full",
                max_tokens=4096,
                temperature=0.3,
            )
        except DmClientError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM call failed during extraction: {exc}",
            ) from exc

        json_str = _extract_json(raw)
        try:
            data = json.loads(json_str)
            content = ModuleContent.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            log.warning(
                "extract_module: attempt %d/%d failed validation: %s",
                attempt,
                _EXTRACTION_MAX_RETRIES,
                exc,
            )
            corrective = (
                f"The previous response failed validation:\n{exc}\n\n"
                "Please fix the JSON and return only the corrected JSON object."
            )
            continue

        # Success: insert the module row.
        module_name = f"{campaign.name} (extracted)"
        module_row = models.Module(
            author_id=user.id,
            name=module_name,
            description=content.synopsis[:200] if content.synopsis else None,
            min_level=content.level_range[0] if content.level_range else None,
            max_level=content.level_range[-1] if len(content.level_range) > 1 else None,
            tone=content.tone,
            estimated_sessions=content.estimated_sessions,
            content=content.model_dump(mode="json"),
            source_session_id=session_id,
            public=False,
        )
        db.add(module_row)
        await db.commit()
        await db.refresh(module_row)

        return ExtractModuleResponse(
            module_id=module_row.id,
            name=module_name,
            synopsis=content.synopsis,
        )

    # All retries exhausted.
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=(
            f"Module extraction failed after {_EXTRACTION_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        ),
    )


__all__ = [
    "ExtractModuleResponse",
    "SessionMessageResponse",
    "SessionSnapshot",
    "SubmitActionRequest",
    "router",
]
