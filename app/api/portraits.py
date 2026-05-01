"""Canonical portrait endpoints (Phase 5 step 7).

Two endpoints, one each for PCs and NPCs:

- ``POST /api/characters/{character_id}/portrait`` — request a
  canonical portrait for a player character. Idempotent in spirit
  (the worker dedupes on ``prompt_hash``) but always returns a fresh
  job id; the caller can poll ``characters.canonical_image_id`` to
  see when it lands.
- ``POST /api/npcs/{npc_id}/portrait`` — same for NPCs that the
  spawn_npc tool didn't auto-portrait (e.g. NPCs created via a
  module import, or explicit operator request).

The endpoint enqueues a job — it doesn't wait for the image. The
client either polls or receives the eventual ``image_ready`` over
the session WebSocket if the campaign has an active session.

Authorisation: the caller must be a member of the character's /
NPC's campaign. Reading the FK off the row gives us the campaign
id without needing it in the URL — saves a round of indirection
for the client.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.db import models
from app.deps import CurrentUser, DbSession
from app.images.portrait import build_portrait_prompt, enqueue_portrait, get_queue_client

router = APIRouter(prefix="/api", tags=["portraits"])
log = logging.getLogger(__name__)


class PortraitRequest(BaseModel):
    """Optional override hook for the auto-composed prompt.

    The default prompt is built from the character / NPC fields via
    :func:`app.images.portrait.build_portrait_prompt`. A caller can
    pass an explicit ``prompt`` to override (UI lets the player tune
    "more rugged, scar across left eye"), and an explicit
    ``description`` to enrich the auto-composed one without writing
    the whole prompt from scratch.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str | None = Field(
        default=None,
        max_length=2000,
        description="Explicit prompt; overrides the auto-composed one if set.",
    )
    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Extra description appended to the auto-composed prompt.",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Session to broadcast image_ready into. Optional —"
            " character creation outside an active session leaves this null."
        ),
    )


class PortraitResponse(BaseModel):
    """The pre-allocated image id the worker will commit to disk and DB.

    The client uses this to render an ``image_pending`` placeholder
    immediately and swap it for the real ``image_ready`` event when
    the worker publishes (or polls ``characters.canonical_image_id``
    if it doesn't have an active session).
    """

    image_id: str
    prompt: str


async def _require_campaign_member(db: DbSession, *, user_id: str, campaign_id: str) -> None:
    """Reject 403 if the user isn't a member of the named campaign.

    Sole authorisation gate for portrait requests — once you're in
    the campaign you can request portraits for any of its characters
    or NPCs (campaign-level permissions, not character-level).
    """

    membership = await db.get(models.CampaignMember, (campaign_id, user_id))
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="not a member of this campaign",
        )


@router.post(
    "/characters/{character_id}/portrait",
    response_model=PortraitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_character_portrait(
    character_id: str,
    payload: PortraitRequest,
    user: CurrentUser,
    db: DbSession,
) -> PortraitResponse:
    """Enqueue a canonical portrait job for the named character.

    202 Accepted because the work is async — the row + file land
    when the worker finishes. The response carries the image id the
    worker will commit so the client can wire optimistic UI.
    """

    character = await db.get(models.Character, character_id)
    if character is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="character not found")
    await _require_campaign_member(db, user_id=user.id, campaign_id=character.campaign_id)

    prompt = payload.prompt or build_portrait_prompt(
        name=character.name,
        race=character.race,
        class_name=character.class_name,
        alignment=character.alignment,
        description=payload.description,
    )
    image_id = await enqueue_portrait(
        get_queue_client(),
        campaign_id=character.campaign_id,
        prompt=prompt,
        session_id=payload.session_id,
        subject_character_id=character.id,
    )
    return PortraitResponse(image_id=image_id, prompt=prompt)


@router.post(
    "/npcs/{npc_id}/portrait",
    response_model=PortraitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_npc_portrait(
    npc_id: str,
    payload: PortraitRequest,
    user: CurrentUser,
    db: DbSession,
) -> PortraitResponse:
    """Enqueue a canonical portrait job for the named NPC.

    Same shape as the character endpoint — separated so the FK link
    is unambiguous (one image is canonical for one subject)."""

    npc = await db.get(models.Npc, npc_id)
    if npc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="npc not found")
    await _require_campaign_member(db, user_id=user.id, campaign_id=npc.campaign_id)

    prompt = payload.prompt or build_portrait_prompt(
        name=npc.name,
        description=payload.description or npc.description,
    )
    image_id = await enqueue_portrait(
        get_queue_client(),
        campaign_id=npc.campaign_id,
        prompt=prompt,
        session_id=payload.session_id,
        subject_npc_id=npc.id,
    )
    return PortraitResponse(image_id=image_id, prompt=prompt)


__all__ = ["PortraitRequest", "PortraitResponse", "router"]
