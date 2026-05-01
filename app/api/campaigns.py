"""Campaign endpoints.

Phase 2 only had ``POST /api/campaigns`` (create). Phase 6 adds the
list / detail / invite / join surface the campaign-dashboard view
needs:

  - GET    /api/campaigns                  list mine
  - POST   /api/campaigns                  create (Phase 2)
  - GET    /api/campaigns/{id}             detail incl. characters,
                                           recent sessions, members
  - POST   /api/campaigns/{id}/invite      owner-only invite-code mint
  - POST   /api/campaigns/join             redeem an invite code

Invite codes are signed tokens (``itsdangerous.URLSafeTimedSerializer``)
rather than DB rows — keeps the schema lean and avoids a janitor job
for expired codes. Spec §11 lists the endpoints but doesn't pin the
lifecycle; trusted-LAN deployment per spec §13 means we can stay
simple. Defaults: 7-day TTL, multi-use within TTL, owner-only mint.
A future phase can promote to a row-backed surface if the audit trail
matters.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select

from app.config import get_settings
from app.db import models
from app.deps import CurrentUser, DbSession

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])


# ---------------------------------------------------------------------------
# Invite-code signer
# ---------------------------------------------------------------------------

# Salt namespaces this signer apart from the cookie-session signer
# (which uses Starlette's own SessionMiddleware secret derivation).
# Same secret, different cryptographic context.
_INVITE_SALT = "campaign-invite"
# Seven days is long enough for a player to redeem on a sane workday-
# scale schedule, short enough that a leaked code stops working
# eventually. Deployment that wants tighter (or looser) lifecycle can
# revisit; this is the v1 default.
_INVITE_MAX_AGE_S = 7 * 24 * 3600


def _invite_signer() -> URLSafeTimedSerializer:
    """Build a signer using the app session secret. Constructed per
    call so tests that override ``session_secret`` (via
    ``get_settings.cache_clear()``) see the new value immediately."""

    return URLSafeTimedSerializer(get_settings().session_secret, salt=_INVITE_SALT)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateCampaignRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class CampaignResponse(BaseModel):
    """Compact campaign shape returned by ``POST /api/campaigns`` and
    used as the row in ``GET /api/campaigns``."""

    id: str
    name: str
    owner_id: str
    ruleset: str
    created_at: str


class MemberSummary(BaseModel):
    """One member of a campaign, with the class names of the
    characters they own in this campaign (the design's dashboard
    AT THE TABLE row reads ``elara_v cleric``)."""

    user_id: str
    username: str
    role: str
    character_classes: list[str]


class CharacterCardSummary(BaseModel):
    """Subset of a character's fields for dashboard cards."""

    id: str
    name: str
    race: str
    class_name: str
    level: int
    hp_current: int
    hp_max: int
    ac: int
    xp: int
    status: str
    canonical_image_id: str | None
    is_mine: bool


class SessionSummary(BaseModel):
    """Recent-sessions panel entry. ``summary`` is the LLM-generated
    rolling summary persisted on the row (Phase 3) — italic prose
    in the DM's voice rendered as the card body."""

    id: str
    started_at: str
    ended_at: str | None
    summary: str | None


class CampaignListEntry(BaseModel):
    """One row in ``GET /api/campaigns``. ``most_recent`` flags the
    campaign whose latest session is most recent across all of the
    user's campaigns — drives the dashboard's "MOST RECENT CAMPAIGN"
    cap-tab on the featured card."""

    id: str
    name: str
    ruleset: str
    owner_id: str
    member_count: int
    my_character_count: int
    last_played_at: str | None
    has_active_session: bool
    most_recent: bool


class CampaignDetail(BaseModel):
    """Payload behind ``GET /api/campaigns/{id}``. Composes everything
    a single campaign card on the dashboard needs — characters,
    members, recent sessions — so the template doesn't have to fan out
    to multiple endpoints per row."""

    id: str
    name: str
    ruleset: str
    owner_id: str
    image_style: str | None
    members: list[MemberSummary]
    characters: list[CharacterCardSummary]
    recent_sessions: list[SessionSummary]
    active_session_id: str | None


class InviteResponse(BaseModel):
    """Owner-mint response. ``code`` is a signed URL-safe string the
    player pastes into the dashboard's join form. ``expires_in_seconds``
    documents the TTL so a UI can render "valid for 7d"."""

    code: str
    expires_in_seconds: int


class JoinRequest(BaseModel):
    code: str = Field(min_length=1, max_length=512)


class JoinResponse(BaseModel):
    """Redeem result — surfaces the campaign id so the caller can
    navigate to it without re-fetching the list."""

    campaign_id: str
    name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _campaign_to_response(campaign: models.Campaign) -> CampaignResponse:
    return CampaignResponse(
        id=campaign.id,
        name=campaign.name,
        owner_id=campaign.owner_id,
        ruleset=campaign.ruleset,
        created_at=campaign.created_at,
    )


async def _require_membership(
    db: DbSession, *, campaign_id: str, user: models.User
) -> tuple[models.Campaign, models.CampaignMember]:
    """Resolve campaign + membership; raise 404/403 the same way the
    play-screen route does so client error handling matches."""

    campaign = await db.get(models.Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="campaign not found")
    membership = await db.get(models.CampaignMember, (campaign_id, user.id))
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="not a member of this campaign",
        )
    return campaign, membership


async def _list_member_summaries(
    db: DbSession, *, campaign_id: str
) -> list[MemberSummary]:
    """Members for a single campaign, with the character classes each
    player owns inside it. Used by the dashboard's AT THE TABLE row."""

    rows = (
        await db.execute(
            select(models.CampaignMember, models.User)
            .join(models.User, models.User.id == models.CampaignMember.user_id)
            .where(models.CampaignMember.campaign_id == campaign_id)
            .order_by(models.User.username)
        )
    ).all()

    char_rows = (
        await db.execute(
            select(models.Character.user_id, models.Character.class_name)
            .where(models.Character.campaign_id == campaign_id)
        )
    ).all()
    classes_by_user: dict[str, list[str]] = {}
    for user_id, class_name in char_rows:
        classes_by_user.setdefault(user_id, []).append(class_name)

    return [
        MemberSummary(
            user_id=member.user_id,
            username=user.username,
            role=member.role,
            character_classes=classes_by_user.get(member.user_id, []),
        )
        for member, user in rows
    ]


async def _campaign_last_played(
    db: DbSession, *, campaign_id: str
) -> tuple[str | None, bool]:
    """Latest session activity for a campaign: ``(last_played_at,
    has_active_session)``. ``last_played_at`` is the most recent of
    ``ended_at`` (or ``started_at`` if still active); ``None`` if the
    campaign has no sessions yet."""

    row = (
        await db.execute(
            select(models.Session)
            .where(models.Session.campaign_id == campaign_id)
            .order_by(models.Session.started_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if row is None:
        return None, False
    last = row.ended_at or row.started_at
    return last, row.ended_at is None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=CampaignResponse, status_code=status.HTTP_201_CREATED)
async def create_campaign(
    payload: CreateCampaignRequest,
    user: CurrentUser,
    db: DbSession,
) -> CampaignResponse:
    """Create a campaign owned by the current user."""

    campaign = models.Campaign(name=payload.name, owner_id=user.id)
    db.add(campaign)
    await db.flush()
    db.add(
        models.CampaignMember(
            campaign_id=campaign.id,
            user_id=user.id,
            role="owner",
        )
    )
    await db.commit()
    await db.refresh(campaign)
    return _campaign_to_response(campaign)


@router.get("", response_model=list[CampaignListEntry])
async def list_campaigns(
    user: CurrentUser,
    db: DbSession,
) -> list[CampaignListEntry]:
    """List campaigns the current user is a member of, ordered by
    last activity (most recent first). The most recently played
    campaign is flagged so the dashboard can render it as the
    featured card."""

    member_rows = (
        await db.execute(
            select(models.CampaignMember.campaign_id)
            .where(models.CampaignMember.user_id == user.id)
        )
    ).scalars().all()
    if not member_rows:
        return []

    campaign_ids = list(member_rows)

    campaigns = (
        await db.execute(
            select(models.Campaign).where(models.Campaign.id.in_(campaign_ids))
        )
    ).scalars().all()

    # Member counts in one query.
    count_rows = (
        await db.execute(
            select(
                models.CampaignMember.campaign_id,
                func.count(models.CampaignMember.user_id),
            )
            .where(models.CampaignMember.campaign_id.in_(campaign_ids))
            .group_by(models.CampaignMember.campaign_id)
        )
    ).all()
    member_count_by_campaign = {cid: int(n) for cid, n in count_rows}

    my_char_rows = (
        await db.execute(
            select(
                models.Character.campaign_id,
                func.count(models.Character.id),
            )
            .where(models.Character.user_id == user.id)
            .where(models.Character.campaign_id.in_(campaign_ids))
            .group_by(models.Character.campaign_id)
        )
    ).all()
    my_char_by_campaign = {cid: int(n) for cid, n in my_char_rows}

    entries: list[tuple[CampaignListEntry, str | None]] = []
    for campaign in campaigns:
        last_played_at, has_active = await _campaign_last_played(db, campaign_id=campaign.id)
        entries.append(
            (
                CampaignListEntry(
                    id=campaign.id,
                    name=campaign.name,
                    ruleset=campaign.ruleset,
                    owner_id=campaign.owner_id,
                    member_count=member_count_by_campaign.get(campaign.id, 0),
                    my_character_count=my_char_by_campaign.get(campaign.id, 0),
                    last_played_at=last_played_at,
                    has_active_session=has_active,
                    most_recent=False,
                ),
                last_played_at,
            )
        )

    # Sort: campaigns with activity first (most recent last_played_at
    # at the top); never-played campaigns last, in insertion order.
    entries.sort(key=lambda pair: (pair[1] is None, pair[1] or ""), reverse=False)
    # Above sort puts None last (True > False) and ascending — but we
    # want most-recent first within the dated group. Re-sort: dated
    # entries by date desc, then undated.
    dated = sorted([e for e in entries if e[1] is not None], key=lambda p: p[1] or "", reverse=True)
    undated = [e for e in entries if e[1] is None]
    ordered = [e[0] for e in dated] + [e[0] for e in undated]

    if dated:
        ordered[0] = ordered[0].model_copy(update={"most_recent": True})

    return ordered


@router.get("/{campaign_id}", response_model=CampaignDetail)
async def get_campaign(
    campaign_id: str,
    user: CurrentUser,
    db: DbSession,
) -> CampaignDetail:
    """Full campaign detail — characters, members, recent sessions,
    active-session pointer if any."""

    campaign, _ = await _require_membership(db, campaign_id=campaign_id, user=user)
    members = await _list_member_summaries(db, campaign_id=campaign_id)

    char_rows = list(
        (
            await db.execute(
                select(models.Character)
                .where(models.Character.campaign_id == campaign_id)
                .order_by(models.Character.name)
            )
        ).scalars()
    )
    characters = [
        CharacterCardSummary(
            id=c.id,
            name=c.name,
            race=c.race,
            class_name=c.class_name,
            level=c.level,
            hp_current=c.hp_current,
            hp_max=c.hp_max,
            ac=c.ac,
            xp=c.xp,
            status=c.status,
            canonical_image_id=c.canonical_image_id,
            is_mine=c.user_id == user.id,
        )
        for c in char_rows
    ]

    session_rows = list(
        (
            await db.execute(
                select(models.Session)
                .where(models.Session.campaign_id == campaign_id)
                .order_by(desc(models.Session.started_at))
                .limit(8)
            )
        ).scalars()
    )
    recent_sessions = [
        SessionSummary(
            id=s.id,
            started_at=s.started_at,
            ended_at=s.ended_at,
            summary=s.summary,
        )
        for s in session_rows
    ]

    active = next((s for s in session_rows if s.ended_at is None), None)

    return CampaignDetail(
        id=campaign.id,
        name=campaign.name,
        ruleset=campaign.ruleset,
        owner_id=campaign.owner_id,
        image_style=campaign.image_style,
        members=members,
        characters=characters,
        recent_sessions=recent_sessions,
        active_session_id=active.id if active else None,
    )


@router.post(
    "/{campaign_id}/invite",
    response_model=InviteResponse,
    status_code=status.HTTP_201_CREATED,
)
async def mint_invite(
    campaign_id: str,
    user: CurrentUser,
    db: DbSession,
) -> InviteResponse:
    """Owner-only: mint a signed invite token for ``campaign_id``.

    Returns the token verbatim. The client renders it as a
    copy-pasteable code in the campaign card; players paste it into
    the dashboard's join form. The token is opaque — the server
    re-derives the campaign id by verifying the signature.
    """

    campaign, membership = await _require_membership(
        db, campaign_id=campaign_id, user=user
    )
    if membership.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="only the campaign owner can mint invites",
        )

    payload: dict[str, Any] = {"campaign_id": campaign.id, "by": user.id}
    code = _invite_signer().dumps(payload)
    return InviteResponse(code=code, expires_in_seconds=_INVITE_MAX_AGE_S)


@router.post("/join", response_model=JoinResponse)
async def join_via_invite(
    payload: JoinRequest,
    user: CurrentUser,
    db: DbSession,
) -> JoinResponse:
    """Redeem an invite code: add the current user as a player member
    of the encoded campaign. Idempotent — already-a-member is a
    successful no-op (returns the campaign, doesn't 409)."""

    try:
        decoded = _invite_signer().loads(payload.code, max_age=_INVITE_MAX_AGE_S)
    except SignatureExpired as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invite code has expired",
        ) from exc
    except BadSignature as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invite code is invalid",
        ) from exc

    if not isinstance(decoded, dict) or "campaign_id" not in decoded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invite code payload is malformed",
        )
    campaign_id = str(decoded["campaign_id"])

    campaign = await db.get(models.Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="campaign no longer exists",
        )

    existing = await db.get(models.CampaignMember, (campaign_id, user.id))
    if existing is None:
        db.add(
            models.CampaignMember(
                campaign_id=campaign_id,
                user_id=user.id,
                role="player",
            )
        )
        await db.commit()
    return JoinResponse(campaign_id=campaign.id, name=campaign.name)


__all__ = [
    "CampaignDetail",
    "CampaignListEntry",
    "CampaignResponse",
    "CharacterCardSummary",
    "CreateCampaignRequest",
    "InviteResponse",
    "JoinRequest",
    "JoinResponse",
    "MemberSummary",
    "SessionSummary",
    "router",
]
