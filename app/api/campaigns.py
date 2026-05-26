"""Campaign endpoints.

Phase 2 only had ``POST /api/campaigns`` (create). Phase 6 added the
list / detail / invite / join surface the campaign-dashboard view
needs. Phase 7 promotes invites from stateless signed tokens to
row-backed single-use codes with audit + revocation:

  - GET    /api/campaigns                       list mine
  - POST   /api/campaigns                       create (Phase 2)
  - GET    /api/campaigns/{id}                  detail incl. characters,
                                                recent sessions, members
  - POST   /api/campaigns/{id}/invite           owner-only invite-code mint
  - GET    /api/campaigns/{id}/invites          owner-only: list invites + state
  - DELETE /api/campaigns/invites/{invite_id}   owner-only: revoke
  - POST   /api/campaigns/join                  redeem an invite code

Invite token shape (Phase 7): ``{"invite_id": <uuid>, "campaign_id":
<uuid>}`` signed with the app session secret + a salt. The redeem
endpoint looks up the row to confirm it exists, isn't revoked, isn't
expired, and isn't already used. Single-use semantics: once redeemed,
the row's ``used_by`` / ``used_at`` are set and further redemptions
return 400.

Legacy grace: Phase 6 in-flight tokens (``{"campaign_id", "by"}`` shape,
no ``invite_id``) are accepted via the old verification path until
``_LEGACY_GRACE_END`` (7 days post-deploy), with a deprecation warning
logged per redemption. After that cutoff, legacy tokens 400.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

import hashlib

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import desc, func, select
from uuid_extensions import uuid7

from app.config import get_settings
from app.db import models
from app.deps import CurrentUser, DbSession
from app.llm.embeddings import get_embedder
from app.llm.modules import ModuleContent
from app.ratelimit import join_rate_limit

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

# Legacy grace cutoff for Phase 6 stateless tokens. Tokens that decode to
# the old ``{"campaign_id", "by"}`` shape (no ``invite_id``) are accepted
# until this timestamp, with a deprecation warning logged on each use.
# After the cutoff they 400. Tests monkeypatch this value to exercise
# both the in-grace and post-grace paths.
_LEGACY_GRACE_END = "2026-05-08T00:00:00Z"


def _invite_signer() -> URLSafeTimedSerializer:
    """Build a signer using the app session secret. Constructed per
    call so tests that override ``session_secret`` (via
    ``get_settings.cache_clear()``) see the new value immediately."""

    return URLSafeTimedSerializer(get_settings().session_secret, salt=_INVITE_SALT)


def _now_iso() -> str:
    """ISO-8601 UTC timestamp with millisecond precision and a trailing
    ``Z``. Matches the ``strftime('%Y-%m-%dT%H:%M:%fZ','now')`` shape
    SQLite emits for server-default timestamps so all timestamp string
    comparisons stay textually orderable."""

    now = _dt.datetime.now(_dt.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _expires_iso(now: _dt.datetime | None = None) -> str:
    """Compute the absolute expiry timestamp 7 days from ``now``."""

    when = (now or _dt.datetime.now(_dt.UTC)) + _dt.timedelta(seconds=_INVITE_MAX_AGE_S)
    return when.strftime("%Y-%m-%dT%H:%M:%S.") + f"{when.microsecond // 1000:03d}Z"


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
    documents the TTL so a UI can render "valid for 7d". ``invite_id``
    is the row id — exposed so a UI can surface it in a "manage
    invites" view alongside the list endpoint's entries."""

    code: str
    expires_in_seconds: int
    invite_id: str


class InviteListEntry(BaseModel):
    """One row in ``GET /api/campaigns/{id}/invites``. Surfaces enough
    state for an owner-side UI to show "active / used by Bob / revoked".
    The ``code`` field is intentionally absent: the signed token is the
    secret, returned only at mint time. After that, only the row id
    travels — revoke happens by id, not by re-pasting the code.
    """

    invite_id: str
    created_by: str
    created_at: str
    expires_at: str
    revoked_at: str | None
    used_by: str | None
    used_at: str | None
    state: str  # "active" | "used" | "revoked" | "expired"


class JoinRequest(BaseModel):
    code: str = Field(min_length=1, max_length=512)


class JoinResponse(BaseModel):
    """Redeem result — surfaces the campaign id so the caller can
    navigate to it without re-fetching the list."""

    campaign_id: str
    name: str


class LoadModuleRequest(BaseModel):
    """Body for POST /api/campaigns/from-module."""

    module_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    image_style_override: str | None = None


class LoadModuleResponse(BaseModel):
    """Response for a successful module load."""

    campaign_id: str
    name: str
    locations_created: int
    npcs_created: int
    image_jobs_enqueued: int


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


async def _list_member_summaries(db: DbSession, *, campaign_id: str) -> list[MemberSummary]:
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
            select(models.Character.user_id, models.Character.class_name).where(
                models.Character.campaign_id == campaign_id
            )
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


async def _campaign_last_played(db: DbSession, *, campaign_id: str) -> tuple[str | None, bool]:
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
        (
            await db.execute(
                select(models.CampaignMember.campaign_id).where(
                    models.CampaignMember.user_id == user.id
                )
            )
        )
        .scalars()
        .all()
    )
    if not member_rows:
        return []

    campaign_ids = list(member_rows)

    campaigns = (
        (await db.execute(select(models.Campaign).where(models.Campaign.id.in_(campaign_ids))))
        .scalars()
        .all()
    )

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


def _invite_state(invite: models.CampaignInvite, *, now_iso: str) -> str:
    """Classify an invite row for the list endpoint.

    Order matters: revoked dominates used dominates expired. A row that's
    both revoked and used reports "revoked" because that's the operator-
    visible action; "used" still appears in ``used_by`` for the audit.
    """

    if invite.revoked_at is not None:
        return "revoked"
    if invite.used_at is not None:
        return "used"
    if invite.expires_at <= now_iso:
        return "expired"
    return "active"


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
    """Owner-only: mint a row-backed single-use invite token for ``campaign_id``.

    Inserts a ``campaign_invites`` row with a 7-day TTL, then signs a
    token of shape ``{"invite_id": <id>, "campaign_id": <campaign_id>}``.
    The redeem endpoint looks up the row by id to confirm it exists,
    isn't revoked, isn't expired, and isn't already used.
    """

    campaign, membership = await _require_membership(db, campaign_id=campaign_id, user=user)
    if membership.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="only the campaign owner can mint invites",
        )

    invite = models.CampaignInvite(
        campaign_id=campaign.id,
        created_by=user.id,
        expires_at=_expires_iso(),
    )
    db.add(invite)
    await db.commit()
    await db.refresh(invite)

    payload: dict[str, Any] = {
        "invite_id": invite.id,
        "campaign_id": campaign.id,
    }
    code = _invite_signer().dumps(payload)
    return InviteResponse(
        code=code,
        expires_in_seconds=_INVITE_MAX_AGE_S,
        invite_id=invite.id,
    )


@router.get(
    "/{campaign_id}/invites",
    response_model=list[InviteListEntry],
)
async def list_invites(
    campaign_id: str,
    user: CurrentUser,
    db: DbSession,
) -> list[InviteListEntry]:
    """Owner-only: enumerate all invites for the campaign with their
    audit + lifecycle state. Drives a "manage invites" UI surface;
    the wire shape is owner-private (it includes who used what)."""

    campaign, membership = await _require_membership(db, campaign_id=campaign_id, user=user)
    if membership.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="only the campaign owner can list invites",
        )

    rows = list(
        (
            await db.scalars(
                select(models.CampaignInvite)
                .where(models.CampaignInvite.campaign_id == campaign.id)
                .order_by(desc(models.CampaignInvite.created_at))
            )
        ).all()
    )
    now = _now_iso()
    return [
        InviteListEntry(
            invite_id=invite.id,
            created_by=invite.created_by,
            created_at=invite.created_at,
            expires_at=invite.expires_at,
            revoked_at=invite.revoked_at,
            used_by=invite.used_by,
            used_at=invite.used_at,
            state=_invite_state(invite, now_iso=now),
        )
        for invite in rows
    ]


@router.delete("/invites/{invite_id}")
async def revoke_invite(
    invite_id: str,
    user: CurrentUser,
    db: DbSession,
) -> Response:
    """Owner-only: revoke an invite by id. Idempotent — revoking an
    already-revoked invite is a 204 no-op (the desired state is the
    actual state). Revoking a used invite is allowed: it changes the
    row's lifecycle classification (per ``_invite_state``) so a
    "manage" view shows the operator action without erasing the audit.
    Trying to revoke a never-existed invite is 404; revoking another
    owner's invite is 403."""

    invite = await db.get(models.CampaignInvite, invite_id)
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="invite not found")
    # Authorisation: must be the owner of the invite's campaign.
    membership = await db.get(models.CampaignMember, (invite.campaign_id, user.id))
    if membership is None or membership.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="only the campaign owner can revoke invites",
        )
    if invite.revoked_at is None:
        invite.revoked_at = _now_iso()
        await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/join",
    response_model=JoinResponse,
    dependencies=[Depends(join_rate_limit)],
)
async def join_via_invite(
    payload: JoinRequest,
    user: CurrentUser,
    db: DbSession,
) -> JoinResponse:
    """Redeem an invite code: add the current user as a player member
    of the encoded campaign.

    Phase 7 path (token has ``invite_id``): look up the row, enforce
    single-use semantics. Reject revoked / expired / already-used.
    On success, set ``used_by`` / ``used_at`` and add a CampaignMember
    row. The whole sequence runs in one commit so a crash mid-flight
    can't leave a half-redeemed state.

    Legacy path (Phase 6 token, no ``invite_id``): until
    ``_LEGACY_GRACE_END``, accept the old ``campaign_id``/``by``
    payload with a deprecation warning. Past the cutoff, return 400
    with a "please request a new code" message.
    """

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

    invite_id = decoded.get("invite_id")
    if invite_id is None:
        # Legacy Phase 6 token. Accept until the cutoff; log a
        # deprecation warning either way so the audit log shows the
        # old shape is still in circulation.
        if _now_iso() >= _LEGACY_GRACE_END:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "invite code is from a deprecated format and the "
                    "grace period has ended; please ask the campaign "
                    "owner for a fresh code"
                ),
            )
        log.warning(
            "legacy invite redemption (pre-Phase 7 token shape) campaign_id=%s "
            "user_id=%s grace_until=%s",
            campaign_id,
            user.id,
            _LEGACY_GRACE_END,
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

    # Phase 7 token: look up the row and enforce single-use.
    invite = await db.get(models.CampaignInvite, str(invite_id))
    if invite is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invite code is invalid",
        )
    # Defense in depth: signed token + matching campaign_id field.
    if invite.campaign_id != campaign_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invite code is invalid",
        )
    if invite.revoked_at is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invite code has been revoked",
        )
    now = _now_iso()
    if invite.expires_at <= now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invite code has expired",
        )
    if invite.used_at is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invite code has already been used",
        )

    # Mark used + add membership atomically. ``used_by`` is set even if
    # the user is somehow already a member (e.g. owner added them
    # directly) so the audit row is consistent.
    invite.used_by = user.id
    invite.used_at = now
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


def _prompt_hash(prompt: str) -> str:
    """SHA-256 hex of the prompt — used for image dedup per spec §10."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@router.post(
    "/from-module",
    response_model=LoadModuleResponse,
    status_code=status.HTTP_201_CREATED,
)
async def load_module(
    payload: LoadModuleRequest,
    user: CurrentUser,
    db: DbSession,
) -> LoadModuleResponse:
    """Create a campaign from a module.

    Transaction (all-or-nothing):
    1. Load + validate the module.
    2. Idempotence check: if a campaign already has this module_id, 409.
    3. Insert Campaign (with module_id).
    4. Mint UUIDv7 per symbol → build symbolic_id_map.
    5. Insert Locations (respecting parent_symbol hierarchy).
    6. Insert NPCs (with location_id from map).
    7. Insert WorldFacts (embedded).
    8. Write module_state with all beats pending.
    9. Insert CampaignMember (owner).
    10. Commit.

    Post-commit:
    11. For each image_manifest entry: dedup by prompt_hash; enqueue if
        not already generated. Return immediately — image generation is
        async.

    Idempotence: loading a module into the same campaign twice returns 409
    (the module_id already being set is the guard).
    """
    module_row = await db.get(models.Module, payload.module_id)
    if module_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="module not found")

    try:
        content = ModuleContent.model_validate(module_row.content)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"module content failed validation: {exc}",
        ) from exc

    # Build the campaign.
    effective_image_style = payload.image_style_override or content.image_style or module_row.tone
    campaign = models.Campaign(
        name=payload.name,
        owner_id=user.id,
        module_id=module_row.id,
        image_style=effective_image_style,
        image_negative_prompt=content.image_negative_prompt,
    )
    db.add(campaign)
    await db.flush()

    # Mint UUIDv7 per symbol.
    symbolic_id_map: dict[str, str] = {}
    for loc in content.locations:
        symbolic_id_map[loc.symbol] = str(uuid7())
    for npc in content.npcs:
        symbolic_id_map[npc.symbol] = str(uuid7())
    for enc in content.encounters:
        symbolic_id_map[enc.symbol] = str(uuid7())
    for beat in content.plot_beats:
        symbolic_id_map[beat.symbol] = str(uuid7())
    for secret in content.secrets:
        symbolic_id_map[secret.symbol] = str(uuid7())
    for ending in content.endings:
        symbolic_id_map[ending.symbol] = str(uuid7())

    # Insert Locations. Two-pass: insert parents first, then children.
    # Locations without a parent_symbol are inserted first; those with
    # a parent_symbol get their parent_id from the map.
    loc_by_symbol: dict[str, models.Location] = {}
    parents = [loc for loc in content.locations if loc.parent_symbol is None]
    children = [loc for loc in content.locations if loc.parent_symbol is not None]

    for loc in parents + children:
        loc_id = symbolic_id_map[loc.symbol]
        parent_id: str | None = None
        if loc.parent_symbol is not None:
            parent_id = symbolic_id_map.get(loc.parent_symbol)

        location_row = models.Location(
            id=loc_id,
            campaign_id=campaign.id,
            name=loc.name,
            description=loc.description,
            parent_id=parent_id,
            location_metadata=loc.metadata,
        )
        db.add(location_row)
        loc_by_symbol[loc.symbol] = location_row

    await db.flush()

    # Insert NPCs.
    for npc in content.npcs:
        npc_id = symbolic_id_map[npc.symbol]
        starting_loc_id = symbolic_id_map.get(npc.starting_location_symbol)
        npc_row = models.Npc(
            id=npc_id,
            campaign_id=campaign.id,
            name=npc.name,
            description=npc.description,
            stats=npc.stats,
            location_id=starting_loc_id,
        )
        db.add(npc_row)

    await db.flush()

    # Insert WorldFacts with embeddings.
    embedder = get_embedder()
    facts_inserted = 0
    if content.world_facts:
        fact_texts = [wf.fact for wf in content.world_facts]
        try:
            vectors = await embedder.embed(fact_texts)
            for wf, vec in zip(content.world_facts, vectors, strict=True):
                fact_row = models.WorldFact(
                    campaign_id=campaign.id,
                    fact=wf.fact,
                    embedding=vec.astype(np.float32, copy=False).tobytes(),
                    embedding_dim=int(vec.shape[0]),
                    tags=wf.tags,
                    importance=wf.importance,
                )
                db.add(fact_row)
            facts_inserted = len(content.world_facts)
        except Exception:
            log.warning("load_module: embedding world_facts failed; skipping facts", exc_info=True)

    # Build module_state.
    module_state = {
        "module_id": module_row.id,
        "symbolic_id_map": symbolic_id_map,
        "beats_pending": [beat.symbol for beat in content.plot_beats],
        "beats_hit": [],
        "secrets_revealed": [],
        "encounters_run": [],
        "endings_reached": [],
    }
    campaign.module_state = module_state

    # Insert CampaignMember (owner).
    db.add(models.CampaignMember(campaign_id=campaign.id, user_id=user.id, role="owner"))

    await db.commit()
    await db.refresh(campaign)

    # Post-commit: image dedup / enqueue from image_manifest.
    image_manifest: list[dict] = module_row.image_manifest or []
    jobs_enqueued = 0
    if image_manifest:
        try:
            from app.images.portrait import get_queue_client
            from app.images.queue import ImageJob, push_job

            queue_client = get_queue_client()
            for entry in image_manifest:
                prompt = entry.get("prompt", "")
                ph = entry.get("prompt_hash") or _prompt_hash(prompt)
                if not prompt:
                    continue
                # Dedup: check if this hash is already generated.
                existing = (
                    await db.scalars(
                        select(models.GeneratedImage).where(
                            models.GeneratedImage.prompt_hash == ph
                        )
                    )
                ).first()
                if existing is not None:
                    continue
                # Enqueue.
                image_id = str(uuid7())
                job = ImageJob(
                    id=image_id,
                    campaign_id=campaign.id,
                    session_id=None,
                    kind=entry.get("kind", "npc"),
                    prompt=prompt,
                )
                await push_job(queue_client, job)
                jobs_enqueued += 1
        except Exception:
            log.warning("load_module: image enqueue failed; continuing", exc_info=True)

    return LoadModuleResponse(
        campaign_id=campaign.id,
        name=campaign.name,
        locations_created=len(content.locations),
        npcs_created=len(content.npcs),
        image_jobs_enqueued=jobs_enqueued,
    )


__all__ = [
    "CampaignDetail",
    "CampaignListEntry",
    "CampaignResponse",
    "CharacterCardSummary",
    "CreateCampaignRequest",
    "InviteListEntry",
    "InviteResponse",
    "JoinRequest",
    "JoinResponse",
    "LoadModuleRequest",
    "LoadModuleResponse",
    "MemberSummary",
    "SessionSummary",
    "router",
]
