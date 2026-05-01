"""Dashboard context builder.

The /dashboard route renders ``campaign_dashboard.html`` with a
composed payload: list of campaigns the user belongs to, full detail
on the most-recent (featured) one, and a recent-sessions index. The
template doesn't fetch — it consumes.

Composing here (rather than in the route) keeps the FastAPI handler
short and lets the same composition feed any future surface (an
HTMX partial refresh, a JSON dashboard endpoint, a test fixture).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, select

from app.api.campaigns import (
    CampaignListEntry,
    _campaign_last_played,
    _list_member_summaries,
)
from app.db import models
from app.deps import DbSession


async def build_context(db: DbSession, *, user: models.User) -> dict[str, Any]:
    """Compose the dashboard render context for ``user``.

    Returns the full dict the Jinja template expects:

    * ``user`` — base.html chrome reads it.
    * ``campaigns`` — :class:`CampaignListEntry` rows, ordered by
      most-recent-first.
    * ``featured`` — featured campaign card payload, or ``None`` if
      the user has no campaigns.
    * ``other_campaigns`` — list of secondary campaign-row payloads.
    * ``recent_sessions`` — sidebar entries, newest first, across all
      member campaigns.
    * ``total_my_characters`` — small greeting-strip count.
    * ``last_played_relative`` — pre-rendered relative timestamp for
      the greeting strip.
    """

    listing = await _list_campaigns_for(db, user=user)
    featured: dict[str, Any] | None = None
    other_campaigns: list[dict[str, Any]] = []

    for entry in listing:
        relative = _relative_time(entry.last_played_at)
        if featured is None and entry.most_recent:
            featured = await _featured_payload(db, user=user, entry=entry)
        else:
            other_campaigns.append(
                {
                    "id": entry.id,
                    "name": entry.name,
                    "ruleset": entry.ruleset,
                    "member_count": entry.member_count,
                    "last_played_relative": relative,
                    "active_session_id": await _active_session_id(db, campaign_id=entry.id),
                }
            )

    recent_sessions = await _recent_sessions(db, user=user)
    total_my_chars = sum(e.my_character_count for e in listing)
    last_played_relative = _relative_time(listing[0].last_played_at if listing else None)

    return {
        "user": user,
        "campaigns": listing,
        "featured": featured,
        "other_campaigns": other_campaigns,
        "recent_sessions": recent_sessions,
        "total_my_characters": total_my_chars,
        "last_played_relative": last_played_relative,
    }


async def _list_campaigns_for(db: DbSession, *, user: models.User) -> list[CampaignListEntry]:
    """Reuse the API endpoint's composition. Keeps dashboard and
    JSON list endpoint perfectly in sync."""

    # Inline the list_campaigns body: a thin re-implementation here
    # avoids the FastAPI Depends overhead and lets us share the
    # CampaignListEntry shape directly.
    member_ids = (
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
    if not member_ids:
        return []

    campaign_ids = list(member_ids)

    campaigns = list(
        (
            await db.execute(select(models.Campaign).where(models.Campaign.id.in_(campaign_ids)))
        ).scalars()
    )

    member_count_rows = (
        await db.execute(
            select(
                models.CampaignMember.campaign_id,
                models.CampaignMember.user_id,
            ).where(models.CampaignMember.campaign_id.in_(campaign_ids))
        )
    ).all()
    member_count_by_campaign: dict[str, int] = {}
    for cid, _uid in member_count_rows:
        member_count_by_campaign[cid] = member_count_by_campaign.get(cid, 0) + 1

    my_char_rows = (
        (
            await db.execute(
                select(models.Character.campaign_id)
                .where(models.Character.user_id == user.id)
                .where(models.Character.campaign_id.in_(campaign_ids))
            )
        )
        .scalars()
        .all()
    )
    my_char_by_campaign: dict[str, int] = {}
    for cid in my_char_rows:
        my_char_by_campaign[cid] = my_char_by_campaign.get(cid, 0) + 1

    rows: list[tuple[CampaignListEntry, str | None]] = []
    for c in campaigns:
        last_played_at, has_active = await _campaign_last_played(db, campaign_id=c.id)
        rows.append(
            (
                CampaignListEntry(
                    id=c.id,
                    name=c.name,
                    ruleset=c.ruleset,
                    owner_id=c.owner_id,
                    member_count=member_count_by_campaign.get(c.id, 0),
                    my_character_count=my_char_by_campaign.get(c.id, 0),
                    last_played_at=last_played_at,
                    has_active_session=has_active,
                    most_recent=False,
                ),
                last_played_at,
            )
        )

    dated = sorted([r for r in rows if r[1] is not None], key=lambda p: p[1] or "", reverse=True)
    undated = [r for r in rows if r[1] is None]
    ordered = [r[0] for r in dated] + [r[0] for r in undated]
    if dated:
        ordered[0] = ordered[0].model_copy(update={"most_recent": True})
    return ordered


async def _featured_payload(
    db: DbSession, *, user: models.User, entry: CampaignListEntry
) -> dict[str, Any]:
    """Compose the featured-card render context for the most-recent
    campaign. Includes the user's characters in this campaign, the
    member roster, and the active-session pointer."""

    members = await _list_member_summaries(db, campaign_id=entry.id)
    char_rows = list(
        (
            await db.execute(
                select(models.Character)
                .where(models.Character.campaign_id == entry.id)
                .where(models.Character.user_id == user.id)
                .order_by(models.Character.name)
            )
        ).scalars()
    )
    active_session_id = await _active_session_id(db, campaign_id=entry.id)
    last_session = (
        await db.execute(
            select(models.Session)
            .where(models.Session.campaign_id == entry.id)
            .order_by(desc(models.Session.started_at))
            .limit(1)
        )
    ).scalar_one_or_none()

    resume_caption: str | None = None
    if active_session_id and last_session and last_session.summary:
        # First sentence of the rolling summary as the world-voice
        # whereabouts caption — reads as a one-line scene anchor.
        first_sentence = last_session.summary.split(". ")[0].strip()
        # Trim and lowercase the first letter so it joins the verb
        # cleanly ("Resume Session — the chapel below the hollow…").
        if first_sentence:
            resume_caption = first_sentence
            if resume_caption.endswith("."):
                resume_caption = resume_caption[:-1]

    seal_glyph = _seal_glyph(entry.name)

    return {
        "id": entry.id,
        "name": entry.name,
        "ruleset": entry.ruleset,
        "member_count": entry.member_count,
        "members": members,
        "my_characters": char_rows,
        "has_active_session": entry.has_active_session,
        "active_session_id": active_session_id,
        "last_played_at": entry.last_played_at,
        "last_played_relative": _relative_time(entry.last_played_at),
        "resume_caption": resume_caption,
        "seal_glyph": seal_glyph,
    }


async def _active_session_id(db: DbSession, *, campaign_id: str) -> str | None:
    """Return the id of the campaign's currently-active session if any."""

    row = (
        await db.execute(
            select(models.Session.id)
            .where(models.Session.campaign_id == campaign_id)
            .where(models.Session.ended_at.is_(None))
            .order_by(desc(models.Session.started_at))
            .limit(1)
        )
    ).scalar_one_or_none()
    return row


async def _recent_sessions(
    db: DbSession, *, user: models.User, limit: int = 6
) -> list[dict[str, Any]]:
    """Cross-campaign sidebar list — newest sessions across every
    campaign the user belongs to, with a pre-resolved campaign name."""

    member_ids = (
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
    if not member_ids:
        return []
    rows = list(
        (
            await db.execute(
                select(models.Session, models.Campaign)
                .join(models.Campaign, models.Campaign.id == models.Session.campaign_id)
                .where(models.Session.campaign_id.in_(list(member_ids)))
                .order_by(desc(models.Session.started_at))
                .limit(limit)
            )
        ).all()
    )
    items: list[dict[str, Any]] = []
    for session, campaign in rows:
        items.append(
            {
                "campaign_name": campaign.name,
                "summary": session.summary,
                "relative": _relative_time(session.ended_at or session.started_at),
                "active_session_id": session.id if session.ended_at is None else None,
            }
        )
    return items


def _relative_time(iso_ts: str | None) -> str | None:
    """Render an ISO-8601 timestamp as ``"3d ago"`` / ``"2h ago"`` /
    ``"just now"``. Returns ``None`` if the input is ``None`` so the
    template can guard with ``{% if relative %}``."""

    if iso_ts is None:
        return None
    try:
        # SQLite's strftime('%Y-%m-%dT%H:%M:%fZ','now') format is
        # parseable by fromisoformat once we strip the trailing Z.
        ts = datetime.fromisoformat(iso_ts.rstrip("Z")).replace(tzinfo=UTC)
    except ValueError:
        return None
    now = datetime.now(UTC)
    delta = now - ts
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    years = days // 365
    return f"{years}y ago"


def _seal_glyph(campaign_name: str) -> str:
    """Two-letter glyph for the wax-seal disc on the resume button.
    Initials of the first two whitespace-separated words; falls back
    to the first two letters of the name."""

    parts = [p for p in campaign_name.strip().split() if p]
    if len(parts) >= 2:
        return (parts[0][:1] + parts[1][:1]).upper()
    if parts:
        return parts[0][:2].upper()
    return "DM"


__all__ = ["build_context"]
