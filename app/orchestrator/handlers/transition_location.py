"""Handler for the ``transition_location`` tool.

Moves the party between locations within a campaign. The DM may pass
either ``location_id`` (when it already knows the canonical id of an
established place) or ``name`` (the place name as it appears in
narration). With ``name``, the handler resolves the existing location
by name match against this campaign and creates a new ``Location``
row with the supplied ``description`` if no match is found. Either
path validates the location belongs to the active campaign before
mutating ``sessions.current_location_id``.

Side effect: a ``session_messages`` row with ``sender_kind='system'``
is appended to the session log so the location change is part of the
verbatim history.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Location, SessionMessage
from app.db.models import Session as DmSession
from app.llm.tools import ToolResult, TransitionLocation, register
from app.orchestrator.context import current_context

# Fuzzy-match cutoff for the name → existing-location lookup.
# 0.78 is conservative: "jeb's smithy" matches "Jeb's Smithy" trivially
# (1.0 after normalisation), and a clear typo like "jebs smithy" still
# clears the bar (~0.86). A genuinely different place ("the keep" vs
# "the cellar") falls below it and we create a new row instead of
# silently teleporting the party somewhere they didn't ask for.
_NAME_MATCH_CUTOFF = 0.78


def _normalise(name: str) -> str:
    return name.strip().casefold()


def _match_existing(name: str, candidates: Iterable[Location]) -> Location | None:
    """Resolve ``name`` against ``candidates``: exact (case-insensitive)
    first, then ``difflib.get_close_matches`` against the normalised
    names. Returns the matched ``Location`` or ``None``."""

    target = _normalise(name)
    by_norm: dict[str, Location] = {}
    for loc in candidates:
        by_norm[_normalise(loc.name)] = loc
    direct = by_norm.get(target)
    if direct is not None:
        return direct
    close = difflib.get_close_matches(target, list(by_norm.keys()), n=1, cutoff=_NAME_MATCH_CUTOFF)
    if not close:
        return None
    return by_norm[close[0]]


@register("transition_location")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: TransitionLocation) -> ToolResult:
    """Validate or create the location, update the session, log a
    system message. Accepts ``location_id`` (canonical id) or ``name``
    (resolved by match-or-create within the campaign)."""

    ctx = current_context()
    session = await db.get(DmSession, ctx.session_id)
    if session is None:
        # Should never happen — orchestrator validates session before
        # dispatch — but defensive.
        return ToolResult(
            content="transition_location failed: active session not found.",
            side_effects={"kind": "error", "reason": "unknown_session"},
        )

    resolution: str  # 'id' | 'name_match' | 'name_create'
    location: Location | None

    if args.location_id is not None:
        location = await db.get(Location, args.location_id)
        if location is None or location.campaign_id != session.campaign_id:
            return ToolResult(
                content=(
                    f"transition_location failed: location_id {args.location_id!r}"
                    " does not exist in this campaign. Reference the place by"
                    " name instead and the engine will resolve or create it."
                ),
                side_effects={"kind": "error", "reason": "unknown_location"},
            )
        resolution = "id"
    else:
        # Name path: match-or-create within this campaign.
        assert args.name is not None  # enforced by Pydantic post-init
        candidates_stmt = select(Location).where(Location.campaign_id == session.campaign_id)
        candidates = list((await db.scalars(candidates_stmt)).all())
        match = _match_existing(args.name, candidates)
        if match is not None:
            location = match
            resolution = "name_match"
        else:
            location = Location(
                campaign_id=session.campaign_id,
                name=args.name.strip(),
                description=args.description or None,
            )
            db.add(location)
            await db.flush()
            resolution = "name_create"

    previous_location_id = session.current_location_id
    session.current_location_id = location.id

    log = SessionMessage(
        session_id=session.id,
        sender_kind="system",
        sender_id=None,
        audience=[],
        content=f"Location: {location.name}. {args.description}",
    )
    db.add(log)
    await db.flush()

    summary = f"Party moved to {location.name}. {args.description}"
    return ToolResult(
        content=summary,
        side_effects={
            "kind": "state_update",
            "field": "current_location_id",
            "previous": previous_location_id,
            "new": location.id,
            "location_name": location.name,
            "description": args.description,
            "resolution": resolution,
        },
    )


__all__ = ["handle"]
