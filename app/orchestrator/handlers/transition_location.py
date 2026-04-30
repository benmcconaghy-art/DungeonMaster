"""Handler for the ``transition_location`` tool.

Moves the party between locations within a campaign. The handler
validates that the location belongs to the active campaign before
mutating ``sessions.current_location_id`` — the LLM might invent IDs,
and we'd rather refuse cleanly than corrupt session state.

Side effect: a ``session_messages`` row with ``sender_kind='system'``
is appended to the session log so the location change is part of the
verbatim history.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Location, SessionMessage
from app.db.models import Session as DmSession
from app.llm.tools import ToolResult, TransitionLocation, register
from app.orchestrator.context import current_context


@register("transition_location")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: TransitionLocation) -> ToolResult:
    """Validate the location, update the session, log a system message."""

    ctx = current_context()
    session = await db.get(DmSession, ctx.session_id)
    if session is None:
        # Should never happen — orchestrator validates session before
        # dispatch — but defensive.
        return ToolResult(
            content="transition_location failed: active session not found.",
            side_effects={"kind": "error", "reason": "unknown_session"},
        )

    location = await db.get(Location, args.location_id)
    if location is None or location.campaign_id != session.campaign_id:
        return ToolResult(
            content=(
                f"transition_location failed: location_id {args.location_id!r}"
                " does not exist in this campaign. Check the location list and"
                " try again, or describe the move without the tool call."
            ),
            side_effects={"kind": "error", "reason": "unknown_location"},
        )

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
        },
    )


__all__ = ["handle"]
