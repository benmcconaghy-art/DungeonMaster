"""Handler for the ``mark_beat`` tool.

Records that an adventure-module plot beat has fired. The DM calls this when
the narrative moment described by the beat's trigger_hint has occurred.

Beat tracking is LLM-judged: the trigger_hint is natural-language guidance,
not a mechanical condition. The engine validates the beat_id is known (present
in beats_pending or beats_hit) and moves it from pending → hit if not already
done. An already-hit beat is a no-op with a structured note so the DM doesn't
get penalised for calling it twice.

Campaign must have module_state populated (i.e. be loaded from a module).
Non-module campaigns return an informative error.
"""

from __future__ import annotations

from app.db.models import Campaign
from app.db.models import Session as DmSession
from app.llm.tools import MarkBeat, ToolResult, register
from app.orchestrator.context import current_context
from sqlalchemy.ext.asyncio import AsyncSession


@register("mark_beat")
async def handle(db: AsyncSession, args: MarkBeat) -> ToolResult:
    ctx = current_context()
    session = await db.get(DmSession, ctx.session_id)
    if session is None:
        return ToolResult(
            content="mark_beat failed: active session not found.",
            side_effects={"kind": "error", "reason": "unknown_session"},
        )

    campaign = await db.get(Campaign, session.campaign_id)
    if campaign is None:
        return ToolResult(
            content="mark_beat failed: campaign not found.",
            side_effects={"kind": "error", "reason": "unknown_campaign"},
        )

    module_state: dict = campaign.module_state or {}
    if not module_state or not module_state.get("module_id"):
        return ToolResult(
            content=(
                "mark_beat failed: this campaign is not loaded from a module. "
                "Beat tracking requires a module-backed campaign."
            ),
            side_effects={"kind": "error", "reason": "no_module"},
        )

    beats_pending: list[str] = list(module_state.get("beats_pending", []))
    beats_hit: list[str] = list(module_state.get("beats_hit", []))

    beat_id = args.beat_id

    # Validate the beat exists in this module at all.
    known_beats = set(beats_pending) | set(beats_hit)
    if beat_id not in known_beats:
        return ToolResult(
            content=(
                f"mark_beat failed: beat_id {beat_id!r} is not a known beat in this module. "
                f"Known pending beats: {', '.join(beats_pending) or '(none)'}."
            ),
            side_effects={"kind": "error", "reason": "unknown_beat", "beat_id": beat_id},
        )

    # Idempotent: already hit → structured no-op.
    if beat_id in beats_hit:
        return ToolResult(
            content=(
                f"Beat {beat_id!r} was already marked as hit. No state change."
            ),
            side_effects={
                "kind": "beat_already_hit",
                "beat_id": beat_id,
            },
        )

    # Move pending → hit.
    beats_pending.remove(beat_id)
    beats_hit.append(beat_id)

    updated_state = {
        **module_state,
        "beats_pending": beats_pending,
        "beats_hit": beats_hit,
    }
    campaign.module_state = updated_state
    await db.flush()

    summary = args.summary or f"Beat {beat_id!r} marked as hit."
    return ToolResult(
        content=summary,
        side_effects={
            "kind": "beat_marked",
            "beat_id": beat_id,
            "summary": summary,
            "beats_remaining": len(beats_pending),
        },
    )


__all__ = ["handle"]
