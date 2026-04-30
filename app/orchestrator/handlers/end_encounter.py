"""Handler for the ``end_encounter`` tool.

Sets ``encounters.status`` to the outcome the LLM reports. The schema
defines ``status`` as a free-form text column (no CHECK), so we
write the literal outcome value (``victory``, ``flee``, ``parley``,
``tpk``, ``other``) and let downstream consumers (analytics, summary
generator) read it.

The summary string is appended to a synthetic side-effect record;
Phase 3+ will use it when the session-summary regenerator runs.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Encounter
from app.llm.tools import EndEncounter, ToolResult, register
from app.orchestrator.context import current_context


@register("end_encounter")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: EndEncounter) -> ToolResult:
    """Mark the encounter as finished; record the outcome."""

    ctx = current_context()
    enc = await db.get(Encounter, args.encounter_id)
    if enc is None or enc.session_id != ctx.session_id:
        return ToolResult(
            content=(
                f"end_encounter failed: encounter {args.encounter_id!r} not"
                " found in this session. Verify the encounter_id."
            ),
            side_effects={"kind": "error", "reason": "unknown_encounter"},
        )

    if enc.status != "active":
        return ToolResult(
            content=(
                f"end_encounter no-op: encounter {enc.name} is already"
                f" {enc.status!r}. Refusing to overwrite."
            ),
            side_effects={"kind": "error", "reason": "not_active"},
        )

    enc.status = args.outcome
    await db.flush()

    summary = f"Encounter '{enc.name}' ended ({args.outcome})." + (
        f" {args.summary}" if args.summary else ""
    )
    return ToolResult(
        content=summary,
        side_effects={
            "kind": "encounter_ended",
            "encounter_id": enc.id,
            "outcome": args.outcome,
            "summary": args.summary,
        },
    )


__all__ = ["handle"]
