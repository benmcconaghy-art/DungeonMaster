"""Handler for the ``clear_status_effect`` tool.

Removes a status effect from a character's status_effects list. No-op
(with a note in the result) if the effect is not currently applied —
the model should not be penalised for clearing an effect that already
lapsed or was never applied.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Character
from app.llm.tools import ClearStatusEffect, ToolResult, register


@register("clear_status_effect")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: ClearStatusEffect) -> ToolResult:
    """Read current status_effects from DB, remove the named effect, persist."""

    ch = await db.get(Character, args.character_id)
    if ch is None:
        return ToolResult(
            content=(
                f"clear_status_effect failed: no character with id {args.character_id!r} —"
                " refusing to mutate state. Verify the character_id."
            ),
            side_effects={"kind": "error", "reason": "unknown_target"},
        )

    current_effects: list[str] = list(ch.status_effects or [])
    was_present = args.effect in current_effects

    if was_present:
        current_effects.remove(args.effect)
        ch.status_effects = current_effects
        await db.flush()

    remaining = ", ".join(current_effects) if current_effects else "none"
    if was_present:
        summary = f"{ch.name} is no longer {args.effect}. Remaining effects: {remaining}."
    else:
        summary = (
            f"{ch.name} did not have '{args.effect}' applied — no change."
            f" Current effects: {remaining}."
        )

    return ToolResult(
        content=summary,
        side_effects={
            "kind": "state_update",
            "target_id": ch.id,
            "target_name": ch.name,
            "field": "status_effects",
            "effect": args.effect,
            "was_present": was_present,
            "effects_after": current_effects,
        },
    )


__all__ = ["handle"]
