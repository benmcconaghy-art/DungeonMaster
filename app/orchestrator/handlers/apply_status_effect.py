"""Handler for the ``apply_status_effect`` tool.

Applies a transient status effect (free-form string) to a character.
Idempotent — applying an effect that is already present produces a
success result rather than duplicating the entry.

Common BFRPG effects: "poisoned", "paralyzed", "charmed", "blessed",
"dying", "stable", "unconscious". Module-specific effects are also
valid; the field is deliberately free-form.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Character
from app.llm.tools import ApplyStatusEffect, ToolResult, register


@register("apply_status_effect")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: ApplyStatusEffect) -> ToolResult:
    """Read current status_effects from DB, append if absent, persist."""

    ch = await db.get(Character, args.character_id)
    if ch is None:
        return ToolResult(
            content=(
                f"apply_status_effect failed: no character with id {args.character_id!r} —"
                " refusing to mutate state. Verify the character_id."
            ),
            side_effects={"kind": "error", "reason": "unknown_target"},
        )

    current_effects: list[str] = list(ch.status_effects or [])
    already_present = args.effect in current_effects

    if not already_present:
        current_effects.append(args.effect)
        ch.status_effects = current_effects
        await db.flush()

    hint_phrase = f" (duration: {args.duration_hint})" if args.duration_hint else ""
    effects_str = ", ".join(current_effects)
    if already_present:
        summary = f"{ch.name} is already {args.effect} — no change."
    else:
        summary = f"{ch.name} is now {args.effect}{hint_phrase}. Effects: {effects_str}."

    return ToolResult(
        content=summary,
        side_effects={
            "kind": "state_update",
            "target_id": ch.id,
            "target_name": ch.name,
            "field": "status_effects",
            "effect": args.effect,
            "duration_hint": args.duration_hint,
            "already_present": already_present,
            "effects_after": list(ch.status_effects or []),
        },
    )


__all__ = ["handle"]
