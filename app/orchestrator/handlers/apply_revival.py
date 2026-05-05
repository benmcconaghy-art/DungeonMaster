"""Handler for the ``apply_revival`` tool.

Revives a downed character (hp_current <= 0) to 1 HP. This is the
*only* tool authorised to bypass the 0-HP rule — ordinary ``heal``
refuses 0-HP targets (BFRPG-correct; Phase 6.12). Revival is its own
ritual, distinct from healing: a cleric's prayer, a potion of life,
divine intervention.

Phase 3 comment in apply_damage.py: "finer status flags to come" —
this handler also clears any dying/stable-class status effects on
revival, since those implied states are resolved by the act of revival.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Character
from app.llm.tools import ApplyRevival, ToolResult, register

# Effects that unambiguously resolve on revival. Clearing these prevents
# the model from having to call clear_status_effect after apply_revival.
_DYING_EFFECTS: frozenset[str] = frozenset({"dying", "stable", "unconscious"})


@register("apply_revival")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: ApplyRevival) -> ToolResult:
    """Read HP from DB, verify target is downed, set to 1 HP, clear dying effects."""

    ch = await db.get(Character, args.character_id)
    if ch is None:
        return ToolResult(
            content=(
                f"apply_revival failed: no character with id {args.character_id!r} —"
                " refusing to mutate state. Verify the character_id."
            ),
            side_effects={"kind": "error", "reason": "unknown_target"},
        )

    if ch.hp_current > 0:
        return ToolResult(
            content=(
                f"apply_revival refused: {ch.name} is at {ch.hp_current} HP —"
                " apply_revival is only for downed characters (HP ≤ 0)."
                " Use heal for living characters."
            ),
            side_effects={"kind": "error", "reason": "target_not_downed"},
        )

    previous_hp = ch.hp_current
    previous_status = ch.status

    ch.hp_current = 1
    ch.status = "alive"

    # Clear implied dying states so the model doesn't have to follow up.
    current_effects: list[str] = list(ch.status_effects or [])
    cleared = [e for e in current_effects if e in _DYING_EFFECTS]
    ch.status_effects = [e for e in current_effects if e not in _DYING_EFFECTS]

    await db.flush()

    source_phrase = f" ({args.source})" if args.source else ""
    summary = f"{ch.name} revived{source_phrase}: HP {previous_hp} -> 1/{ch.hp_max}." + (
        f" Cleared effects: {', '.join(cleared)}." if cleared else ""
    )
    return ToolResult(
        content=summary,
        side_effects={
            "kind": "state_update",
            "target_id": ch.id,
            "target_name": ch.name,
            "field": "hp_current",
            "previous": previous_hp,
            "new": 1,
            "source": args.source,
            "previous_status": previous_status,
            "new_status": "alive",
            "cleared_effects": cleared,
        },
    )


__all__ = ["handle"]
