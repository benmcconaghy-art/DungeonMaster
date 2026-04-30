"""Handler for the ``heal`` tool.

Adds HP to a character, capped at the character's ``hp_max``. The
handler always reads ``hp_current`` and ``hp_max`` from the database
— never trust LLM-supplied state (the "LLM tried to lie" pattern from
``test-writer.md``).

Phase 2 only handles ``Character`` targets and refuses to heal a
0-HP character (revival is its own ritual, per the engine's
:func:`app.game.rules.heal`).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Character
from app.game.rules import CharacterStats
from app.game.rules import heal as engine_heal
from app.llm.tools import Heal, ToolResult, register


def _stats_from_row(ch: Character) -> CharacterStats:
    """Lift a Character row into the engine view (HP fields only matter)."""

    return CharacterStats(
        name=ch.name,
        class_name=ch.class_name,
        level=ch.level,
        hp_current=ch.hp_current,
        hp_max=ch.hp_max,
        ac=ch.ac,
        str_score=ch.str_score,
        int_score=ch.int_score,
        wis_score=ch.wis_score,
        dex_score=ch.dex_score,
        con_score=ch.con_score,
        cha_score=ch.cha_score,
    )


@register("heal")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: Heal) -> ToolResult:
    """Read HP from DB, add, cap at hp_max, persist."""

    ch = await db.get(Character, args.target_id)
    if ch is None:
        return ToolResult(
            content=(
                f"heal failed: no character with id {args.target_id!r} —"
                " refusing to mutate state. Verify the target_id."
            ),
            side_effects={"kind": "error", "reason": "unknown_target"},
        )

    if ch.hp_current <= 0:
        return ToolResult(
            content=(
                f"heal failed: {ch.name} is at {ch.hp_current} HP. Ordinary heal"
                " cannot revive a downed character — describe a revival ritual"
                " or reach for a different mechanic."
            ),
            side_effects={"kind": "error", "reason": "target_downed"},
        )

    stats = _stats_from_row(ch)
    result = engine_heal(stats, args.amount)
    ch.hp_current = result.new_hp
    await db.flush()

    delta = result.new_hp - result.previous_hp
    source_phrase = f" ({args.source})" if args.source else ""
    summary = (
        f"{ch.name} healed {delta}{source_phrase}: HP {result.previous_hp}"
        f" -> {result.new_hp}/{ch.hp_max}."
    )
    return ToolResult(
        content=summary,
        side_effects={
            "kind": "state_update",
            "target_id": ch.id,
            "field": "hp_current",
            "previous": result.previous_hp,
            "new": result.new_hp,
            "amount": delta,
            "source": args.source,
        },
    )


__all__ = ["handle"]
