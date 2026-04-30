"""Handler for the ``apply_damage`` tool.

Reduces a character's HP by the engine-supplied amount. The handler
always reads the current HP from the database — never trust
LLM-supplied state (``test-writer.md``: "LLM tried to lie" test). If
HP drops to zero or below, the engine's Death and Dismemberment table
is rolled and the result applied.

Phase 2 only handles ``Character`` targets. Phase 5+ extends this to
NPC and monster targets when the spawn flow lands.
"""

from __future__ import annotations

import random

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Character
from app.game.death import death_and_dismemberment
from app.game.rules import CharacterStats
from app.game.rules import apply_damage as engine_apply_damage
from app.llm.tools import ApplyDamage, ToolResult, register


def _stats_from_row(ch: Character) -> CharacterStats:
    """Lift a Character ORM row into the engine's pure-data view.

    Only the fields the death-and-dismemberment caller needs are
    populated; fields irrelevant to HP arithmetic (saves, attack
    bonus, etc.) are left at their dataclass defaults.
    """

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


@register("apply_damage")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: ApplyDamage) -> ToolResult:
    """Read HP from DB, subtract, persist; roll on Death table at <= 0."""

    ch = await db.get(Character, args.target_id)
    if ch is None:
        return ToolResult(
            content=(
                f"apply_damage failed: no character with id {args.target_id!r} —"
                " refusing to mutate state. Verify the target_id."
            ),
            side_effects={"kind": "error", "reason": "unknown_target"},
        )

    stats = _stats_from_row(ch)
    damage = engine_apply_damage(stats, args.amount, source=args.source)

    if not damage.dropped_to_zero:
        ch.hp_current = damage.new_hp
        await db.flush()
        summary = (
            f"{ch.name} took {args.amount} damage from {args.source}: "
            f"HP {damage.previous_hp} -> {damage.new_hp}/{ch.hp_max}."
        )
        return ToolResult(
            content=summary,
            side_effects={
                "kind": "state_update",
                "target_id": ch.id,
                "field": "hp_current",
                "previous": damage.previous_hp,
                "new": damage.new_hp,
                "amount": args.amount,
                "source": args.source,
                "dropped_to_zero": False,
            },
        )

    # HP at or below zero — roll on the Death and Dismemberment table.
    rng = random.Random()
    death = death_and_dismemberment(stats, damage, rng=rng)

    if death.outcome == "dead":
        ch.hp_current = 0
        ch.status = "dead"
        await db.flush()
        summary = (
            f"{ch.name} took {args.amount} damage from {args.source} and died."
            f" (Death table {death.total} -> dead.)"
        )
        return ToolResult(
            content=summary,
            side_effects={
                "kind": "state_update",
                "target_id": ch.id,
                "field": "hp_current",
                "previous": damage.previous_hp,
                "new": 0,
                "amount": args.amount,
                "source": args.source,
                "dropped_to_zero": True,
                "death_outcome": death.outcome,
                "death_total": death.total,
                "death_detail": dict(death.detail),
            },
        )

    # Survivable outcome.
    new_hp = death.hp_after if death.hp_after is not None else 0
    ch.hp_current = new_hp
    # Phase 2 keeps status='alive' for survivable outcomes; phase 3 adds
    # finer status flags ('downed', 'crippled', etc.).
    await db.flush()
    summary = (
        f"{ch.name} took {args.amount} damage from {args.source} and dropped"
        f" to {damage.new_hp} HP. Death table rolled {death.total}: "
        f"{death.outcome.replace('_', ' ')}. HP now {new_hp}/{ch.hp_max}."
    )
    return ToolResult(
        content=summary,
        side_effects={
            "kind": "state_update",
            "target_id": ch.id,
            "field": "hp_current",
            "previous": damage.previous_hp,
            "new": new_hp,
            "amount": args.amount,
            "source": args.source,
            "dropped_to_zero": True,
            "death_outcome": death.outcome,
            "death_total": death.total,
            "death_detail": dict(death.detail),
        },
    )


__all__ = ["handle"]
