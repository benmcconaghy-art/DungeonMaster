"""Handler for the ``start_encounter`` tool.

Creates an active ``encounters`` row for the session, rolls
initiative for every monster the LLM declared (the DM never rolls
initiative in prose — engine territory, AGENTS.md invariant #1), and
returns the resolved order so the LLM can narrate the opening
moments of combat.

Phase 2 only includes monsters in initiative. Phase 5+ reads the
party's character list and merges PCs into the order; for now the
orchestrator's caller is single-player and the LLM can request a
PC-side dice roll separately.
"""

from __future__ import annotations

import random

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Encounter
from app.game.rules import Participant, roll_initiative
from app.llm.tools import StartEncounter, ToolResult, register
from app.orchestrator.context import current_context


@register("start_encounter")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: StartEncounter) -> ToolResult:
    """Create an encounter, roll initiative, persist."""

    ctx = current_context()
    rng = random.Random()

    # Build participants for initiative. Each declared monster contributes
    # ``count`` participants, each with a unique participant_id so ties
    # break deterministically. Phase 2 doesn't model monster Dex
    # individually, so dex_modifier defaults to 0.
    participants: list[Participant] = []
    monsters_payload: list[dict[str, object]] = []
    for monster in args.monsters:
        monsters_payload.append(
            {
                "name": monster.name,
                "count": monster.count,
                "hp": monster.hp,
                "notes": monster.notes,
            }
        )
        for idx in range(monster.count):
            pid = f"{monster.name}#{idx + 1}"
            participants.append(
                Participant(
                    participant_id=pid,
                    name=monster.name,
                    dex_modifier=0,
                    is_player=False,
                )
            )

    order = roll_initiative(participants, rng=rng)
    initiative_payload = [
        {
            "participant_id": e.participant_id,
            "name": e.name,
            "initiative": e.initiative,
            "is_player": e.is_player,
        }
        for e in order.entries
    ]

    enc = Encounter(
        session_id=ctx.session_id,
        name=args.name,
        status="active",
        monsters=monsters_payload,
        initiative=initiative_payload,
        round_number=order.round_number,
        current_turn=order.index,
    )
    db.add(enc)
    await db.flush()

    order_str = ", ".join(
        f"{entry['name']} ({entry['initiative']})" for entry in initiative_payload
    )
    summary = (
        f"Encounter '{args.name}' started (id={enc.id}). Round 1."
        f" Initiative order: {order_str}."
    )
    return ToolResult(
        content=summary,
        side_effects={
            "kind": "encounter_started",
            "encounter_id": enc.id,
            "name": args.name,
            "round_number": order.round_number,
            "initiative": initiative_payload,
            "monsters": monsters_payload,
        },
    )


__all__ = ["handle"]
