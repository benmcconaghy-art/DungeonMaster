"""Handler for the ``start_encounter`` tool.

Creates an active ``encounters`` row for the session, rolls initiative
for every monster the LLM declared AND every alive PC in the campaign
(the DM never rolls initiative in prose — engine territory,
AGENTS.md invariant #1), and returns the resolved order so the LLM
can narrate the opening moments of combat.

Phase 4 change: PCs are auto-merged into initiative. Their
``participant_id`` is the character row id so the WS hub's initiative
gate can match a player's ``character_id`` against ``current_turn``.
The PC's Dex modifier is the BFRPG curve applied to ``dex_score``.
"""

from __future__ import annotations

import random

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Character, Encounter
from app.db.models import Session as DmSession
from app.game.rules import Participant, ability_modifier, roll_initiative
from app.llm.tools import StartEncounter, ToolResult, register
from app.orchestrator.context import current_context


@register("start_encounter")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: StartEncounter) -> ToolResult:
    """Create an encounter, roll initiative for monsters + alive PCs, persist."""

    ctx = current_context()
    rng = random.Random()

    # Build the participants list. Monsters first (so a stable ordering
    # falls out for ties at the same initiative value); PCs second.
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

    # Resolve the campaign for the encounter via session, then enrol
    # every alive PC. Their ``participant_id`` matches the character row
    # id so the WS gate's lookup works directly.
    session_row = await db.get(DmSession, ctx.session_id)
    if session_row is None:
        raise ValueError(f"unknown session_id {ctx.session_id!r} during start_encounter")
    campaign_id = session_row.campaign_id

    pc_stmt = (
        select(Character)
        .where(Character.campaign_id == campaign_id)
        .where(Character.status == "alive")
        .order_by(Character.name)
    )
    pcs = list((await db.scalars(pc_stmt)).all())
    for pc in pcs:
        participants.append(
            Participant(
                participant_id=pc.id,
                name=pc.name,
                dex_modifier=ability_modifier(pc.dex_score),
                is_player=True,
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
