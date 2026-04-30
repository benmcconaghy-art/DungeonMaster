"""Handler for the ``request_dice_roll`` tool.

The LLM passes a dice expression and a purpose; we evaluate via the
engine, persist a ``dice_rolls`` audit row, and return a
human-readable summary the LLM consumes on the next prompt to narrate
the outcome.

Critical rule: the engine adjudicates, the LLM narrates (AGENTS.md
invariant #1). This handler emits *mechanical* prose only — never
flavour. The DM injects flavour on its next turn.
"""

from __future__ import annotations

import random

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DiceRoll
from app.game.dice import roll
from app.llm.tools import DiceTarget, RequestDiceRoll, ToolResult, register
from app.orchestrator.context import current_context


def _format_target_phrase(target: DiceTarget | None) -> str:
    """Return the ``vs <KIND> <value>`` half of the summary line.

    Empty string if no resolvable target.
    """

    if target is None or target.kind == "none" or target.value is None:
        return ""
    return f" vs {target.kind.upper()} {target.value}"


@register("request_dice_roll")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: RequestDiceRoll) -> ToolResult:
    """Roll the dice, persist the audit row, return the summary.

    Phase 2 expects this handler to be called inside the orchestrator's
    own ``async with db.begin():`` block — it issues an ``add(...)``
    and counts on the surrounding context manager to commit.
    """

    ctx = current_context()
    rng = random.Random()
    result = roll(args.expression, rng=rng)

    if args.actor == "dm":
        actor_kind = "dm"
        actor_id: str | None = None
    else:
        actor_kind = "character"
        actor_id = args.actor

    audit = DiceRoll(
        session_id=ctx.session_id,
        actor_kind=actor_kind,
        actor_id=actor_id,
        expression=args.expression,
        individual=list(result.individual),
        total=result.total,
        purpose=args.purpose,
    )
    db.add(audit)
    await db.flush()

    target_phrase = _format_target_phrase(args.target)
    # Compute success/failure where the target is resolvable.
    success_phrase = ""
    if (
        args.target is not None
        and args.target.kind in ("ac", "dc")
        and args.target.value is not None
    ):
        success_phrase = " -> success" if result.total >= args.target.value else " -> failure"

    rolled_str = ",".join(str(v) for v in result.individual) or str(result.total)
    summary = (
        f"Rolled {args.expression} for {args.purpose!r}: total {result.total}"
        f" (dice: {rolled_str}){target_phrase}{success_phrase}"
    )

    return ToolResult(
        content=summary,
        side_effects={
            "kind": "dice_roll",
            "dice_roll_id": audit.id,
            "expression": args.expression,
            "total": result.total,
            "individual": list(result.individual),
            "purpose": args.purpose,
            "target": (
                {"kind": args.target.kind, "value": args.target.value}
                if args.target is not None
                else None
            ),
            "natural_one": result.natural_one,
            "natural_twenty": result.natural_twenty,
        },
    )


__all__ = ["handle"]
