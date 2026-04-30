"""Death and Dismemberment table (OSR-style, BFRPG-compatible).

The house rule (spec §4) is on by default: a character whose HP drops
to 0 or below rolls on this table rather than dying outright. Survivable
scars and lasting injuries replace insta-death and add narrative
texture to fights.

Mechanic
========

Roll ``2d6 + below_zero_by + crit_penalty`` and look the result up in
the table::

    ≤  3      Knocked out — 0 HP, unconscious 1d6 turns, then stable
    4-6       Lingering injury — 0 HP, unconscious 1d10 minutes; full
              recovery in 1d3 days
    7-9       Lasting scar — 0 HP for the encounter, leaves a
              cosmetic mark; no mechanical penalty
    10-12     Debilitating wound — permanent -1 to a random ability
              score; HP set to 1 after a long rest
    13-15     Crippled limb / sense — permanent: the character loses
              use of one limb or eye; a roll picks which
    16+       Dead — instant kill, no save

Inputs:

- ``below_zero_by`` is taken from the killing blow's
  :class:`~app.game.rules.DamageResult.below_zero_by`. It scales the
  severity directly: a small overkill is survivable, a massive overkill
  usually isn't.
- ``critical`` (default ``False``) — set when the killing blow was a
  natural 20 or other declared critical. Adds ``+2``.

The output is a :class:`DeathResult` with the rolled dice and the
table outcome. Specifically:

- ``outcome`` is a :class:`DeathOutcome` literal — one of
  ``knocked_out``, ``lingering_injury``, ``lasting_scar``,
  ``debilitating_wound``, ``crippled``, ``dead``.
- ``hp_after`` is what the call site should write back. ``None`` for
  dead.
- ``description_key`` is a stable key the LLM can use as a hook for
  narration. The engine never writes prose itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from app.game.dice import Roll, roll
from app.game.rules import CharacterStats, DamageResult

DeathOutcome = Literal[
    "knocked_out",
    "lingering_injury",
    "lasting_scar",
    "debilitating_wound",
    "crippled",
    "dead",
]


@dataclass(frozen=True, slots=True)
class DeathResult:
    """Outcome of a roll on the Death and Dismemberment table.

    ``roll`` is the underlying ``2d6`` (raw, before ``modifier``).
    ``total`` is the modified result actually used to look up the row.
    ``hp_after`` is the HP the call site should write to the character;
    ``None`` means the character died.
    """

    character: str
    roll: Roll
    modifier: int
    total: int
    outcome: DeathOutcome
    hp_after: int | None
    description_key: str
    detail: dict[str, object]


def _outcome_for(total: int) -> DeathOutcome:
    """Map the modified 2d6 result to the table row."""

    if total <= 3:
        return "knocked_out"
    if total <= 6:
        return "lingering_injury"
    if total <= 9:
        return "lasting_scar"
    if total <= 12:
        return "debilitating_wound"
    if total <= 15:
        return "crippled"
    return "dead"


_ABILITIES: tuple[str, ...] = ("str", "int", "wis", "dex", "con", "cha")
_LIMBS: tuple[str, ...] = (
    "right_arm",
    "left_arm",
    "right_leg",
    "left_leg",
    "right_eye",
    "left_eye",
)


def death_and_dismemberment(
    character: CharacterStats,
    killing_blow: DamageResult,
    *,
    rng: random.Random,
    critical: bool = False,
) -> DeathResult:
    """Roll on the Death and Dismemberment table for a downed character.

    Caller invariant: ``killing_blow.dropped_to_zero`` is ``True``. The
    function asserts this so a misuse fails loudly rather than silently
    rolling on a no-op blow.
    """

    if not killing_blow.dropped_to_zero:
        raise ValueError(
            "death_and_dismemberment called on a non-killing blow "
            f"({killing_blow.target} at HP {killing_blow.new_hp})"
        )

    base_roll = roll("2d6", rng=rng)
    crit_penalty = 2 if critical else 0
    modifier = killing_blow.below_zero_by + crit_penalty
    total = base_roll.total + modifier
    outcome = _outcome_for(total)

    detail: dict[str, object] = {}
    hp_after: int | None
    if outcome == "knocked_out":
        unconscious_turns = roll("1d6", rng=rng).total
        detail["unconscious_turns"] = unconscious_turns
        hp_after = 0
    elif outcome == "lingering_injury":
        unconscious_minutes = roll("1d10", rng=rng).total
        recovery_days = roll("1d3", rng=rng).total
        detail["unconscious_minutes"] = unconscious_minutes
        detail["recovery_days"] = recovery_days
        hp_after = 0
    elif outcome == "lasting_scar":
        # Pick a body part for flavour; no mechanical effect.
        scar_location = _LIMBS[rng.randrange(len(_LIMBS))]
        detail["scar_location"] = scar_location
        hp_after = 0
    elif outcome == "debilitating_wound":
        # Permanent -1 to a random ability score. Caller persists the
        # ability score change to the character row.
        ability = _ABILITIES[rng.randrange(len(_ABILITIES))]
        detail["ability_penalised"] = ability
        detail["ability_delta"] = -1
        hp_after = 1
    elif outcome == "crippled":
        limb = _LIMBS[rng.randrange(len(_LIMBS))]
        detail["crippled_part"] = limb
        hp_after = 1
    else:  # dead
        hp_after = None

    return DeathResult(
        character=character.name,
        roll=base_roll,
        modifier=modifier,
        total=total,
        outcome=outcome,
        hp_after=hp_after,
        description_key=outcome,
        detail=detail,
    )
