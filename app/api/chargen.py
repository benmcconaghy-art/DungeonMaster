"""Chargen utility endpoints (Phase 6.5).

Backs the chargen UI's re-roll affordance. The roll lives in the
page until the player commits via the existing
``POST /api/campaigns/{id}/characters`` endpoint — so this surface
is intentionally stateless.

Why not also expose an "eligible races/classes" endpoint? The
eligibility computation is two data lookups: ability scores meeting
``race.ability_requirements`` and class membership in
``race.allowed_classes``. The chargen page already needs the full
race + class tables to render the cards; doing the filter in 30
lines of JS off the embedded data is cheaper than a round-trip
per re-roll. The server-side validation in
``generate_character`` stays authoritative — if the eligibility
filter is wrong, the commit endpoint still rejects.
"""

from __future__ import annotations

import random
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.deps import CurrentUser
from app.game.chargen import roll_abilities
from app.game.rules import ability_modifier

router = APIRouter(prefix="/api/chargen", tags=["chargen"])


class RollAbilitiesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["classic", "heroic"] = Field(
        default="classic",
        description="'classic' = 3d6 in order; 'heroic' = 4d6 drop the lowest, in order.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional RNG seed for reproducible rolls (test convenience).",
    )


class AbilityRoll(BaseModel):
    score: int
    modifier: int


class RollAbilitiesResponse(BaseModel):
    abilities: dict[str, AbilityRoll]


@router.post("/roll-abilities", response_model=RollAbilitiesResponse)
async def roll_abilities_endpoint(
    payload: RollAbilitiesRequest,
    user: CurrentUser,
) -> RollAbilitiesResponse:
    """Roll a fresh ability-score block and return scores + modifiers.

    Authenticated (any logged-in user can roll). The result lives only
    in the caller's UI state — nothing persists until the commit endpoint
    creates a character.
    """

    rng = random.Random(payload.seed) if payload.seed is not None else random.Random()
    scores = roll_abilities(payload.method, rng=rng)

    def _pack(score: int) -> AbilityRoll:
        return AbilityRoll(score=score, modifier=ability_modifier(score))

    return RollAbilitiesResponse(
        abilities={
            "str": _pack(scores.str_score),
            "int": _pack(scores.int_score),
            "wis": _pack(scores.wis_score),
            "dex": _pack(scores.dex_score),
            "con": _pack(scores.con_score),
            "cha": _pack(scores.cha_score),
        }
    )


__all__ = [
    "AbilityRoll",
    "RollAbilitiesRequest",
    "RollAbilitiesResponse",
    "router",
]
