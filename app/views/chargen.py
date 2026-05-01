"""Chargen view-context builder (Phase 6.5).

The chargen page (``GET /campaigns/{id}/chargen``) renders an initial
ability roll plus the full race / class tables, so the JS layer can
do eligibility filtering without a round-trip per heritage hover.
The roll-and-replace path uses ``POST /api/chargen/roll-abilities``;
commit goes through the existing
``POST /api/campaigns/{id}/characters``.

This composer keeps the route handler short and gives tests a single
seam to hit — pass a seeded ``random.Random`` to make the initial roll
reproducible.
"""

from __future__ import annotations

import random
from typing import Any

from fastapi import HTTPException, status

from app.db import models
from app.deps import DbSession
from app.game.chargen import AbilityScores, roll_abilities
from app.game.classes import load_classes
from app.game.races import load_races
from app.game.rules import ability_modifier


def _abilities_payload(scores: AbilityScores) -> dict[str, dict[str, int]]:
    """Pack ``AbilityScores`` as ``{key: {score, modifier}}`` for the template."""

    return {
        "str": {"score": scores.str_score, "modifier": ability_modifier(scores.str_score)},
        "int": {"score": scores.int_score, "modifier": ability_modifier(scores.int_score)},
        "wis": {"score": scores.wis_score, "modifier": ability_modifier(scores.wis_score)},
        "dex": {"score": scores.dex_score, "modifier": ability_modifier(scores.dex_score)},
        "con": {"score": scores.con_score, "modifier": ability_modifier(scores.con_score)},
        "cha": {"score": scores.cha_score, "modifier": ability_modifier(scores.cha_score)},
    }


def _race_card(race: Any) -> dict[str, Any]:
    """Trim a ``RaceDefinition`` down to what the template + JS need."""

    return {
        "name": race.name,
        "description": race.description or "",
        "allowed_classes": list(race.allowed_classes),
        "allowed_alignments": list(race.allowed_alignments),
        "ability_requirements": {
            ability: {"min": req.min, "max": req.max}
            for ability, req in race.ability_requirements.items()
        },
        "special_abilities": list(race.special_abilities),
    }


def _class_card(cls: Any) -> dict[str, Any]:
    """Trim a ``ClassDefinition`` down to what the template + JS need."""

    prime = cls.prime_requisite
    prime_label = prime if isinstance(prime, str) else "/".join(prime)
    return {
        "name": cls.name,
        "description": cls.description or "",
        "hit_die": cls.hit_die,
        "prime_requisite": prime_label,
        "prime_req_bonus_threshold": cls.prime_req_bonus_threshold,
        "weapon_restrictions": cls.weapon_restrictions,
        "armour_restrictions": cls.armour_restrictions,
    }


async def build_context(
    db: DbSession,
    *,
    campaign_id: str,
    user: models.User,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Compose the chargen render context.

    Verifies ``user`` is a member of ``campaign_id`` (raises 404 / 403
    via ``HTTPException``). Rolls an initial classic 3d6 set so the
    page lands with numbers already on it; the player can re-roll or
    switch method from the UI.
    """

    campaign = await db.get(models.Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="campaign not found")
    membership = await db.get(models.CampaignMember, (campaign_id, user.id))
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="not a member of this campaign",
        )

    initial_rng = rng if rng is not None else random.Random()
    scores = roll_abilities("classic", rng=initial_rng)

    races = [_race_card(race) for race in load_races().values()]
    classes = [_class_card(cls) for cls in load_classes().values()]

    return {
        "user": user,
        "campaign": campaign,
        "abilities": _abilities_payload(scores),
        "races": races,
        "classes": classes,
    }


__all__ = ["build_context"]
