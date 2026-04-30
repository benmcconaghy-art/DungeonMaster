"""Character endpoints (minimal Phase 2 surface).

One endpoint: roll up a new PC for a campaign the user owns. Random
ability scores via 3d6-in-order by default; the request can pass fixed
abilities to skip the roll (useful for tests, or for players who roll
elsewhere).

Full character editing, level-up, and equipment endpoints land in later
phases when there's UI for them.
"""

from __future__ import annotations

import random

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.db import models
from app.deps import CurrentUser, DbSession
from app.game.chargen import AbilityScores, generate_character

router = APIRouter(prefix="/api/campaigns", tags=["characters"])


class CreateCharacterRequest(BaseModel):
    """Field is named ``class_name`` (matching the DB column) since
    ``class`` is a Python keyword. Callers send ``class_name`` directly
    rather than ``class`` — no alias needed for the internal API; UI
    work in Phase 6 will introduce a friendlier facade if useful."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    race: str = Field(description="Must match a name in data/bfrpg/races.yaml.")
    class_name: str = Field(description="Must match a name in data/bfrpg/classes.yaml.")
    alignment: str
    method: str = Field(default="classic", description="'classic' (3d6) or 'heroic' (4d6kh3).")
    seed: int | None = Field(
        default=None,
        description="Optional RNG seed for reproducible rolls (test convenience).",
    )
    abilities: dict[str, int] | None = Field(
        default=None,
        description=(
            "Optional fixed ability scores to skip the roll. Keys: str/int/wis/dex/con/cha."
        ),
    )


class CharacterResponse(BaseModel):
    id: str
    name: str
    race: str
    class_name: str
    level: int
    hp_current: int
    hp_max: int
    ac: int
    alignment: str
    gold: int
    str_score: int
    int_score: int
    wis_score: int
    dex_score: int
    con_score: int
    cha_score: int


def _character_to_response(character: models.Character) -> CharacterResponse:
    return CharacterResponse(
        id=character.id,
        name=character.name,
        race=character.race,
        class_name=character.class_name,
        level=character.level,
        hp_current=character.hp_current,
        hp_max=character.hp_max,
        ac=character.ac,
        alignment=character.alignment,
        gold=character.gold,
        str_score=character.str_score,
        int_score=character.int_score,
        wis_score=character.wis_score,
        dex_score=character.dex_score,
        con_score=character.con_score,
        cha_score=character.cha_score,
    )


def _abilities_from_dict(data: dict[str, int]) -> AbilityScores:
    """Pack a flat ``{str: int, ...}`` dict into the dataclass."""

    def pick(key: str) -> int:
        try:
            value = int(data[key])
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"abilities missing key {key!r}",
            ) from exc
        if not 3 <= value <= 18:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"ability {key!r} score {value} out of 3..18 range",
            )
        return value

    return AbilityScores(
        str_score=pick("str"),
        int_score=pick("int"),
        wis_score=pick("wis"),
        dex_score=pick("dex"),
        con_score=pick("con"),
        cha_score=pick("cha"),
    )


@router.post(
    "/{campaign_id}/characters",
    response_model=CharacterResponse,
    status_code=status.HTTP_201_CREATED,
)
async def roll_character(
    campaign_id: str,
    payload: CreateCharacterRequest,
    user: CurrentUser,
    db: DbSession,
) -> CharacterResponse:
    """Create a new PC in ``campaign_id`` for the current user.

    Verifies the user is a member of the campaign before persisting.
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

    rng = random.Random(payload.seed) if payload.seed is not None else random.Random()
    abilities = _abilities_from_dict(payload.abilities) if payload.abilities is not None else None

    if payload.method not in {"classic", "heroic"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown method {payload.method!r}; expected 'classic' or 'heroic'",
        )

    try:
        rolled = generate_character(
            name=payload.name,
            race_name=payload.race,
            class_name=payload.class_name,
            alignment=payload.alignment,
            rng=rng,
            method=payload.method,  # type: ignore[arg-type]
            abilities=abilities,
        )
    except (ValueError, KeyError) as exc:
        # ValueError: chargen rejected (race/class incompatible, alignment, etc.).
        # KeyError: missing race or class in the YAML loaders.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    character = models.Character(
        user_id=user.id,
        campaign_id=campaign_id,
        name=rolled.name,
        race=rolled.race,
        class_name=rolled.class_name,
        level=rolled.level,
        hp_current=rolled.hp_max,
        hp_max=rolled.hp_max,
        ac=rolled.ac,
        str_score=rolled.abilities.str_score,
        int_score=rolled.abilities.int_score,
        wis_score=rolled.abilities.wis_score,
        dex_score=rolled.abilities.dex_score,
        con_score=rolled.abilities.con_score,
        cha_score=rolled.abilities.cha_score,
        gold=rolled.starting_gold,
        alignment=rolled.alignment,
    )
    db.add(character)
    await db.commit()
    await db.refresh(character)
    return _character_to_response(character)


__all__ = ["CharacterResponse", "CreateCharacterRequest", "router"]
