"""Character endpoints.

Phase 2 shipped chargen (POST /api/campaigns/{id}/characters). Phase 6
adds the read + notes-edit surface the character sheet view needs:

  - GET   /api/characters/{id}        — full sheet detail
  - PATCH /api/characters/{id}/notes  — player-editable notes

Notes live inside the existing ``characters.sheet`` JSON column so
no schema migration is needed for v1. A future phase that wants
richer notes (e.g. multi-section, markdown, version history) can
promote them to their own table.
"""

from __future__ import annotations

import random
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.db import models
from app.deps import CurrentUser, DbSession
from app.game.chargen import AbilityScores, generate_character
from app.game.classes import get_class
from app.game.rules import ability_modifier

router = APIRouter(tags=["characters"])
campaign_scoped_router = APIRouter(prefix="/api/campaigns", tags=["characters"])


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
    pronouns: str | None = Field(
        default=None,
        max_length=40,
        description="Free-form pronouns (e.g. 'she/her', 'they/them'). NULL = unspecified.",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Player-supplied appearance flavor text. 500-char limit protects prompt budget "
            "— at ~125 tokens per character this keeps a 4-PC party under 500 extra tokens."
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
    pronouns: str | None = None
    description: str | None = None


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
        pronouns=character.pronouns,
        description=character.description,
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


@campaign_scoped_router.post(
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
        pronouns=payload.pronouns,
        description=payload.description,
    )
    db.add(character)
    await db.commit()
    await db.refresh(character)
    return _character_to_response(character)


# ---------------------------------------------------------------------------
# Phase 6: detail + notes
# ---------------------------------------------------------------------------


# Class names that have spells in BFRPG core. The character sheet
# only renders the spells panel for these — non-spellcasters get the
# panel hidden entirely (matches the design's "section hidden when
# not applicable" convention rather than rendering an empty placeholder).
SPELLCASTER_CLASSES = {"Cleric", "Magic-User"}


class AbilityDetail(BaseModel):
    """One row of the abilities grid: score + computed modifier
    sign-prefixed (``+1``, ``+0``, ``-2``)."""

    score: int
    modifier: int


class SaveDetail(BaseModel):
    """One saving-throw row: kind label + target d20 number to roll
    at-or-above to succeed. ``modifier`` accounts for racial bonuses
    (dwarf +4 etc.) — applied as ``d20 + modifier ≥ target``."""

    kind: str
    label: str
    target: int


class InventoryItemResponse(BaseModel):
    id: str
    name: str
    item_type: str
    quantity: int
    equipped: bool


class SpellResponse(BaseModel):
    id: str
    spell_name: str
    spell_level: int
    prepared: bool


class CharacterDetailResponse(BaseModel):
    """Full sheet payload for the ``/characters/{id}`` view.

    Composed server-side — the template doesn't recompute modifiers
    or look up class saves. ``is_mine`` lets the UI decide whether to
    show the player-editable affordances (notes textarea, portrait
    regen). ``is_spellcaster`` controls the spells-panel visibility.
    """

    id: str
    name: str
    race: str
    class_name: str
    level: int
    alignment: str
    status: str
    xp: int
    gold: int
    hp_current: int
    hp_max: int
    ac: int
    abilities: dict[str, AbilityDetail]
    saves: list[SaveDetail]
    inventory: list[InventoryItemResponse]
    spells: list[SpellResponse]
    is_spellcaster: bool
    notes: str
    pronouns: str | None
    description: str | None
    canonical_image_id: str | None
    is_mine: bool


class UpdateNotesRequest(BaseModel):
    """Player-editable notes update. Empty string clears them."""

    notes: str = Field(max_length=5000)


class UpdateAppearanceRequest(BaseModel):
    """Owner-only appearance update. Either field may be None to clear it."""

    model_config = ConfigDict(extra="forbid")

    pronouns: str | None = Field(default=None, max_length=40)
    description: str | None = Field(default=None, max_length=500)


def _signed_modifier(score: int) -> int:
    return ability_modifier(score)


def _saves_for(class_name: str, level: int) -> dict[str, int]:
    """Resolve the saves row for a class+level. Falls back to the
    closest defined level if the requested one isn't present (level
    progression is sparse for some classes' lower levels)."""

    try:
        cls = get_class(class_name)
    except KeyError:
        return {}
    saves_by_level = cls.saves
    if not saves_by_level:
        return {}
    if level in saves_by_level:
        return dict(saves_by_level[level])
    # Use the highest defined level <= requested.
    candidates = [lv for lv in saves_by_level if lv <= level]
    if candidates:
        return dict(saves_by_level[max(candidates)])
    # Otherwise lowest defined level.
    return dict(saves_by_level[min(saves_by_level)])


_SAVE_LABELS = {
    "death_ray": "Death Ray / Poison",
    "magic_wand": "Magic Wands",
    "paralysis": "Paralysis / Petrify",
    "dragon_breath": "Dragon Breath",
    "spells": "Spells",
}


async def _require_character_visibility(
    db: DbSession, *, character_id: str, user: models.User
) -> models.Character:
    """Resolve a character and verify the current user can see it.

    Visible iff the user is a member of the character's parent
    campaign — players see each other's sheets within the same table.
    Editing is gated separately by ``is_mine``.
    """

    character = await db.get(models.Character, character_id)
    if character is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="character not found")
    membership = await db.get(models.CampaignMember, (character.campaign_id, user.id))
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="not a member of this campaign",
        )
    return character


def _detail_response(
    character: models.Character,
    *,
    viewer_id: str,
    inventory: list[models.InventoryItem],
    spells: list[models.SpellKnown],
) -> CharacterDetailResponse:
    def _ability(score: int) -> AbilityDetail:
        return AbilityDetail(score=score, modifier=_signed_modifier(score))

    abilities = {
        "str": _ability(character.str_score),
        "int": _ability(character.int_score),
        "wis": _ability(character.wis_score),
        "dex": _ability(character.dex_score),
        "con": _ability(character.con_score),
        "cha": _ability(character.cha_score),
    }
    saves_dict = _saves_for(character.class_name, character.level)
    saves = [
        SaveDetail(kind=kind, label=_SAVE_LABELS.get(kind, kind), target=int(target))
        for kind, target in saves_dict.items()
    ]
    sheet: dict[str, Any] = character.sheet or {}
    notes = str(sheet.get("notes", ""))
    return CharacterDetailResponse(
        id=character.id,
        name=character.name,
        race=character.race,
        class_name=character.class_name,
        level=character.level,
        alignment=character.alignment,
        status=character.status,
        xp=character.xp,
        gold=character.gold,
        hp_current=character.hp_current,
        hp_max=character.hp_max,
        ac=character.ac,
        abilities=abilities,
        saves=saves,
        inventory=[
            InventoryItemResponse(
                id=item.id,
                name=item.name,
                item_type=item.item_type,
                quantity=item.quantity,
                equipped=item.equipped,
            )
            for item in inventory
        ],
        spells=[
            SpellResponse(
                id=spell.id,
                spell_name=spell.spell_name,
                spell_level=spell.spell_level,
                prepared=spell.prepared,
            )
            for spell in spells
        ],
        is_spellcaster=character.class_name in SPELLCASTER_CLASSES,
        notes=notes,
        pronouns=character.pronouns,
        description=character.description,
        canonical_image_id=character.canonical_image_id,
        is_mine=character.user_id == viewer_id,
    )


@router.get("/api/characters/{character_id}", response_model=CharacterDetailResponse)
async def get_character(
    character_id: str,
    user: CurrentUser,
    db: DbSession,
) -> CharacterDetailResponse:
    """Return the full sheet detail. Visible to any member of the
    parent campaign; editing is gated separately."""

    character = await _require_character_visibility(db, character_id=character_id, user=user)
    inventory = list(
        (
            await db.execute(
                select(models.InventoryItem)
                .where(models.InventoryItem.character_id == character_id)
                .order_by(models.InventoryItem.equipped.desc(), models.InventoryItem.name)
            )
        ).scalars()
    )
    spells = list(
        (
            await db.execute(
                select(models.SpellKnown)
                .where(models.SpellKnown.character_id == character_id)
                .order_by(models.SpellKnown.spell_level, models.SpellKnown.spell_name)
            )
        ).scalars()
    )
    return _detail_response(character, viewer_id=user.id, inventory=inventory, spells=spells)


@router.patch(
    "/api/characters/{character_id}/notes",
    response_model=CharacterDetailResponse,
)
async def update_notes(
    character_id: str,
    payload: UpdateNotesRequest,
    user: CurrentUser,
    db: DbSession,
) -> CharacterDetailResponse:
    """Owner-only: replace the character's player-editable notes.

    Notes live in the existing ``characters.sheet`` JSON under
    ``"notes"`` — keeps the schema lean. A future phase that wants
    per-section notes or version history can promote them.
    """

    character = await _require_character_visibility(db, character_id=character_id, user=user)
    if character.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="only the owning player can edit notes",
        )
    sheet = dict(character.sheet or {})
    sheet["notes"] = payload.notes
    character.sheet = sheet
    await db.commit()
    await db.refresh(character)

    inventory = list(
        (
            await db.execute(
                select(models.InventoryItem)
                .where(models.InventoryItem.character_id == character_id)
                .order_by(models.InventoryItem.equipped.desc(), models.InventoryItem.name)
            )
        ).scalars()
    )
    spells = list(
        (
            await db.execute(
                select(models.SpellKnown)
                .where(models.SpellKnown.character_id == character_id)
                .order_by(models.SpellKnown.spell_level, models.SpellKnown.spell_name)
            )
        ).scalars()
    )
    return _detail_response(character, viewer_id=user.id, inventory=inventory, spells=spells)


@router.patch(
    "/api/characters/{character_id}/appearance",
    response_model=CharacterDetailResponse,
)
async def update_appearance(
    character_id: str,
    payload: UpdateAppearanceRequest,
    user: CurrentUser,
    db: DbSession,
) -> CharacterDetailResponse:
    """Owner-only: set or clear pronouns and appearance description.

    Both fields are nullable — passing None clears the field. The 500-char
    limit on description protects per-turn DM prompt budget (see
    AGENTS.md convention on player-supplied freeform fields).
    """

    character = await _require_character_visibility(db, character_id=character_id, user=user)
    if character.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="only the owning player can edit appearance",
        )
    character.pronouns = payload.pronouns
    character.description = payload.description
    await db.commit()
    await db.refresh(character)

    inventory = list(
        (
            await db.execute(
                select(models.InventoryItem)
                .where(models.InventoryItem.character_id == character_id)
                .order_by(models.InventoryItem.equipped.desc(), models.InventoryItem.name)
            )
        ).scalars()
    )
    spells = list(
        (
            await db.execute(
                select(models.SpellKnown)
                .where(models.SpellKnown.character_id == character_id)
                .order_by(models.SpellKnown.spell_level, models.SpellKnown.spell_name)
            )
        ).scalars()
    )
    return _detail_response(character, viewer_id=user.id, inventory=inventory, spells=spells)


__all__ = [
    "AbilityDetail",
    "CharacterDetailResponse",
    "CharacterResponse",
    "CreateCharacterRequest",
    "InventoryItemResponse",
    "SaveDetail",
    "SpellResponse",
    "UpdateAppearanceRequest",
    "UpdateNotesRequest",
    "campaign_scoped_router",
    "router",
]
