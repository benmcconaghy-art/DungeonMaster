"""Pydantic schema for ``data/bfrpg/races.yaml``.

This schema is owned by the rules-engine agent (the bfrpg-data agent's
brief doesn't describe ``races.yaml``). The shape covers the four core
BFRPG races — Human, Dwarf, Elf, Halfling — and is general enough for
sensible homebrew additions later.

Field summary (all snake_case)::

    - name                       e.g. "Dwarf"
    - description                short flavour string (optional)
    - ability_requirements       per-ability minimum / maximum scores
    - allowed_classes            list of class names this race can take
    - level_caps                 mapping {class_name: max_level | null}
                                 null means uncapped (Human)
    - save_modifiers             {save_kind: bonus} additive ints,
                                 e.g. {magic_wand: 4, paralysis: 4, ...}
    - special_abilities          list of free-form strings (display)
    - allowed_alignments         subset of {lawful, neutral, chaotic}
    - languages                  list of language names
    - movement                   base movement in feet (per turn) — BFRPG
                                 uses 40 for most, 30 for dwarves /
                                 halflings; engine uses this directly
    - infravision                infravision range in feet (0 if none)
    - source                     "core" / "custom" / null
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.game.schemas.class_def import SaveKind

Alignment = Literal["lawful", "neutral", "chaotic"]
Ability = Literal["str", "int", "wis", "dex", "con", "cha"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AbilityRequirement(_Strict):
    """Min / max score for a single ability."""

    min: int | None = None
    max: int | None = None


class RaceDefinition(_Strict):
    """One BFRPG race definition.

    ``level_caps`` maps class name to maximum attainable level, with
    ``None`` (YAML ``null``) meaning uncapped. A class missing from the
    mapping but present in ``allowed_classes`` is treated as uncapped
    (the convention humans use without an explicit table). The engine
    enforces caps at level-up time, not at chargen.
    """

    name: str
    description: str | None = None
    ability_requirements: dict[Ability, AbilityRequirement] = Field(default_factory=dict)
    allowed_classes: Annotated[list[str], Field(min_length=1)]
    level_caps: dict[str, int | None] = Field(default_factory=dict)
    save_modifiers: dict[SaveKind, int] = Field(default_factory=dict)
    special_abilities: list[str] = Field(default_factory=list)
    allowed_alignments: Annotated[list[Alignment], Field(min_length=1)]
    languages: list[str] = Field(default_factory=list)
    movement: int = 40
    infravision: int = 0
    source: str | None = None


class RacesFile(_Strict):
    """Wrapper for the validator: ``{"races": [...]}``."""

    races: Annotated[list[RaceDefinition], Field(min_length=1)]
