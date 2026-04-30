"""Pydantic schema for ``data/bfrpg/monsters.yaml``.

``hit_dice`` is a string because BFRPG uses fractional notation like
``"1-1"`` and ``"½"``; the loader parses it into the engine's
:class:`HitDice` representation.

Movement is a structured ``movement_modes: {mode: feet}`` mapping rather
than a single int: fliers, swimmers, and climbers commonly have multiple
locomotion modes and Phase 2's combat / pursuit logic needs to ask "what
speed in this terrain?" structurally rather than by parsing prose.
Conventional keys (not enforced as a Literal so future modes don't
require schema churn): ``land``, ``fly``, ``swim``, ``climb``, ``burrow``,
``hover``. Single-mode creatures use ``{"land": N}``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Alignment = Literal["lawful", "neutral", "chaotic"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MonsterAttack(_Strict):
    name: str
    damage: str
    to_hit_bonus: int = 0
    range: list[int] | None = None  # short / medium / long
    notes: str | None = None
    # Free-form text describing a special effect of this attack (poison,
    # paralysis, energy drain, etc.). The engine surfaces it to the LLM
    # for narration; mechanical resolution of the named effect is the
    # caller's responsibility.
    special: str | None = None


class MonsterDefinition(_Strict):
    """One creature stat block.

    Some monsters have no melee/ranged attacks (e.g. green slime — it
    only attacks via reactive abilities described in
    ``special_abilities``). For those we accept an empty ``attacks``
    list and rely on ``special_abilities`` to describe the threat.
    """

    name: str
    hit_dice: str
    hp_typical: Annotated[int, Field(ge=1)]
    ac: int
    movement_modes: Annotated[dict[str, Annotated[int, Field(ge=0)]], Field(min_length=1)]
    attacks: list[MonsterAttack] = Field(default_factory=list)
    no_appearing: str | None = None
    save_as: str  # e.g. "F1", "MU3"
    morale: Annotated[int, Field(ge=2, le=12)]
    alignment: Alignment
    treasure_type: str | None = None
    xp: Annotated[int, Field(ge=0)]
    description: str
    ecology: str | None = None
    special_abilities: list[str] | None = None
    source: str | None = None


class MonstersFile(_Strict):
    """Wrapper: ``{"monsters": [...]}``."""

    monsters: Annotated[list[MonsterDefinition], Field(min_length=1)]
