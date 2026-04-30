"""Pydantic schema for ``data/bfrpg/classes.yaml``.

Mirrors the shape documented in ``.claude/agents/bfrpg-data.md`` —
``hit_die``, ``prime_requisite``, ``saves`` keyed by level, and
``attack_bonus_progression`` keyed by level. Per-class additions
(Cleric ``turn_undead`` table, Thief ``skills`` table, Magic-User
``spells_per_day`` table, etc.) are accepted as optional structured
fields so the bfrpg-data agent can include them without schema churn.

YAML files validate against ``ClassesFile``: the file is a top-level
list, so the loader passes ``{"classes": [...]}`` to the wrapper.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# BFRPG saves bucket the d20-≥-N target into five categories. We keep the
# names lowercased and snake_cased for consistency with the rest of the
# codebase.
SaveKind = Literal["death_ray", "magic_wand", "paralysis", "dragon_breath", "spells"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SaveTargets(_Strict):
    """The five BFRPG save targets at a given level (``d20 ≥ N``)."""

    death_ray: int
    magic_wand: int
    paralysis: int
    dragon_breath: int
    spells: int


# Bonded by BFRPG's d-notation. ``d4`` through ``d12`` covers every core
# class. We don't accept ``2d6``-style hit dice — BFRPG hit dice are a
# single die that resolves once per level.
HitDie = Literal["d4", "d6", "d8", "d10", "d12"]
PrimeReq = Literal["STR", "INT", "WIS", "DEX", "CON", "CHA"]


class TurnUndeadEntry(_Strict):
    """One row of the cleric's turn-undead table.

    Per BFRPG: a cleric of level ``cleric_level`` rolling on the
    ``hd_or_type`` row (e.g. ``skeleton``, ``zombie``, ``ghoul``) needs
    a ``2d6 ≥ target`` result. ``"T"`` means "automatic turn", ``"D"``
    means "automatic destroy". We model both as the literal strings so
    the bfrpg-data agent can match the source-table notation directly.
    """

    hd_or_type: str
    target: int | Literal["T", "D", "-"]


class ThiefSkills(_Strict):
    """The thief's percentile skill table at a given level."""

    open_locks: int
    remove_traps: int
    pick_pockets: int
    move_silently: int
    climb_walls: int
    hide: int
    listen: int


class SpellsPerDay(_Strict):
    """Spell slots per day for a given caster level.

    Keys are spell levels (1-6 in core BFRPG); values are slot counts.
    Stored as a free-form mapping so future expansions don't break
    validation.
    """

    model_config = ConfigDict(extra="allow")


class ClassDefinition(_Strict):
    """One BFRPG class (Cleric, Fighter, Magic-User, Thief, …).

    The optional ``special_abilities`` field is a list of free-form
    strings naming the class's distinguishing features ("Turn Undead",
    "Backstab", "Read Magic"). Mechanical detail for those abilities
    lives in the dedicated structured fields below where applicable;
    the string list is for prose / display.
    """

    name: str
    hit_die: HitDie
    prime_requisite: PrimeReq | list[PrimeReq]
    prime_req_bonus_threshold: int | None = None
    weapon_restrictions: str
    armour_restrictions: str
    saves: dict[int, SaveTargets]
    attack_bonus_progression: dict[int, int]
    xp_progression: dict[int, int] | None = None
    special_abilities: list[str] = Field(default_factory=list)

    # Optional, class-specific structured tables.
    turn_undead: list[TurnUndeadEntry] | None = None
    thief_skills: dict[int, ThiefSkills] | None = None
    spells_per_day: dict[int, SpellsPerDay] | None = None

    # Catch-all for content-only metadata the agent may attach
    # (description, source attribution). Engine ignores these.
    description: str | None = None
    source: str | None = None

    @field_validator("saves", "attack_bonus_progression", "xp_progression", mode="before")
    @classmethod
    def _coerce_int_keys(cls, value: Any) -> Any:
        """YAML numbers parse as ``int`` already, but allow string keys too."""

        if isinstance(value, dict):
            return {int(k): v for k, v in value.items()}
        return value


class ClassesFile(_Strict):
    """Wrapper used by the validator: ``{"classes": [...]}``."""

    classes: Annotated[list[ClassDefinition], Field(min_length=1)]
