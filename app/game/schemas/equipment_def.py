"""Pydantic schema for ``data/bfrpg/equipment.yaml``.

Mirrors the layout in ``.claude/agents/bfrpg-data.md`` — a single
top-level mapping with keys ``weapons``, ``armour``, and ``gear``.
``cost_gp`` and ``cost_sp`` are the two coin-units BFRPG mixes; we
accept both so the bfrpg-data agent can match the source pricing
without a unit-conversion pass at author time.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WeaponType = Literal["melee", "ranged", "thrown"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Costed(_Strict):
    """Mixin: at least one of ``cost_gp`` / ``cost_sp`` / ``cost_cp`` set."""

    cost_gp: int | None = None
    cost_sp: int | None = None
    cost_cp: int | None = None

    @model_validator(mode="after")
    def _require_a_cost(self) -> _Costed:
        if self.cost_gp is None and self.cost_sp is None and self.cost_cp is None:
            raise ValueError("at least one of cost_gp, cost_sp, cost_cp must be set")
        return self


class WeaponDefinition(_Costed):
    name: str
    damage: str
    weight: Annotated[int, Field(ge=0)]
    type: WeaponType
    range: list[int] | None = None  # short / medium / long for ranged / thrown
    two_handed: bool = False
    size: Literal["small", "medium", "large"] | None = None
    source: str | None = None


class ArmorDefinition(_Costed):
    name: str
    ac_bonus: Annotated[int, Field(ge=0)]
    weight: Annotated[int, Field(ge=0)]
    is_shield: bool = False
    source: str | None = None


class GearDefinition(_Costed):
    name: str
    weight: Annotated[int, Field(ge=0)] = 0
    description: str | None = None
    source: str | None = None


class EquipmentFile(_Strict):
    """Top-level equipment YAML: weapons, armour, gear."""

    weapons: Annotated[list[WeaponDefinition], Field(min_length=1)]
    armour: Annotated[list[ArmorDefinition], Field(min_length=1)]
    gear: list[GearDefinition] = Field(default_factory=list)
