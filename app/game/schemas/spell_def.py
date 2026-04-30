"""Pydantic schema for ``data/bfrpg/spells.yaml``.

Mirrors the shape in ``.claude/agents/bfrpg-data.md``: a top-level list
of spells with name, level, caster class, range, duration, description,
optional damage / heal expression, and optional reversed form.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

CasterClass = Literal["cleric", "magic_user", "druid"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpellDefinition(_Strict):
    """A single BFRPG spell."""

    name: str
    level: Annotated[int, Field(ge=1, le=9)]
    caster_class: CasterClass
    range: str
    duration: str
    description: str
    damage_or_heal: str | None = None
    reversed_form: str | None = None
    components: list[str] | None = None
    save: str | None = None
    source: str | None = None


class SpellsFile(_Strict):
    """Wrapper: ``{"spells": [...]}``."""

    spells: Annotated[list[SpellDefinition], Field(min_length=1)]
