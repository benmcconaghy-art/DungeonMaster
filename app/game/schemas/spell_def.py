"""Pydantic schema for ``data/bfrpg/spells.yaml``.

A top-level list of spells. ``caster_classes`` is a list because several
BFRPG spells appear on both the cleric and magic-user lists at the same
or different levels (e.g. Light, Hold Person, Detect Magic). Storing the
caster eligibility as a list collapses what would otherwise be duplicate
entries with disambiguating names; lookup helpers in
:mod:`app.game.classes` filter to a single class at query time.
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
    caster_classes: Annotated[list[CasterClass], Field(min_length=1)]
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
