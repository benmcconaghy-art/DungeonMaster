"""Tool handler registration.

Importing this package triggers the ``@register("tool_name")`` decorators
in each per-tool module, populating
:data:`app.llm.tools._HANDLERS`. The orchestrator imports this package
once at module load (via ``from . import handlers # noqa: F401``) so a
single import bootstraps the entire dispatch table.

Per-tool modules:

  - ``request_dice_roll`` — evaluate dice and write an audit row.
  - ``apply_damage`` — read HP from DB, subtract, persist;
    triggers Death and Dismemberment if HP drops to zero.
  - ``heal`` — restore HP, capped at hp_max.
  - ``transition_location`` — move the party between locations.
  - ``whisper`` — DM-to-one-player private message.
  - ``start_encounter`` — create an active encounter, roll initiative.
  - ``end_encounter`` — close an encounter, record the outcome.

The dispatcher calls each handler inside its own tight transaction
block (AGENTS.md invariant #2 — the streaming call must not happen
while a transaction is held).
"""

from __future__ import annotations

# ``ToolHandler`` in ``app.llm.tools`` references ``AsyncSession`` via a
# string annotation behind ``TYPE_CHECKING``. Pydantic refuses to
# instantiate the model until the forward reference is resolved, so we
# import ``AsyncSession`` here and call ``model_rebuild()`` *before*
# the per-handler modules import — each handler's ``@register`` runs
# during its module body, and the decorator builds a ``ToolHandler``.
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — type resolved by pydantic

from app.llm.tools import ToolHandler

ToolHandler.model_rebuild()

# Importing each module evaluates its @register(...) decorator. The
# imports look unused but they're load-bearing — keep them.
from app.orchestrator.handlers import (  # noqa: E402
    apply_damage,
    end_encounter,
    heal,
    request_dice_roll,
    start_encounter,
    transition_location,
    whisper,
)

__all__ = [
    "apply_damage",
    "end_encounter",
    "heal",
    "request_dice_roll",
    "start_encounter",
    "transition_location",
    "whisper",
]
