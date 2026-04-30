"""Tool schemas and dispatcher for LLM-driven state mutations.

The DM never declares mechanical outcomes — it requests them via tool
calls (AGENTS.md invariant #1). This module is the single source of
truth for those tools: a Pydantic model per tool defines the parameter
shape, a registry maps tool names to their handlers, and the dispatcher
executes a tool call against authoritative DB state.

Spec §7's TOOLS list (verbatim, with type-strictness added):

  - request_dice_roll, apply_damage, heal
  - award_xp, award_treasure
  - transition_location, spawn_npc, generate_scene_image
  - whisper, start_encounter, end_encounter
  - mark_beat, reveal_secret

Phase 2 wires up the handlers for the seven that are immediately
relevant to the single-player text-only DM loop: request_dice_roll,
apply_damage, heal, transition_location, whisper, start_encounter,
end_encounter. The other six are declared here so the model can see
them in its tool list (and Phase 3 / 5 / 8 fill in their handlers
without the schema churning).

The OpenAI tool-call convention expects each tool surfaced as
``{"type": "function", "function": {"name": ..., "description": ...,
"parameters": JSONSchema}}``. ``tool_definitions()`` emits exactly that
shape from these Pydantic models.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# Tool argument schemas
# ---------------------------------------------------------------------------


class _ToolArgs(BaseModel):
    """Base for every tool argument model. Strict so the model can't slip
    extra fields past us (which would silently no-op)."""

    model_config = ConfigDict(extra="forbid")


class DiceTarget(_ToolArgs):
    """Target the dice roll resolves against, if any."""

    kind: Literal["ac", "dc", "none"]
    value: int | None = None


class RequestDiceRoll(_ToolArgs):
    """Ask the engine to roll dice. Used for any check, save, attack, or
    damage roll the LLM wants to drive."""

    expression: str = Field(description="Dice expression, e.g. '1d20+3' or '2d6'.")
    purpose: str = Field(description="Human-readable reason for the roll.")
    actor: str = Field(description="character_id, monster_id, or 'dm'.")
    target: DiceTarget | None = None


class ApplyDamage(_ToolArgs):
    """Reduce a creature's HP by the given amount. The handler reads the
    current HP from the database — never trust LLM-supplied state."""

    target_id: str
    amount: int = Field(ge=1)
    source: str = Field(description="Short tag for the dice_rolls audit, e.g. 'goblin scimitar'.")


class Heal(_ToolArgs):
    target_id: str
    amount: int = Field(ge=1)
    source: str = Field(default="", description="Optional context string.")


class AwardXp(_ToolArgs):
    character_ids: list[str] = Field(min_length=1)
    amount: int = Field(ge=1)
    reason: str


class TreasureItem(_ToolArgs):
    name: str
    item_type: str = "misc"
    quantity: int = Field(default=1, ge=1)


class AwardTreasure(_ToolArgs):
    character_ids: list[str] = Field(min_length=1)
    items: list[TreasureItem] = Field(default_factory=list)
    gold: int = Field(default=0, ge=0)


class TransitionLocation(_ToolArgs):
    location_id: str
    description: str = Field(description="Brief narrative caption for the move.")


class NpcStats(_ToolArgs):
    """Loose creature stats for an ad-hoc NPC the DM is spawning. Strict
    types deferred — most fields here are LLM-authored text."""

    model_config = ConfigDict(extra="allow")


class SpawnNpc(_ToolArgs):
    name: str
    stats: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    auto_portrait: bool = Field(
        default=True,
        description=(
            "If true (default), enqueue a canonical portrait so the NPC stays"
            " visually consistent across later scene edits. Set false for"
            " transient walk-ons (a guard, a stable hand) where the image"
            " cost is wasted."
        ),
    )


class GenerateSceneImage(_ToolArgs):
    """Queue an illustration. ``prompt`` is either a from-scratch scene
    description (no reference set) or a Kontext edit instruction
    ("same character, torchlit crypt, sword drawn, blood on armour").

    Reference fields are mutually exclusive:

    - both unset → FLUX ``/generate`` from the prompt alone.
    - ``reference_character_id`` set → look up that PC's
      canonical portrait and dispatch via Kontext ``/edit`` for
      identity-preserving scene rendering.
    - ``reference_npc_id`` set → same for NPCs.
    """

    prompt: str
    kind: Literal["scene", "npc", "item", "map"] = "scene"
    reference_character_id: str | None = Field(
        default=None,
        description=(
            "If set, treat ``prompt`` as a Kontext edit instruction"
            " against that character's canonical portrait."
        ),
    )
    reference_npc_id: str | None = Field(
        default=None,
        description=(
            "If set, treat ``prompt`` as a Kontext edit instruction"
            " against that NPC's canonical portrait."
        ),
    )


class Whisper(_ToolArgs):
    """Private DM message to a single character — visible to the DM in
    history (so it stays consistent) but only sent to the target's
    client."""

    character_id: str
    content: str


class EncounterMonster(_ToolArgs):
    """One stat block in a starting encounter."""

    name: str
    count: int = Field(default=1, ge=1)
    hp: int | None = None
    notes: str | None = None


class StartEncounter(_ToolArgs):
    name: str
    monsters: list[EncounterMonster] = Field(min_length=1)


class EndEncounter(_ToolArgs):
    encounter_id: str
    outcome: Literal["victory", "flee", "parley", "tpk", "other"] = "victory"
    summary: str = ""


class MarkBeat(_ToolArgs):
    """Phase 8 — module-state tracking."""

    beat_id: str
    summary: str = ""


class RevealSecret(_ToolArgs):
    """Phase 8 — module-state tracking."""

    secret_id: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# A tool descriptor pairs the argument schema with a one-line human-readable
# description (which the LLM sees in the tool prompt).
class ToolSpec(BaseModel):
    name: str
    description: str
    args_model: type[_ToolArgs]
    # Whether Phase 2 has a working handler. False means the tool is
    # *declared* (LLM sees it) but calling it raises NotImplementedError
    # until its phase implements the handler.
    implemented: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)


TOOLS: dict[str, ToolSpec] = {
    "request_dice_roll": ToolSpec(
        name="request_dice_roll",
        description=(
            "Ask the engine to roll dice. Use for any check, save, attack, or damage roll."
            " Never invent a result; always call this and use what comes back."
        ),
        args_model=RequestDiceRoll,
        implemented=True,
    ),
    "apply_damage": ToolSpec(
        name="apply_damage",
        description=(
            "Reduce a creature's HP. The engine clamps to zero or below and triggers Death"
            " and Dismemberment if appropriate."
        ),
        args_model=ApplyDamage,
        implemented=True,
    ),
    "heal": ToolSpec(
        name="heal",
        description="Restore a creature's HP, capped at hp_max.",
        args_model=Heal,
        implemented=True,
    ),
    "award_xp": ToolSpec(
        name="award_xp",
        description="Award experience points to one or more characters.",
        args_model=AwardXp,
        implemented=False,
    ),
    "award_treasure": ToolSpec(
        name="award_treasure",
        description=(
            "Award gold and/or items to characters. Gold awarded triggers the house-rule XP-for"
            "-treasure conversion at the call site."
        ),
        args_model=AwardTreasure,
        implemented=False,
    ),
    "transition_location": ToolSpec(
        name="transition_location",
        description="Move the party to a different location.",
        args_model=TransitionLocation,
        implemented=True,
    ),
    "spawn_npc": ToolSpec(
        name="spawn_npc",
        description=(
            "Introduce a new NPC into the current scene. Recurring NPCs get a"
            " canonical portrait by default; pass auto_portrait=false for"
            " transient walk-ons."
        ),
        args_model=SpawnNpc,
        implemented=True,
    ),
    "generate_scene_image": ToolSpec(
        name="generate_scene_image",
        description=(
            "Queue a scene illustration. Use sparingly — major locations, climactic beats, first"
            " appearances of significant NPCs. Pass reference_character_id or reference_npc_id"
            " when a known character should appear in the scene; the engine will use the"
            " canonical portrait via Kontext /edit so their identity stays consistent."
        ),
        args_model=GenerateSceneImage,
        implemented=True,
    ),
    "whisper": ToolSpec(
        name="whisper",
        description="Send a private message visible only to one character's player.",
        args_model=Whisper,
        implemented=True,
    ),
    "start_encounter": ToolSpec(
        name="start_encounter",
        description=(
            "Begin a combat encounter. The engine will roll initiative; the LLM should not."
        ),
        args_model=StartEncounter,
        implemented=True,
    ),
    "end_encounter": ToolSpec(
        name="end_encounter",
        description="Close out an active encounter and record its outcome.",
        args_model=EndEncounter,
        implemented=True,
    ),
    "mark_beat": ToolSpec(
        name="mark_beat",
        description="Record that an adventure-module plot beat has been hit.",
        args_model=MarkBeat,
        implemented=False,
    ),
    "reveal_secret": ToolSpec(
        name="reveal_secret",
        description="Record that a module secret has come out in play.",
        args_model=RevealSecret,
        implemented=False,
    ),
}


# ---------------------------------------------------------------------------
# JSON-Schema export for the OpenAI tools parameter
# ---------------------------------------------------------------------------


def _schema_for(args_model: type[_ToolArgs]) -> dict[str, Any]:
    """Return the OpenAI-compatible JSON schema for one tool's args."""

    schema = args_model.model_json_schema()
    # Strip the title pydantic injects; OpenAI tool format doesn't need it
    # and it adds noise to the prompt.
    schema.pop("title", None)
    return schema


def tool_definitions(*, only_implemented: bool = False) -> list[dict[str, Any]]:
    """Build the ``tools`` list passed to ``chat.completions.create(...)``.

    If ``only_implemented`` is True, hide the not-yet-implemented tools so
    the model doesn't try to call them. Defaults to False because hiding
    them would let the LLM "forget" they exist — better to surface and
    have the dispatcher refuse cleanly.
    """

    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": _schema_for(spec.args_model),
            },
        }
        for spec in TOOLS.values()
        if not only_implemented or spec.implemented
    ]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    """What a tool handler returns to the orchestrator. The orchestrator
    feeds ``content`` back to the LLM as the tool message; ``side_effects``
    is a structured record for the audit log and the WS broadcast."""

    content: str = Field(description="LLM-facing summary of what happened.")
    side_effects: dict[str, Any] = Field(default_factory=dict)


HandlerFn = Callable[["AsyncSession", _ToolArgs], Awaitable[ToolResult]]


class ToolHandler(BaseModel):
    """Registered handler for one tool. Bound at orchestrator-import time."""

    name: str
    fn: HandlerFn

    model_config = ConfigDict(arbitrary_types_allowed=True)


_HANDLERS: dict[str, ToolHandler] = {}


def register(name: str) -> Callable[[HandlerFn], HandlerFn]:
    """Decorator: register a function as the handler for one named tool."""

    def decorator(fn: HandlerFn) -> HandlerFn:
        if name not in TOOLS:
            raise KeyError(f"register: unknown tool {name!r}")
        _HANDLERS[name] = ToolHandler(name=name, fn=fn)
        return fn

    return decorator


def get_handler(name: str) -> ToolHandler | None:
    """Look up a registered handler. ``None`` if the tool exists but no
    handler is bound (i.e. a Phase 3/5/8 tool the LLM tried to call early)."""

    return _HANDLERS.get(name)


def parse_tool_args(name: str, raw: dict[str, Any]) -> _ToolArgs:
    """Validate raw JSON args against the named tool's argument schema.

    Raises ``KeyError`` if the tool name is unknown, ``ValidationError``
    if the args are malformed.
    """

    spec = TOOLS.get(name)
    if spec is None:
        raise KeyError(f"unknown tool: {name!r}")
    return spec.args_model.model_validate(raw)


__all__ = [
    "TOOLS",
    "ApplyDamage",
    "AwardTreasure",
    "AwardXp",
    "DiceTarget",
    "EncounterMonster",
    "EndEncounter",
    "GenerateSceneImage",
    "HandlerFn",
    "Heal",
    "MarkBeat",
    "RequestDiceRoll",
    "RevealSecret",
    "SpawnNpc",
    "StartEncounter",
    "ToolHandler",
    "ToolResult",
    "ToolSpec",
    "TransitionLocation",
    "TreasureItem",
    "Whisper",
    "get_handler",
    "parse_tool_args",
    "register",
    "tool_definitions",
]
