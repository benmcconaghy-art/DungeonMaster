"""ModuleContent schema and helpers for Phase 8 adventure modules.

Per spec §10, a module is a single-file JSON document containing the
complete structure of an adventure: synopsis, tone, locations, NPCs,
encounters, plot beats, secrets, endings, and world facts.

Symbolic IDs (snake_case, namespaced: loc_, npc_, enc_, beat_, sec_, end_)
live in the module JSON. The loader mints fresh UUIDv7 per symbol and writes
the mapping to campaigns.module_state.symbolic_id_map for runtime resolution.

Spoiler discipline: player API serialisers must NEVER expose module.content.secrets
or plot_beats[i].dm_notes. Only the DM system prompt receives them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LocationContent(BaseModel):
    """A location within the module."""

    symbol: str = Field(description="Snake_case namespaced ID, e.g. 'loc_gatehouse'")
    name: str = Field(description="Human-readable location name")
    description: str = Field(description="Narrative description of the location")
    parent_symbol: str | None = Field(
        default=None,
        description="Symbol of the parent location, if this location is nested")
    image_role: str | None = Field(
        default=None,
        description="Role for image generation, e.g. 'scene:gatehouse_exterior'")
    metadata: dict[str, Any] = Field(default_factory=dict)


class NpcStats(BaseModel):
    """Loose creature stats for an NPC. Matches the NPC.stats shape in the DB."""

    model_config = ConfigDict(extra="allow")


class NpcContent(BaseModel):
    """An NPC within the module."""

    symbol: str = Field(description="Snake_case namespaced ID, e.g. 'npc_castellan'")
    name: str = Field(description="Human-readable NPC name")
    description: str = Field(description="Narrative description")
    motivation: str = Field(description="What drives this NPC")
    starting_location_symbol: str = Field(
        description="Symbol of the location where this NPC is first found")
    stats: dict[str, Any] = Field(default_factory=dict, description="Creature stats")
    sample_dialogue: str | None = Field(
        default=None,
        description="Example quotes or speech patterns")
    image_role: str | None = Field(
        default=None,
        description="Role for image generation, e.g. 'npc:castellan_thorvald'")
    secrets: list[str] | None = Field(
        default=None,
        description="List of secret symbols this NPC knows or reveals")


class EncounterMonster(BaseModel):
    """One monster in an encounter."""

    name: str = Field(description="Monster name")
    count: int = Field(default=1, ge=1, description="Number of this monster")
    tactics: str | None = Field(default=None, description="How this monster fights")


class EncounterContent(BaseModel):
    """An encounter within the module."""

    symbol: str = Field(description="Snake_case namespaced ID, e.g. 'enc_goblin_ambush'")
    name: str = Field(description="Human-readable encounter name")
    trigger_hint: str = Field(
        description="Natural language hint for when this encounter triggers")
    monsters: list[EncounterMonster] = Field(default_factory=list)
    treasure_hint: str | None = Field(default=None, description="Hint about treasure")


class PlotBeat(BaseModel):
    """A plot beat (story milestone) within the module."""

    symbol: str = Field(description="Snake_case namespaced ID, e.g. 'beat_arrival_briefing'")
    title: str = Field(description="Human-readable beat title")
    trigger_hint: str = Field(
        description="Natural language hint for when this beat should be marked")
    outcome: str = Field(description="What happens when this beat fires")
    leads_to: str | None = Field(
        default=None,
        description="Symbol of the next beat this leads to, if any")
    dm_notes: str | None = Field(
        default=None,
        description="DM-only notes about this beat — NEVER shown to players")


class Secret(BaseModel):
    """A secret that can be revealed during play."""

    symbol: str = Field(description="Snake_case namespaced ID, e.g. 'sec_vance_curse'")
    content: str = Field(description="The secret content")
    reveal_when: str = Field(
        description="Natural language hint for when this secret should be revealed")
    leads_to_beat: str | None = Field(
        default=None,
        description="Symbol of the beat this secret leads to, if any")


class Ending(BaseModel):
    """A possible ending to the module."""

    symbol: str = Field(description="Snake_case namespaced ID, e.g. 'end_clean_victory'")
    trigger: str = Field(description="How this ending is triggered")
    outcome: str = Field(description="What happens in this ending")


class WorldFact(BaseModel):
    """A world fact for long-term memory."""

    fact: str = Field(description="The fact text")
    tags: list[str] = Field(default_factory=list, description="Tags for retrieval")
    importance: int = Field(default=5, ge=1, le=10, description="Importance score 1-10")


class ModuleContent(BaseModel):
    """The complete schema for an adventure module.

    Per spec §10, this is the canonical shape for a module JSON file.
    """

    format_version: str = Field(default="1.0", description="Module format version")
    synopsis: str = Field(description="Brief summary of the module")
    tone: str = Field(description="Tone description, e.g. 'gritty', 'heroic'")
    image_style: str | None = Field(
        default=None,
        description="FLUX style prompt for module images, e.g. 'dark fantasy, detailed'")
    image_negative_prompt: str | None = Field(
        default=None,
        description="FLUX negative prompt, e.g. 'modern objects, photographic'")
    level_range: list[int] = Field(
        default_factory=lambda: [1, 3],
        description="Recommended character level range [min, max]")
    estimated_sessions: int = Field(
        default=6,
        ge=1,
        description="Estimated number of sessions to complete")
    starting_hook: str = Field(description="The opening hook to draw players in")
    starting_location_symbol: str = Field(
        description="Symbol of the starting location")
    locations: list[LocationContent] = Field(default_factory=list)
    npcs: list[NpcContent] = Field(default_factory=list)
    encounters: list[EncounterContent] = Field(default_factory=list)
    plot_beats: list[PlotBeat] = Field(default_factory=list)
    secrets: list[Secret] = Field(default_factory=list)
    endings: list[Ending] = Field(default_factory=list)
    world_facts: list[WorldFact] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "EncounterContent",
    "EncounterMonster",
    "Ending",
    "LocationContent",
    "ModuleContent",
    "NpcContent",
    "NpcStats",
    "PlotBeat",
    "Secret",
    "WorldFact",
]
