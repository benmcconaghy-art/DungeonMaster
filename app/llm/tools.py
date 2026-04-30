"""Tool schemas and dispatcher for LLM-driven state mutations.

The DM never declares mechanical outcomes — it requests them via tool calls.
This module defines the Pydantic schema for each tool (``apply_damage``,
``award_xp``, ``mark_beat``, ``request_dice_roll``, ``generate_scene_image``,
…) and the dispatcher that maps a tool call to the corresponding handler in
``app/orchestrator/handlers/``.

See AGENTS.md invariant #4 — tool calls are the ONLY way state mutates from
LLM output.
"""

from __future__ import annotations
