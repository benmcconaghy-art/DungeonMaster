"""The DM turn loop.

Receives a player action, builds the layered prompt (``app/llm/prompts.py``),
streams the LLM response, parses tool calls, dispatches each through the
appropriate handler in ``app/orchestrator/handlers/``, persists the
narration + tool results atomically, and publishes the resulting events to
the session WebSocket hub.

Critical invariant (AGENTS.md #2): never hold a write transaction across
the streaming call. Persist input, release, stream, reopen for the
completion write.

Phase 2 implements this; Phase 0 ships only the placeholder.
"""

from __future__ import annotations
