"""Long-term memory: summarisation, world-fact extraction, NumPy retrieval.

Two responsibilities:

1. Summarisation — periodic LLM calls that compress session and campaign
   history into the ``sessions.summary`` and ``campaigns.long_summary``
   fields. Runs async so it never blocks the turn loop.

2. World-fact retrieval — per-campaign, brute-force cosine similarity over
   L2-normalised embeddings stored as BLOBs on ``world_facts``. Embeddings
   are pre-normalised so cosine reduces to a dot product (AGENTS.md
   invariant #5). Spec §7 describes the routine.

Phase 3 fills this in; Phase 0 only stubs the module.
"""

from __future__ import annotations
