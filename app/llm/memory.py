"""Long-term memory: summarisation, world-fact extraction, NumPy retrieval.

Two responsibilities (per spec §7):

1. Summarisation — periodic LLM calls that compress session and campaign
   history into the ``sessions.summary`` and ``campaigns.long_summary``
   fields. Runs async so it never blocks the turn loop.

2. World-fact retrieval — per-campaign, brute-force cosine similarity
   over L2-normalised embeddings stored as BLOBs on ``world_facts``.
   Embeddings are pre-normalised so cosine reduces to a dot product
   (AGENTS.md invariant #5).

Phase 2 surface — only the verbatim memory tier is implemented here.
The summarisation and vector-retrieval pieces land in Phase 3.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SessionMessage


async def recent_turns(
    db: AsyncSession,
    session_id: str,
    n: int = 20,
) -> list[SessionMessage]:
    """Return the last ``n`` ``SessionMessage`` rows in chronological order.

    Used by :func:`app.llm.prompts.build_dm_prompt` for the
    ``[RECENT TURNS]`` block. The ordering is "ascending by created_at"
    so the LLM sees the conversation in natural reading order; the
    ``LIMIT`` is applied to the descending tail and then reversed in
    Python.

    Phase 2 returns *every* message regardless of audience — Phase 5+
    multi-player support will filter whispers per-character. For the
    single-player Phase 2 loop the player sees everything the DM has
    said anyway.
    """

    if n <= 0:
        return []

    stmt = (
        select(SessionMessage)
        .where(SessionMessage.session_id == session_id)
        .order_by(SessionMessage.created_at.desc())
        .limit(n)
    )
    result = await db.scalars(stmt)
    rows = list(result.all())
    rows.reverse()
    return rows


__all__ = ["recent_turns"]
