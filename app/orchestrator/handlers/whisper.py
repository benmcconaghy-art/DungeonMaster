"""Handler for the ``whisper`` tool.

Persists a private message addressed to one character. The SSE bridge
filters by ``audience`` so other players don't see it; the DM
prompt continues to surface every message in history regardless of
audience, so the DM stays consistent with what it whispered.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SessionMessage
from app.llm.tools import ToolResult, Whisper, register
from app.orchestrator.context import current_context


@register("whisper")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: Whisper) -> ToolResult:
    """Append a whisper as a session_messages row with audience set."""

    ctx = current_context()
    msg = SessionMessage(
        session_id=ctx.session_id,
        sender_kind="dm",
        sender_id=None,
        audience=[args.character_id],
        content=args.content,
    )
    db.add(msg)
    await db.flush()

    summary = f"Whispered privately to character {args.character_id}."
    return ToolResult(
        content=summary,
        side_effects={
            "kind": "whisper",
            "message_id": msg.id,
            "audience": [args.character_id],
            "content": args.content,
        },
    )


__all__ = ["handle"]
