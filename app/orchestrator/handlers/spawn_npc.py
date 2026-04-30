"""Handler for the ``spawn_npc`` tool.

Introduces a new NPC into the active session's campaign. Spec §8
"Character & NPC consistency via Kontext": when a recurring NPC is
spawned, the worker generates a canonical portrait once and links it
via ``npcs.canonical_image_id``; later scene edits use that portrait
as the Kontext source so the NPC stays visually consistent across
appearances.

The ``auto_portrait`` flag on :class:`SpawnNpc` lets the LLM opt out
of the portrait for transient NPCs ("a goblin scout", "the stable
hand who takes their horses") — generating an image for every
walk-on costs 17s of FLUX time and clutters the cache. The default
is ``True`` because the modal NPC the LLM names is a recurring one.

Side effect taxonomy:

- ``kind=npc_spawned`` always — the row landed.
- ``portrait_image_id`` is set when ``auto_portrait`` was true; the
  image worker links it back to ``npcs.canonical_image_id`` when the
  job finishes. The frontend can render an ``image_pending`` card on
  receipt.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Npc
from app.db.models import Session as DmSession
from app.images.portrait import build_portrait_prompt, enqueue_portrait, get_queue_client
from app.llm.tools import SpawnNpc, ToolResult, register
from app.orchestrator.context import current_context

log = logging.getLogger(__name__)


@register("spawn_npc")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: SpawnNpc) -> ToolResult:
    """Persist the NPC, optionally enqueue a canonical portrait."""

    ctx = current_context()
    session_row = await db.get(DmSession, ctx.session_id)
    if session_row is None:
        return ToolResult(
            content="spawn_npc failed: active session not found.",
            side_effects={"kind": "error", "reason": "unknown_session"},
        )
    campaign_id = session_row.campaign_id

    npc = Npc(
        campaign_id=campaign_id,
        name=args.name,
        description=args.description or None,
        stats=args.stats,
    )
    db.add(npc)
    await db.flush()

    portrait_image_id: str | None = None
    if args.auto_portrait:
        prompt = build_portrait_prompt(name=args.name, description=args.description or None)
        try:
            portrait_image_id = await enqueue_portrait(
                get_queue_client(),
                campaign_id=campaign_id,
                prompt=prompt,
                session_id=ctx.session_id,
                subject_npc_id=npc.id,
            )
        except Exception:
            # Queue push failure (Valkey unreachable, transport error)
            # must not roll back the NPC row — the LLM has already
            # narrated the introduction. Log loudly and continue
            # without a portrait; an operator can retry via the API
            # endpoint after the fix.
            log.exception(
                "spawn_npc: portrait enqueue failed for npc %s; npc row kept",
                npc.id,
            )

    summary_parts = [f"NPC '{args.name}' spawned (id={npc.id})."]
    if portrait_image_id is not None:
        summary_parts.append(f"Canonical portrait queued (image_id={portrait_image_id}).")
    summary = " ".join(summary_parts)

    side_effects: dict[str, object] = {
        "kind": "npc_spawned",
        "npc_id": npc.id,
        "name": args.name,
    }
    if portrait_image_id is not None:
        side_effects["portrait_image_id"] = portrait_image_id

    return ToolResult(content=summary, side_effects=side_effects)


__all__ = ["handle"]
