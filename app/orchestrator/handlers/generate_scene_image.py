"""Handler for the ``generate_scene_image`` tool.

Routes the LLM's request between two FLUX endpoints depending on
whether a recurring character should appear in the scene:

- No reference set → plain ``/generate`` (txt2img).
- ``reference_character_id`` → look up the PC's
  ``canonical_image_id`` and dispatch via Kontext ``/edit`` so the
  character's face/build/gear stay consistent across appearances.
- ``reference_npc_id`` → same for NPCs (and module-defined NPCs that
  carry a ``canonical_image_id`` from module import).

Spec §8 "Character & NPC consistency via Kontext" item 2:
"instead of /generate from scratch, the worker calls /edit with the
canonical portrait as the source". The handler is the gate — the
worker just dispatches whichever flavour landed on the queue.

Failure handling:

- Unknown / cross-campaign character / NPC → structured error,
  no enqueue. The LLM gets a clean tool-message back so it can
  retry without the bogus reference.
- Subject exists but lacks a canonical portrait → fall back to
  plain ``/generate`` and log. This is graceful degradation: the
  scene still renders, just without identity preservation. A
  separate request can portrait the subject afterwards.
- Both reference fields set → reject as ``invalid_args`` rather
  than guess which one the LLM meant.
- Queue push failure → return an error tool result. Unlike
  spawn_npc, there's no DB row at risk here, so re-raising would
  also be acceptable; we keep the failure path consistent across
  image-enqueue handlers.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Character, Npc
from app.db.models import Session as DmSession
from app.images.portrait import enqueue_scene, get_queue_client
from app.llm.tools import GenerateSceneImage, ToolResult, register
from app.orchestrator.context import current_context

log = logging.getLogger(__name__)


@register("generate_scene_image")  # type: ignore[arg-type]
async def handle(db: AsyncSession, args: GenerateSceneImage) -> ToolResult:
    """Resolve the reference subject (if any), then enqueue."""

    if args.reference_character_id is not None and args.reference_npc_id is not None:
        return ToolResult(
            content=(
                "generate_scene_image failed: pass at most one of"
                " reference_character_id or reference_npc_id, not both."
            ),
            side_effects={"kind": "error", "reason": "invalid_args"},
        )

    ctx = current_context()
    session = await db.get(DmSession, ctx.session_id)
    if session is None:
        return ToolResult(
            content="generate_scene_image failed: active session not found.",
            side_effects={"kind": "error", "reason": "unknown_session"},
        )
    campaign_id = session.campaign_id

    reference_image_id: str | None = None
    edit_instruction: str | None = None
    used_reference_kind: str | None = None
    used_reference_id: str | None = None

    if args.reference_character_id is not None:
        character = await db.get(Character, args.reference_character_id)
        if character is None or character.campaign_id != campaign_id:
            return ToolResult(
                content=(
                    f"generate_scene_image failed: character"
                    f" {args.reference_character_id!r} not found in this campaign."
                ),
                side_effects={"kind": "error", "reason": "unknown_reference"},
            )
        if character.canonical_image_id is None:
            log.info(
                "generate_scene_image: character %s has no canonical portrait;"
                " falling back to /generate",
                character.id,
            )
        else:
            reference_image_id = character.canonical_image_id
            edit_instruction = args.prompt
            used_reference_kind = "character"
            used_reference_id = character.id
    elif args.reference_npc_id is not None:
        npc = await db.get(Npc, args.reference_npc_id)
        if npc is None or npc.campaign_id != campaign_id:
            return ToolResult(
                content=(
                    f"generate_scene_image failed: npc"
                    f" {args.reference_npc_id!r} not found in this campaign."
                ),
                side_effects={"kind": "error", "reason": "unknown_reference"},
            )
        if npc.canonical_image_id is None:
            log.info(
                "generate_scene_image: npc %s has no canonical portrait;"
                " falling back to /generate",
                npc.id,
            )
        else:
            reference_image_id = npc.canonical_image_id
            edit_instruction = args.prompt
            used_reference_kind = "npc"
            used_reference_id = npc.id

    try:
        image_id = await enqueue_scene(
            get_queue_client(),
            campaign_id=campaign_id,
            prompt=args.prompt,
            kind=args.kind,
            session_id=ctx.session_id,
            reference_image_id=reference_image_id,
            edit_instruction=edit_instruction,
        )
    except Exception:
        log.exception("generate_scene_image: queue push failed")
        return ToolResult(
            content="generate_scene_image failed: image queue unavailable.",
            side_effects={"kind": "error", "reason": "queue_unavailable"},
        )

    mode = "edit" if reference_image_id is not None else "generate"
    summary = (
        f"Scene image queued (image_id={image_id}, kind={args.kind}, mode={mode})."
        if used_reference_kind is None
        else (
            f"Scene image queued via Kontext /edit using {used_reference_kind}"
            f" {used_reference_id} canonical portrait (image_id={image_id})."
        )
    )

    side_effects: dict[str, object] = {
        "kind": "image_queued",
        "image_id": image_id,
        "image_kind": args.kind,
        "mode": mode,
    }
    if used_reference_kind is not None:
        side_effects["reference_kind"] = used_reference_kind
        side_effects["reference_id"] = used_reference_id

    return ToolResult(content=summary, side_effects=side_effects)


__all__ = ["handle"]
