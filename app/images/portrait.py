"""Image-enqueue helpers used by the FastAPI side of the image flow.

Spec §8 "Character & NPC consistency via Kontext" describes the
two-stage flow: a single portrait at creation time becomes the
``canonical_image_id`` on ``characters`` / ``npcs``, and Kontext
``/edit`` requests in later scenes use it as the source so the
character's identity stays stable across sessions.

This module owns:

- :func:`build_portrait_prompt` — turns sparse character/NPC fields
  into a portrait prompt that FLUX can run.
- :func:`enqueue_portrait` — pushes a portrait :class:`ImageJob`
  with the right ``subject_*`` FK so the worker links the resulting
  row back to the character / NPC.
- :func:`enqueue_scene` — pushes a scene illustration job, either a
  plain ``/generate`` request or a Kontext ``/edit`` job that
  references a canonical portrait for character consistency.
- :func:`get_queue_client` — process-wide redis client for the
  FastAPI side, mirroring :func:`app.realtime.pubsub.get_pubsub`.
  The image worker owns its own client (separate process); this
  one belongs to the web app.

The portrait kind is always ``"npc"`` per spec §8 "Generation
parameters per kind" — 768x1024 with 32 steps. PCs and NPCs use the
same parameters because they're both single-figure portraits; the
distinction is only that PCs link via ``characters.canonical_image_id``
and NPCs via ``npcs.canonical_image_id``.
"""

from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as redis
from uuid_extensions import uuid7

from app.images.queue import ImageJob, open_queue_client, push_job

log = logging.getLogger(__name__)


def build_portrait_prompt(
    *,
    name: str,
    race: str | None = None,
    class_name: str | None = None,
    alignment: str | None = None,
    description: str | None = None,
) -> str:
    """Compose a portrait prompt from the fields a character or NPC
    typically carries.

    Order is a fixed template so the same character produces the same
    prompt across calls (the dedup hash depends on it). All fields are
    optional except ``name`` — NPCs from ``spawn_npc`` may have only
    name + description; PCs from chargen always have race/class/alignment.

    The output is a single line so the campaign-level ``image_style``
    suffix the worker appends (per spec §8 "Style consistency") reads
    cleanly after it.
    """

    parts: list[str] = [f"Portrait of {name.strip()}"]
    descriptor: list[str] = []
    if race:
        descriptor.append(race.strip())
    if class_name:
        descriptor.append(class_name.strip())
    if descriptor:
        parts.append(f", a {' '.join(descriptor)}")
    if alignment:
        parts.append(f", {alignment.strip()} alignment")
    if description:
        # Strip and trim trailing punctuation so we don't end up with
        # ".. .." after appending the style suffix downstream.
        cleaned = description.strip().rstrip(".")
        if cleaned:
            parts.append(f". {cleaned}")
    return "".join(parts)


async def enqueue_portrait(
    queue_client: redis.Redis,
    *,
    campaign_id: str,
    prompt: str,
    session_id: str | None = None,
    subject_character_id: str | None = None,
    subject_npc_id: str | None = None,
) -> str:
    """Push a portrait job onto ``images:queue`` and return the
    pre-allocated image id.

    Exactly one of ``subject_character_id`` / ``subject_npc_id`` should
    be set so the worker writes the right FK back. Both unset is
    accepted (the row still persists, just with no canonical link) but
    both set raises :class:`ValueError` — there's no sensible meaning
    for "this image is canonical for both a PC and an NPC".

    The id is a UUIDv7 the caller can immediately stash in
    ``characters.canonical_image_id`` / ``npcs.canonical_image_id`` if
    desired, or hand to the WS layer for an ``image_pending``
    placeholder before the worker finishes.
    """

    if subject_character_id is not None and subject_npc_id is not None:
        raise ValueError(
            "enqueue_portrait: only one of subject_character_id or "
            "subject_npc_id may be set"
        )

    image_id = str(uuid7())
    job = ImageJob(
        id=image_id,
        campaign_id=campaign_id,
        session_id=session_id,
        kind="npc",  # spec §8: portraits use the npc parameter set
        prompt=prompt,
        subject_character_id=subject_character_id,
        subject_npc_id=subject_npc_id,
    )
    await push_job(queue_client, job)
    log.info(
        "portrait enqueued: image_id=%s campaign=%s subject_char=%s subject_npc=%s",
        image_id,
        campaign_id,
        subject_character_id,
        subject_npc_id,
    )
    return image_id


async def enqueue_scene(
    queue_client: redis.Redis,
    *,
    campaign_id: str,
    prompt: str,
    kind: str = "scene",
    session_id: str | None = None,
    reference_image_id: str | None = None,
    edit_instruction: str | None = None,
) -> str:
    """Push a scene-illustration job onto ``images:queue``.

    Two modes:

    - ``reference_image_id`` unset → the worker dispatches to FLUX
      ``/generate`` with ``prompt`` as the txt2img prompt.
    - ``reference_image_id`` set together with ``edit_instruction`` →
      the worker dispatches to Kontext ``/edit`` with the referenced
      image as the source. This is the spec §8 "Contextual edits"
      flow that preserves character identity across scenes.

    Returns the pre-allocated image id. The reference / instruction
    invariant is checked here so a misuse fails fast at the enqueuer
    rather than 60 seconds later when the worker hits the
    ``invalid_job`` failure path.
    """

    if reference_image_id is not None and edit_instruction is None:
        raise ValueError(
            "enqueue_scene: reference_image_id requires edit_instruction"
        )
    if edit_instruction is not None and reference_image_id is None:
        raise ValueError(
            "enqueue_scene: edit_instruction requires reference_image_id"
        )

    image_id = str(uuid7())
    job = ImageJob(
        id=image_id,
        campaign_id=campaign_id,
        session_id=session_id,
        kind=kind,
        prompt=prompt,
        reference_image_id=reference_image_id,
        edit_instruction=edit_instruction,
    )
    await push_job(queue_client, job)
    log.info(
        "scene enqueued: image_id=%s campaign=%s kind=%s reference=%s",
        image_id,
        campaign_id,
        kind,
        reference_image_id,
    )
    return image_id


# ---------------------------------------------------------------------------
# Process-wide queue client singleton (FastAPI side)
# ---------------------------------------------------------------------------


_singleton: redis.Redis | None = None


def get_queue_client() -> redis.Redis:
    """Return the process-wide queue client, building on first call.

    Mirrors :func:`app.realtime.pubsub.get_pubsub`. The lifespan hook
    in :mod:`app.main` should ``aclose()`` it on shutdown so gunicorn
    workers shut down cleanly under systemd.
    """

    global _singleton
    if _singleton is None:
        _singleton = open_queue_client()
    return _singleton


def set_queue_client_for_tests(instance: Any) -> None:
    """Replace the singleton with a test double (in-memory fake or a
    pre-built real client pointing at an ephemeral Valkey)."""

    global _singleton
    _singleton = instance


async def reset_for_tests() -> None:
    """Dispose the singleton if any. Subsequent :func:`get_queue_client`
    calls build fresh — used by tests that need a clean transport."""

    global _singleton
    if _singleton is not None:
        try:
            await _singleton.aclose()
        except Exception:
            log.debug("queue client reset: aclose failed", exc_info=True)
        _singleton = None


__all__ = [
    "build_portrait_prompt",
    "enqueue_portrait",
    "enqueue_scene",
    "get_queue_client",
    "reset_for_tests",
    "set_queue_client_for_tests",
]
