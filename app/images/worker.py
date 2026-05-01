"""Image worker — async queue consumer.

Single-concurrency worker that pops image jobs off Valkey
(``images:queue``), hashes the request to dedupe against
``generated_images.prompt_hash``, calls the FLUX client (``/generate``
for txt2img, ``/edit`` for Kontext character / scene edits), persists
the resulting PNG under ``/var/lib/dungeon-master/images/<id>.png`` and
a row in ``generated_images``, then publishes ``image_ready`` on the
session pub/sub channel so the WS hub can broadcast it.

Runs as its own systemd unit (``dungeon-master-imageworker.service``).
``python -m app.images.worker`` is the entry point — see the
``if __name__ == "__main__"`` block at the bottom.

Concurrency: spec §8 has the FLUX service serialising with its own
``asyncio.Lock``, so worker concurrency >1 wouldn't help. We run a
single async loop alongside a parallel watchdog task that runs a
256x256/1-step ``/generate`` probe every 30s to maintain the
``image:status`` Valkey key — see :func:`_watchdog` for why a
``/health`` poll alone isn't sufficient.

Failure handling per spec §8:
- FLUX 503 retry exhaustion → emit ``image_failed`` with
  ``reason="flux_unavailable"``. The placeholder card on the client
  switches to "(scene image unavailable)". The DM's narration is not
  blocked.
- Decode / write / DB errors → emit ``image_failed`` with a specific
  reason; log loudly. We don't re-enqueue: a fresh user action will
  cause the orchestrator to request the image again, and the dedup
  cache short-circuits if it was already generated.
- Unparseable queue payload → log + skip. A poison message must not
  wedge the queue.

Transaction discipline (AGENTS.md invariant #1): the
``generated_images`` row + any FK update on characters/npcs commit
before the ``image_ready`` publish. Subscribers that read the DB
after seeing the message find the row.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app import metrics
from app.config import get_settings
from app.db import models
from app.db.session import SessionLocal
from app.images.client import FluxClient, FluxClientError, get_flux_client
from app.images.health import (
    DEGRADED_THRESHOLD_S,
    POLL_INTERVAL_S,
    write_status,
)
from app.images.queue import (
    ImageJob,
    ImageJobDecodeError,
    open_queue_client,
    pop_job,
)
from app.realtime.messages import ImageFailed, ImageReady
from app.realtime.pubsub import Pubsub, get_pubsub

log = logging.getLogger(__name__)


# Per-kind FLUX parameters from spec §8 "Generation parameters per kind".
# A regression that drops one of these keys would leave that kind
# defaulting silently to scene parameters; the ImageKind literal in
# queue.py is the matching pin.
_KIND_PARAMS: dict[str, dict[str, Any]] = {
    "scene": {"width": 1280, "height": 768, "steps": 28, "guidance": 3.5},
    "npc": {"width": 768, "height": 1024, "steps": 32, "guidance": 3.5},
    "item": {"width": 1024, "height": 1024, "steps": 24, "guidance": 3.5},
    "map": {"width": 1280, "height": 1280, "steps": 36, "guidance": 4.0},
}

# Loop timeouts. ``_POP_TIMEOUT_S`` is the BLPOP wake interval — gives
# the loop a chance to react to cancellation between blocking pops.
_POP_TIMEOUT_S = 5.0


def _now_iso() -> str:
    """ISO-8601 UTC timestamp with millisecond precision, matching the
    schema's ``strftime`` server defaults."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _compose_prompt(prompt: str, style_suffix: str | None) -> str:
    """Style suffix is appended (not prepended) so the user's prompt
    leads. Spec §8 "Style consistency": the campaign-level
    ``image_style`` like "dark fantasy oil painting, muted palette,
    candlelight, painterly brushwork" is appended to every prompt."""

    if style_suffix:
        return f"{prompt}\n\nStyle: {style_suffix}"
    return prompt


def _hash_inputs(
    *,
    campaign_id: str,
    kind: str,
    prompt_with_style: str,
    reference_image_id: str | None,
) -> str:
    """SHA-256 hex of the dedup key. Spec §8: hash inputs are
    ``(campaign_id, kind, prompt + style_suffix, [reference_image_id])``.

    Stable across processes (no Python ``hash()`` randomisation) so a
    web-app enqueuer that pre-computes the same hash to skip pushing
    duplicate jobs would match what the worker computes after popping.
    Phase 5 doesn't pre-compute on the enqueue side; this is here as
    the canonical function in case a future caller wants to.
    """

    h = hashlib.sha256()
    h.update(campaign_id.encode("utf-8"))
    h.update(b"\0")
    h.update(kind.encode("utf-8"))
    h.update(b"\0")
    h.update(prompt_with_style.encode("utf-8"))
    h.update(b"\0")
    if reference_image_id is not None:
        h.update(reference_image_id.encode("utf-8"))
    return h.hexdigest()


async def _publish(pubsub: Pubsub, session_id: str | None, message: Any) -> None:
    """Best-effort publish. ``session_id is None`` means a portrait
    job made outside an active session — there's nothing to broadcast
    to, so we just skip and log."""

    if session_id is None:
        log.debug("image job had no session_id; skipping broadcast")
        return
    try:
        await pubsub.publish(session_id, message)
    except Exception:
        # Pubsub failures should not crash the worker — the DB row is
        # still valid and a reconnecting client will see it via the
        # snapshot path. Logging at warning so a misconfigured Valkey
        # is visible without becoming a stop-the-world.
        log.warning("image worker: publish failed for session %s", session_id, exc_info=True)


async def _load_campaign_style(db: AsyncSession, campaign_id: str) -> tuple[str | None, str | None]:
    """Return ``(image_style, image_negative_prompt)`` for the
    campaign. Both are nullable; the worker treats null and empty as
    equivalent."""

    campaign = await db.get(models.Campaign, campaign_id)
    if campaign is None:
        return None, None
    return campaign.image_style, campaign.image_negative_prompt


async def _find_existing_by_hash(
    db: AsyncSession, prompt_hash: str
) -> models.GeneratedImage | None:
    """Cache lookup. Spec §8: hash-based dedup is load-bearing —
    repeat scenes cost 0 seconds. The unique constraint on
    ``generated_images.prompt_hash`` makes this index-backed."""

    return (
        await db.scalars(
            select(models.GeneratedImage).where(models.GeneratedImage.prompt_hash == prompt_hash)
        )
    ).first()


async def _read_reference_png(db: AsyncSession, reference_image_id: str) -> bytes:
    """Load the canonical portrait bytes that the Kontext ``/edit``
    call needs as ``source_png``. Raises ``ValueError`` /
    ``FileNotFoundError`` if the row is missing or its file is gone —
    the caller turns either into an ``image_failed`` event."""

    row = await db.get(models.GeneratedImage, reference_image_id)
    if row is None:
        raise ValueError(f"reference image not found: {reference_image_id}")
    return Path(row.file_path).read_bytes()


async def _persist_and_link(
    *,
    factory: async_sessionmaker[AsyncSession],
    job: ImageJob,
    final_prompt: str,
    prompt_hash: str,
    file_path: Path,
    width: int | None,
    height: int | None,
) -> None:
    """Single transaction: insert the ``generated_images`` row, then
    update ``characters.canonical_image_id`` or ``npcs.canonical_image_id``
    if the job names a subject. Commits before the broadcast so any
    subscriber who reads the DB on ``image_ready`` finds the row."""

    async with factory() as db, db.begin():
        row = models.GeneratedImage(
            id=job.id,
            campaign_id=job.campaign_id,
            kind=job.kind,
            prompt=final_prompt,
            prompt_hash=prompt_hash,
            file_path=str(file_path),
            session_id=job.session_id,
            width=width,
            height=height,
            source_image_id=job.reference_image_id,
            edit_instruction=job.edit_instruction,
        )
        db.add(row)
        await db.flush()

        if job.subject_character_id is not None:
            character = await db.get(models.Character, job.subject_character_id)
            if character is not None:
                character.canonical_image_id = job.id
            else:
                log.warning(
                    "image worker: subject_character_id=%s missing; skipping FK link",
                    job.subject_character_id,
                )
        if job.subject_npc_id is not None:
            npc = await db.get(models.Npc, job.subject_npc_id)
            if npc is not None:
                npc.canonical_image_id = job.id
            else:
                log.warning(
                    "image worker: subject_npc_id=%s missing; skipping FK link",
                    job.subject_npc_id,
                )


async def _process_job(
    job: ImageJob,
    *,
    flux: FluxClient,
    pubsub: Pubsub,
    factory: async_sessionmaker[AsyncSession],
    storage_dir: Path,
) -> None:
    """Run one job end-to-end. Catches all expected failure modes and
    emits the right ``image_failed`` flavour rather than letting the
    main loop see exceptions.

    Phase 7 metrics: every terminal path (ok / failed-of-various-kinds)
    increments ``image_jobs_total{kind, outcome}``. The OK path also
    observes wall-clock duration on ``image_job_duration_seconds{kind}``
    so an operator can spot FLUX latency regressions over time.
    """

    started = time.monotonic()

    def _failed(reason: str) -> None:
        metrics.image_jobs_total.labels(kind=job.kind, outcome="failed").inc()

    if job.reference_image_id is not None and job.edit_instruction is None:
        log.error(
            "image worker: job %s has reference_image_id but no edit_instruction",
            job.id,
        )
        _failed("invalid_job")
        await _publish(pubsub, job.session_id, ImageFailed(image_id=job.id, reason="invalid_job"))
        return

    async with factory() as db:
        style_suffix, negative_prompt = await _load_campaign_style(db, job.campaign_id)
    final_prompt = _compose_prompt(job.prompt, style_suffix)
    prompt_hash = _hash_inputs(
        campaign_id=job.campaign_id,
        kind=job.kind,
        prompt_with_style=final_prompt,
        reference_image_id=job.reference_image_id,
    )

    async with factory() as db:
        existing = await _find_existing_by_hash(db, prompt_hash)
    if existing is not None:
        log.info(
            "image worker: dedup hit for job %s -> existing row %s",
            job.id,
            existing.id,
        )
        # Dedup hits skip FLUX entirely — count them as "ok" outcomes
        # because the player gets the image they asked for, but don't
        # observe a duration histogram value (no real generation
        # happened, so timing it would skew the latency view).
        metrics.image_jobs_total.labels(kind=job.kind, outcome="ok").inc()
        await _publish(
            pubsub,
            job.session_id,
            ImageReady(image_id=existing.id, url=_image_url(existing.id)),
        )
        return

    source_png: bytes | None = None
    if job.reference_image_id is not None:
        try:
            async with factory() as db:
                source_png = await _read_reference_png(db, job.reference_image_id)
        except (ValueError, FileNotFoundError, OSError) as exc:
            log.warning("image worker: source image fetch failed for job %s: %s", job.id, exc)
            _failed("missing_reference")
            await _publish(
                pubsub,
                job.session_id,
                ImageFailed(image_id=job.id, reason="missing_reference"),
            )
            return

    try:
        if source_png is not None:
            assert job.edit_instruction is not None
            png_bytes, _seed = await flux.edit(job.edit_instruction, source_png)
            width: int | None = None
            height: int | None = None
        else:
            params = _KIND_PARAMS[job.kind]
            png_bytes, _seed = await flux.generate(
                final_prompt,
                negative_prompt=negative_prompt or "",
                width=params["width"],
                height=params["height"],
                steps=params["steps"],
                guidance=params["guidance"],
            )
            width = params["width"]
            height = params["height"]
    except FluxClientError as exc:
        log.warning("image worker: FLUX call failed for job %s: %s", job.id, exc)
        _failed("flux_unavailable")
        await _publish(
            pubsub,
            job.session_id,
            ImageFailed(image_id=job.id, reason="flux_unavailable"),
        )
        return

    file_path = storage_dir / f"{job.id}.png"
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(png_bytes)
    except OSError:
        log.exception("image worker: failed to write %s", file_path)
        _failed("write_failed")
        await _publish(pubsub, job.session_id, ImageFailed(image_id=job.id, reason="write_failed"))
        return

    try:
        await _persist_and_link(
            factory=factory,
            job=job,
            final_prompt=final_prompt,
            prompt_hash=prompt_hash,
            file_path=file_path,
            width=width,
            height=height,
        )
    except Exception:
        log.exception("image worker: DB persist failed for job %s", job.id)
        # File is on disk but no row points at it — leave it; a future
        # cleanup sweep can drop orphans.
        _failed("db_failed")
        await _publish(pubsub, job.session_id, ImageFailed(image_id=job.id, reason="db_failed"))
        return

    metrics.image_jobs_total.labels(kind=job.kind, outcome="ok").inc()
    metrics.image_job_duration_seconds.labels(kind=job.kind).observe(time.monotonic() - started)
    log.info("image worker: completed job %s (kind=%s)", job.id, job.kind)
    await _publish(
        pubsub,
        job.session_id,
        ImageReady(image_id=job.id, url=_image_url(job.id)),
    )


def _image_url(image_id: str) -> str:
    """Internal URL pattern the FastAPI authoriser hands off to nginx
    via X-Accel-Redirect (spec §8 "Storage & serving"). The exact route
    lands in Step 9 with the frontend; the worker only needs to put a
    consistent string in the wire message."""

    return f"/api/images/{image_id}.png"


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


async def _watchdog(flux: FluxClient, queue_client: Any) -> None:
    """Poll FLUX with a deep ``/generate`` probe every
    ``POLL_INTERVAL_S`` seconds; flip ``image:status`` to ``degraded``
    after ``DEGRADED_THRESHOLD_S`` of consecutive failures. Restored to
    ``ok`` on the first successful probe after a degraded period.
    Runs until cancelled by the main task.

    Why probe instead of /health: in production the FLUX service
    has been observed reporting 200 OK from /health while /generate
    returns 500 (a 7GB squatter on the GPU pushed FLUX over VRAM, with
    no signal on /health). The probe runs a 256x256/1-step
    generation — cheap to FLUX, definitive about whether image
    generation is working. Any 5xx, transport error, or malformed
    response counts as a failure tick: this is the "treat sustained
    5xx as degraded" widening, beyond the spec's narrower 503-only
    note (which only governs the client's automatic retry).
    """

    last_success = asyncio.get_running_loop().time()
    current_status = "ok"
    await write_status(queue_client, "ok", since_iso=_now_iso())
    while True:
        try:
            await flux.probe()
            now = asyncio.get_running_loop().time()
            last_success = now
            metrics.flux_health_probe_total.labels(status="ok").inc()
            if current_status != "ok":
                log.info("image watchdog: FLUX recovered; status -> ok")
                current_status = "ok"
                await write_status(queue_client, "ok", since_iso=_now_iso())
        except FluxClientError as exc:
            now = asyncio.get_running_loop().time()
            elapsed = now - last_success
            metrics.flux_health_probe_total.labels(status="degraded").inc()
            log.warning(
                "image watchdog: FLUX probe failed (%.0fs since last success): %s",
                elapsed,
                exc,
            )
            if current_status == "ok" and elapsed >= DEGRADED_THRESHOLD_S:
                log.warning(
                    "image watchdog: %ds threshold crossed; status -> degraded",
                    int(DEGRADED_THRESHOLD_S),
                )
                current_status = "degraded"
                await write_status(queue_client, "degraded", since_iso=_now_iso())
        await asyncio.sleep(POLL_INTERVAL_S)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def main(
    *,
    factory: async_sessionmaker[AsyncSession] | None = None,
    storage_dir: Path | None = None,
) -> None:
    """Run the worker until cancelled (e.g. SIGTERM from systemd).

    Parameters are injectable so tests can drive the loop with a
    one-shot factory + tmp_path. Production passes nothing and
    inherits the module-level :data:`SessionLocal` and
    ``settings.image_storage_path``.
    """

    settings = get_settings()
    factory = factory or SessionLocal
    storage_dir = storage_dir or settings.image_storage_path

    flux = get_flux_client()
    pubsub = get_pubsub()
    queue_client = open_queue_client()
    watchdog = asyncio.create_task(_watchdog(flux, queue_client), name="flux-watchdog")
    log.info("image worker: started (storage=%s)", storage_dir)
    try:
        while True:
            try:
                job = await pop_job(queue_client, timeout=_POP_TIMEOUT_S)
            except ImageJobDecodeError as exc:
                log.error("image worker: poison message dropped: %s", exc)
                continue
            except Exception:
                log.exception("image worker: queue pop failed; sleeping briefly")
                await asyncio.sleep(1.0)
                continue
            if job is None:
                continue
            try:
                await _process_job(
                    job,
                    flux=flux,
                    pubsub=pubsub,
                    factory=factory,
                    storage_dir=storage_dir,
                )
            except Exception:
                log.exception("image worker: unhandled error in _process_job")
    finally:
        watchdog.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watchdog
        await queue_client.aclose()
        await flux.aclose()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(main())
