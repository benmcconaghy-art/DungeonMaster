"""Image job queue — typed shape + Valkey list helpers.

The image generation pipeline crosses two processes (the FastAPI app
that enqueues, the imageworker systemd unit that consumes), so the
job shape lives in its own module that both sides import:

  enqueue (web app)         BRPOP (imageworker)
  ──────────────────        ─────────────────────
  build :class:`ImageJob`   pop bytes off ``images:queue``
  ``push_job(job)``         decode + validate
                            dispatch to FLUX
                            persist + publish

The list key ``images:queue`` is treated as a FIFO via ``RPUSH`` (push
to the right) and ``BLPOP`` (block-pop from the left). The worker uses
a long-lived blocking pop so jobs are picked up the moment they land.

We use the raw redis client here — :class:`~app.realtime.pubsub.Pubsub`
is scoped to the session pub/sub channels and would be the wrong
abstraction for a queue. This module owns its own connection.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, cast

import redis.asyncio as redis
from pydantic import BaseModel, Field

from app.config import get_settings

log = logging.getLogger(__name__)


# Redis list key. Single shared queue across all campaigns; the FLUX
# service serialises with its own asyncio.Lock so worker concurrency >1
# wouldn't help (spec §8 "Throttling & failure").
QUEUE_KEY = "images:queue"


ImageKind = Literal["scene", "npc", "item", "map"]


class ImageJob(BaseModel):
    """A single image-generation job.

    Either ``reference_image_id`` is unset (txt2img via FLUX
    ``/generate``) or it is set together with ``edit_instruction``
    (Kontext ``/edit`` using the referenced image as the source).
    The worker validates this invariant before dispatching.

    ``id`` is pre-allocated by the enqueuer so the worker can write
    the file under a known UUID and the WS layer can show a
    ``image_pending`` placeholder card before generation completes.
    """

    id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    # Active session for the WS broadcast. Optional because canonical
    # portraits can be requested outside of an active play session
    # (e.g. character creation in a campaign with no live session yet);
    # the worker still generates and persists, but skips the
    # ``image_ready`` broadcast.
    session_id: str | None = None
    kind: ImageKind
    prompt: str = Field(min_length=1)
    reference_image_id: str | None = None
    edit_instruction: str | None = None
    # Optional FK targets the worker should update with this image's
    # id once the row commits. Used by the Step 7 portrait flow:
    # ``characters.canonical_image_id`` / ``npcs.canonical_image_id``.
    subject_character_id: str | None = None
    subject_npc_id: str | None = None


def open_queue_client(*, url: str | None = None) -> redis.Redis:
    """Build a queue-scoped redis client.

    Separate from :class:`~app.realtime.pubsub.Pubsub` so queue traffic
    doesn't share connection-pool state with the session pub/sub
    channels. Caller owns ``aclose()``.
    """

    settings = get_settings()
    client: redis.Redis = redis.Redis.from_url(url or settings.redis_url, decode_responses=False)
    return client


async def push_job(client: redis.Redis, job: ImageJob) -> int:
    """Append ``job`` to ``images:queue`` (FIFO via ``RPUSH``).

    Returns the new length of the list — useful for telemetry. Raises
    :class:`redis.RedisError` on transport failure; the enqueuer
    surfaces this as an HTTP 503 to the caller (transient backend
    issue, retry safe).
    """

    payload = job.model_dump_json().encode("utf-8")
    length = await cast(Any, client).rpush(QUEUE_KEY, payload)
    return int(length)


async def pop_job(client: redis.Redis, *, timeout: float = 0.0) -> ImageJob | None:  # noqa: ASYNC109 (BLPOP timeout is a server-side wire parameter, not a Python cancellation deadline)
    """Block-pop the next job off the head of the queue.

    ``timeout=0.0`` blocks forever; pass a finite value (the worker
    uses a few seconds) so the loop can wake periodically and check
    cancellation / health flags between pops. Returns ``None`` on
    timeout, otherwise the validated :class:`ImageJob`.

    Raises :class:`ImageJobDecodeError` if the popped bytes don't
    parse as a valid :class:`ImageJob` — the worker logs and skips,
    rather than wedging the queue on a poison message.
    """

    # redis-py's async ListCommands.blpop is typed as accepting
    # ``int | None`` even though the BLPOP wire protocol has accepted
    # fractional seconds since Redis 6.0. Cast through Any so callers
    # can pass natural floats (0.5, 1.5) without coercing to int.
    result: Any = await cast(Any, client).blpop(QUEUE_KEY, timeout=timeout)
    if result is None:
        return None
    _key, raw = result
    try:
        data = json.loads(raw)
        return ImageJob.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ImageJobDecodeError(f"image queue had unparseable payload: {raw!r}") from exc


class ImageJobDecodeError(RuntimeError):
    """Raised when ``pop_job`` finds bytes that don't decode to a
    valid :class:`ImageJob`. The worker logs and skips — a single
    poison message must not wedge the queue."""


__all__ = [
    "QUEUE_KEY",
    "ImageJob",
    "ImageJobDecodeError",
    "ImageKind",
    "open_queue_client",
    "pop_job",
    "push_job",
]
