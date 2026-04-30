"""FLUX availability state — written by the imageworker watchdog,
read by the orchestrator's prompt builder.

Spec §8 "Throttling & failure":

> The /health endpoint is polled every 30s by a watchdog. If
> unreachable for >2 min, image generation is marked degraded and
> the DM is told (in-prompt) to omit ``generate_scene_image`` calls
> until further notice.

The two processes can't communicate via in-memory state (they're
separate systemd units), so we use a Valkey key as the rendezvous:

  Key:   ``image:status``
  Value: JSON ``{"status": "ok"|"degraded", "since": "<iso8601>"}``

The watchdog updates the key each time it transitions; the
orchestrator's prompt builder reads the key on every turn (cheap —
single GET) and prepends a one-line system note when status is
degraded. Reads default to ``ok`` if the key is missing (e.g. the
worker hasn't started yet) — the FLUX service may still be reachable
and the DM behaviour shouldn't be conservative just because the
watchdog is silent.

Two minutes of unreachability before flipping to degraded matches
the spec's tolerance for transient blips (the FLUX service unloads
its pipeline after every request, so a fresh hit can take 15-30s
during which /health may be slow but not yet failed). Three
consecutive failed 30s polls is the trigger.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, cast

import redis.asyncio as redis

log = logging.getLogger(__name__)


HEALTH_KEY = "image:status"

# Watchdog cadence and threshold from spec §8.
POLL_INTERVAL_S = 30.0
DEGRADED_THRESHOLD_S = 120.0


Status = Literal["ok", "degraded"]


async def write_status(
    client: redis.Redis,
    status: Status,
    *,
    since_iso: str,
) -> None:
    """Update the shared FLUX status key.

    ``since_iso`` is the timestamp of the most recent transition INTO
    the current state — used so the orchestrator can render
    "image generation degraded since 12:34" rather than "currently
    degraded" with no context.
    """

    payload = json.dumps({"status": status, "since": since_iso}).encode("utf-8")
    await cast(Any, client).set(HEALTH_KEY, payload)


async def read_status(client: redis.Redis) -> tuple[Status, str | None]:
    """Read the shared status. Returns ``(status, since_iso)``.

    Defaults to ``("ok", None)`` if the key is missing or unparseable
    — a silent watchdog should not by itself flip the DM into
    image-degraded behaviour.
    """

    raw: Any = await cast(Any, client).get(HEALTH_KEY)
    if raw is None:
        return "ok", None
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        log.warning("image:status had unparseable payload; defaulting to 'ok'")
        return "ok", None
    if not isinstance(decoded, dict):
        return "ok", None
    status = decoded.get("status")
    since = decoded.get("since")
    if status not in ("ok", "degraded"):
        return "ok", None
    return cast(Status, status), since if isinstance(since, str) else None


__all__ = [
    "DEGRADED_THRESHOLD_S",
    "HEALTH_KEY",
    "POLL_INTERVAL_S",
    "Status",
    "read_status",
    "write_status",
]
