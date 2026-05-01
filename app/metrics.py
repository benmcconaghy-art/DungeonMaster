"""Prometheus metrics surface (Phase 7 hardening).

Exposes ``/metrics`` in the Prometheus exposition format. Internal-only
endpoint — production deploy binds the FastAPI worker to localhost and
nginx doesn't expose ``/metrics`` to the LAN proxy block, so reaching it
requires shell access to the box.

Metric selection follows the brief: pick the metrics that tell the
"is the system healthy" + "what is it doing right now" story, not
every possible counter. The orchestrator, LLM client, and image worker
import from here and call the instrumentation helpers; the metrics
themselves live in this module so the registry is one ``import`` away
from any call site without circular imports.

Cardinality discipline:
- ``http_requests_total`` uses the path *template* (``request.url.path``
  pre-resolves to ``/api/campaigns/{campaign_id}/invite`` at this layer
  for path-parameterised routes), not the resolved URL. We further
  drop high-cardinality cases (the campaign-id template is fine; a
  catch-all isn't).
- ``llm_calls_total`` is keyed by ``model`` + ``reasoning_mode`` +
  ``outcome`` only. No per-prompt or per-user labels.
- ``dm_tool_dispatch_total`` is keyed by ``tool_name`` (small enum,
  bounded by the registered tools) + ``outcome``.
- ``image_jobs_total`` is keyed by ``kind`` (npc/scene) + ``outcome``.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,  # noqa: F401  -- imported for future multi-worker scale-out
)

# We use the default global registry. Multi-process / multi-worker
# would need ``multiprocess.MultiProcessCollector`` and a shared dir
# under ``PROMETHEUS_MULTIPROC_DIR``; spec §13's single-gunicorn-worker
# deployment doesn't need it.
REGISTRY: CollectorRegistry | None = None  # ``None`` = use the default global


# ---------------------------------------------------------------------------
# HTTP request metrics — emitted from app.middleware.AccessLogMiddleware
# ---------------------------------------------------------------------------

http_requests_total = Counter(
    "dm_http_requests_total",
    "HTTP requests handled by the FastAPI worker.",
    labelnames=("method", "path", "status"),
)

http_request_duration_seconds = Histogram(
    "dm_http_request_duration_seconds",
    "HTTP request handler latency.",
    labelnames=("method", "path"),
    # Buckets sized for player-facing turn handling (mid-second LLM
    # calls dominate) plus health/short endpoints. Default Prometheus
    # buckets undersample the 1-30s range that matters here.
    buckets=(
        0.005,
        0.025,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
    ),
)


# ---------------------------------------------------------------------------
# LLM call metrics — emitted from app.llm.client.complete()
# ---------------------------------------------------------------------------

llm_calls_total = Counter(
    "dm_llm_calls_total",
    "LLM (vLLM) chat-completion calls.",
    labelnames=("model", "reasoning_mode", "outcome"),
)

llm_tokens_total = Counter(
    "dm_llm_tokens_total",
    "LLM token consumption (prompt + completion are separate label values).",
    labelnames=("model", "kind"),  # kind = "prompt" | "completion"
)


# ---------------------------------------------------------------------------
# DM tool dispatch — emitted from app.orchestrator.dm._dispatch_tool_call
# ---------------------------------------------------------------------------

dm_tool_dispatch_total = Counter(
    "dm_tool_dispatch_total",
    "Tool calls the orchestrator dispatched to the rules engine.",
    labelnames=("tool_name", "outcome"),  # outcome = "ok" | "error"
)

dice_rolls_total = Counter(
    "dm_dice_rolls_total",
    "Dice rolls resolved by the rules engine.",
    labelnames=("purpose", "actor_kind"),
    # purpose = "attack" | "save" | "ability_check" | "damage" | "init" | ...
    # actor_kind = "pc" | "monster" | "system"
)


# ---------------------------------------------------------------------------
# Image generation — emitted from app.images.worker
# ---------------------------------------------------------------------------

image_jobs_total = Counter(
    "dm_image_jobs_total",
    "Image generation jobs processed by the worker.",
    labelnames=("kind", "outcome"),  # kind = "npc" | "scene", outcome = "ok"|"failed"
)

image_job_duration_seconds = Histogram(
    "dm_image_job_duration_seconds",
    "FLUX wall-clock latency per image job.",
    labelnames=("kind",),
    # FLUX measurements (Phase 5 close-out): 256x256/1-step ~5s warm,
    # 1280x768/28-steps ~17s warm. Buckets cover from probe-shaped
    # tiny jobs to slow cold-load scenarios.
    buckets=(1.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0),
)

flux_health_probe_total = Counter(
    "dm_flux_health_probe_total",
    "FLUX deep-health probes performed by the watchdog.",
    labelnames=("status",),  # "ok" | "degraded"
)


# ---------------------------------------------------------------------------
# Realtime / pub-sub — emitted from app.realtime.hub + presence + pubsub
# ---------------------------------------------------------------------------

active_sessions = Gauge(
    "dm_active_sessions",
    "Number of game sessions currently subscribed to by at least one client.",
)

connected_clients = Gauge(
    "dm_connected_clients",
    "Number of WebSocket connections currently bound to the hub.",
)

valkey_publish_total = Counter(
    "dm_valkey_publish_total",
    "Pubsub messages published to Valkey by channel kind.",
    labelnames=("channel_kind",),  # "session" | "image" | other
)


# ---------------------------------------------------------------------------
# Exposition
# ---------------------------------------------------------------------------


def render_exposition() -> tuple[bytes, str]:
    """Render the current metric snapshot in Prometheus exposition
    format. Returns (body, content_type) suitable for a FastAPI
    ``Response`` straight out.
    """

    return generate_latest(), CONTENT_TYPE_LATEST


__all__ = [
    "active_sessions",
    "connected_clients",
    "dice_rolls_total",
    "dm_tool_dispatch_total",
    "flux_health_probe_total",
    "http_request_duration_seconds",
    "http_requests_total",
    "image_job_duration_seconds",
    "image_jobs_total",
    "llm_calls_total",
    "llm_tokens_total",
    "render_exposition",
    "valkey_publish_total",
]
