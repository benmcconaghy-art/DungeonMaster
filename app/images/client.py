"""FLUX HTTP client.

Async wrapper around the FLUX.1 [dev] / FLUX.1 Kontext [dev] service running
on ``YOUR_AI_SERVER:11437`` (spec §8). Exposes four operations:

- ``health()`` — GET ``/health``. Cheap reachability probe; kept for
  app-startup wiring checks. NOT used by the watchdog (see ``probe``).
- ``probe()`` — POST ``/generate`` with 256x256/1-step. The watchdog's
  liveness signal. Bypasses ``/health`` because the FLUX service
  reports ``flux_txt2img_loaded:false`` even when ``/generate`` is
  fully working — a 200 from /health is necessary but not sufficient.
- ``generate(prompt, ...)`` — POST ``/generate`` (FLUX.1 txt2img).
- ``edit(prompt, source_png, ...)`` — POST ``/edit`` (FLUX.1 Kontext
  instruction-based edit, used for character / NPC consistency).

Both ``/generate`` and ``/edit`` are synchronous from the service's
perspective: the POST returns when the image is ready. Cold pipeline
load (~15-30s) plus 28-step inference (~8-18s) plus base64 encode is
typically 25-45s end-to-end, so the read timeout is set generously to
180s.

Failure handling (spec §8 "Throttling & failure"):
- HTTP 503 from the service indicates a transient overload (the
  service is mid-pipeline-load on a different request, or hit a
  transient OOM). Retry up to 3 times with exponential backoff —
  5s, 15s, 45s — before raising :class:`FluxClientError`. Total wall
  time spent waiting under full failure is 65s; the worker swallows
  that and emits ``image_failed`` to the table.
- Other status codes (400/404/500-non-OOM) raise immediately. The
  spec calls out "503 / OOM" specifically; the FLUX FastAPI service
  returns 503 for both pipeline-busy and transient OOM, so a single
  status check covers the case.
- Connect / read timeouts and non-HTTP transport errors raise
  :class:`FluxClientError` directly. The worker treats these like
  any other non-retry failure (DM narration is not blocked).

The DM orchestrator only ever sees ``flux.generate(...)`` /
``flux.edit(...)``. Everything FLUX-specific is contained here.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)


class FluxClientError(RuntimeError):
    """Generic FLUX client failure (transport, non-retryable status,
    retry exhaustion).

    Raised in a unified shape so the image worker's error path doesn't
    have to know about httpx exception hierarchies.
    """


# Backoff schedule for 503 retries. Three sleeps → up to four attempts.
# Module-level constant so tests can monkeypatch it to ``[]`` or
# ``[0.0, 0.0, 0.0]`` for fast retry-exhaustion coverage without
# burning 65 wall-clock seconds.
_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (5.0, 15.0, 45.0)


class FluxClient:
    """Long-lived async client. One instance per process; share via
    :func:`get_flux_client`.

    Holds a single :class:`httpx.AsyncClient` for all generation /
    edit calls. ``health()`` overrides the per-call timeout because
    the watchdog needs a snappy probe, not a 180s wait."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.flux_base_url).rstrip("/")
        # Read timeout covers cold-load + generation + encode. The
        # service serialises with its own asyncio.Lock so a queued
        # request can wait on a peer's cold-load before its own
        # generation even starts.
        self._timeout = timeout or httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def aclose(self) -> None:
        """Dispose the underlying transport. Called from the app
        lifespan and from ``reset_for_tests``."""

        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        """GET ``/health``. Returns the parsed JSON payload.

        Uses a 10s timeout — the watchdog wants a snappy yes/no, not
        a 180s wait against a wedged service. Raises
        :class:`FluxClientError` on any transport / status / parse
        error so the watchdog has a single exception type to catch.
        """

        try:
            response = await self._client.get("/health", timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise FluxClientError(f"FLUX health check failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise FluxClientError(
                f"FLUX /health returned non-object payload: {type(payload).__name__}"
            )
        return payload

    async def probe(self) -> dict[str, Any]:
        """POST ``/generate`` with 256x256/1-step as a deep liveness signal.

        The watchdog calls this every ``POLL_INTERVAL_S`` to verify the
        service is genuinely able to produce an image, not just answer
        a health endpoint. /health alone is insufficient: in observed
        runs the service returns 200 from /health while reporting
        ``flux_txt2img_loaded:false``, and yet /generate succeeds —
        and the converse failure (200 /health, 500 /generate) was the
        original incident that motivated this method.

        Single-shot: no 503 retry. The watchdog's polling cadence is
        itself the retry — chaining the client's 65s backoff inside a
        30s watchdog tick would cascade. A typical warm probe is ~5s;
        a cold-load probe can run a bit longer, hence the 60s read
        timeout (down from /generate's 180s, up from /health's 10s).

        Returns the parsed JSON dict so the watchdog can log
        ``generation_time_seconds`` if it wants. The image bytes in
        the response are discarded — the watchdog has no use for them.
        """

        payload: dict[str, Any] = {
            "prompt": "watchdog probe",
            "width": 256,
            "height": 256,
            "num_inference_steps": 1,
            "guidance_scale": 3.5,
        }
        try:
            response = await self._client.post("/generate", json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise FluxClientError(f"FLUX /generate probe failed: {exc}") from exc
        if not isinstance(data, dict):
            raise FluxClientError(
                f"FLUX /generate probe returned non-object: {type(data).__name__}"
            )
        return data

    async def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        guidance: float = 3.5,
        seed: int | None = None,
    ) -> tuple[bytes, int]:
        """POST ``/generate``. Returns ``(png_bytes, seed_used)``.

        Defaults match FLUX.1 [dev] best practice. The image worker
        passes per-kind overrides (spec §8 "Generation parameters per
        kind"): scenes are 1280x768, NPCs are 768x1024 with 32 steps,
        items are 1024x1024 with 24 steps, maps are 1280x1280 with 36
        steps + guidance 4.0.
        """

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
        }
        data = await self._post_with_retry("/generate", payload)
        return self._decode_image_response(data)

    async def edit(
        self,
        prompt: str,
        source_png: bytes,
        *,
        steps: int = 28,
        guidance: float = 2.5,
        seed: int | None = None,
    ) -> tuple[bytes, int]:
        """POST ``/edit``. Returns ``(png_bytes, seed_used)``.

        ``prompt`` is the edit instruction (e.g. "same character,
        torchlit crypt, sword drawn"). ``source_png`` is the raw bytes
        of the canonical portrait or previous render — encoded to
        base64 inside the wrapper so callers don't have to.
        """

        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_base64": base64.b64encode(source_png).decode("ascii"),
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
        }
        data = await self._post_with_retry("/edit", payload)
        return self._decode_image_response(data)

    # ------------------------------------------------------------------

    async def _post_with_retry(self, path: str, json_payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``path`` with the 503 retry policy.

        Retries on HTTP 503 with the ``_RETRY_BACKOFF_SECONDS`` sleep
        schedule. Other failures (4xx, 5xx ≠ 503, transport, parse)
        raise :class:`FluxClientError` immediately.
        """

        last_status: int | None = None
        for attempt, sleep_after in enumerate((*_RETRY_BACKOFF_SECONDS, None)):
            try:
                response = await self._client.post(path, json=json_payload)
            except httpx.HTTPError as exc:
                # Connect / read / write transport errors are not
                # retried — the spec only retries 503.
                raise FluxClientError(f"FLUX {path} transport error: {exc}") from exc

            if response.status_code == 503:
                last_status = 503
                if sleep_after is None:
                    raise FluxClientError(
                        f"FLUX {path} returned 503 after {attempt + 1} attempts"
                        f" (backoff {_RETRY_BACKOFF_SECONDS!r})"
                    )
                log.warning(
                    "FLUX %s returned 503; retry %d/%d in %.1fs",
                    path,
                    attempt + 1,
                    len(_RETRY_BACKOFF_SECONDS),
                    sleep_after,
                )
                await asyncio.sleep(sleep_after)
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise FluxClientError(
                    f"FLUX {path} returned {response.status_code}: {exc}"
                ) from exc

            try:
                payload = response.json()
            except ValueError as exc:
                raise FluxClientError(f"FLUX {path} returned non-JSON body: {exc}") from exc
            if not isinstance(payload, dict):
                raise FluxClientError(
                    f"FLUX {path} returned non-object payload:" f" {type(payload).__name__}"
                )
            return payload

        # The for-loop always returns or raises; this line is
        # unreachable, but mypy can't see that without it.
        raise FluxClientError(
            f"FLUX {path} retry loop exited unexpectedly (last_status={last_status})"
        )

    @staticmethod
    def _decode_image_response(data: dict[str, Any]) -> tuple[bytes, int]:
        """Pull ``image_base64`` + ``seed_used`` out of a /generate or
        /edit response. Raises :class:`FluxClientError` if either is
        missing or malformed — the worker treats this as a hard
        failure (the row would be useless without the bytes)."""

        b64 = data.get("image_base64")
        if not isinstance(b64, str):
            raise FluxClientError("FLUX response missing image_base64 string")
        seed_used = data.get("seed_used")
        if not isinstance(seed_used, int):
            raise FluxClientError("FLUX response missing seed_used int")
        try:
            png = base64.b64decode(b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise FluxClientError(f"FLUX image_base64 not valid base64: {exc}") from exc
        return png, seed_used


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_singleton: FluxClient | None = None


def get_flux_client() -> FluxClient:
    """Return the process-wide :class:`FluxClient`, building on first
    call. The lifespan hook in ``app.main`` should ``aclose()`` it on
    shutdown."""

    global _singleton
    if _singleton is None:
        _singleton = FluxClient()
    return _singleton


async def reset_for_tests() -> None:
    """Tear down the singleton — used by tests that swap the endpoint."""

    global _singleton
    if _singleton is not None:
        await _singleton.aclose()
        _singleton = None


__all__ = [
    "FluxClient",
    "FluxClientError",
    "get_flux_client",
    "reset_for_tests",
]
