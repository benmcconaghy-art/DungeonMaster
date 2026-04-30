"""Tests for ``app.images.client`` — the FLUX HTTP client wrapper.

Real FLUX is not exercised here; the integration test under
``tests/integration/`` (Phase 5) handles that. These tests pin:

- request shape for ``/health``, ``/generate``, ``/edit``
- response decoding (base64 → bytes, ``seed_used``)
- 503 retry policy + backoff schedule (5s / 15s / 45s)
- non-retryable failure paths (4xx, transport, malformed bodies)

httpx's :class:`MockTransport` is the seam — we wrap the real
:class:`httpx.AsyncClient` constructor so the FluxClient builds a
client backed by our handler.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.images import client as flux_module
from app.images.client import (
    _RETRY_BACKOFF_SECONDS,
    FluxClient,
    FluxClientError,
    get_flux_client,
    reset_for_tests,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_bytes() -> bytes:
    """A handful of bytes that round-trip through base64 — the actual
    PNG header doesn't matter to the client wrapper, which only encodes
    / decodes opaquely."""

    return b"\x89PNG\r\n\x1a\n" + b"fake-image-data"


def _ok_image_payload(*, png: bytes | None = None, seed: int = 42) -> dict[str, Any]:
    return {
        "image_base64": base64.b64encode(png or _png_bytes()).decode("ascii"),
        "seed_used": seed,
        "generation_time_seconds": 12.5,
        "filepath": None,
    }


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Replace ``httpx.AsyncClient`` (as imported by the client module)
    with one that routes through a :class:`MockTransport` running
    ``handler``. Returns a list that captures every ``httpx.Request``
    the client makes, so tests can assert on request shape after the
    call."""

    captured: list[httpx.Request] = []

    def wrapped_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(wrapped_handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return real_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr("app.images.client.httpx.AsyncClient", fake_async_client)
    return captured


def _patch_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch ``asyncio.sleep`` inside the client module so retry
    backoff doesn't burn wall-clock seconds in tests. Returns a list
    capturing every duration the client requested."""

    durations: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        durations.append(seconds)

    monkeypatch.setattr("app.images.client.asyncio.sleep", fake_sleep)
    return durations


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_returns_parsed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: ``/health`` returns the JSON the FLUX service
    actually emits (status, model-loaded flags, GPU info)."""

    payload = {
        "status": "ok",
        "flux_txt2img_loaded": False,
        "kontext_loaded": False,
        "gpu": "NVIDIA GeForce RTX 5090",
        "vram_allocated_gb": 0.01,
        "vram_total_gb": 31.36,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(200, json=payload)

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        result = await client.health()
    finally:
        await client.aclose()

    assert result == payload


@pytest.mark.asyncio
async def test_health_raises_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 500 from /health surfaces as ``FluxClientError`` so the
    watchdog has a single exception type to catch."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        with pytest.raises(FluxClientError, match="health"):
            await client.health()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_health_raises_on_non_object_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the service ever returns a JSON list / scalar instead of an
    object, surface a clean error rather than letting the worker crash
    on ``payload["status"]``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "list"])

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        with pytest.raises(FluxClientError, match="non-object"):
            await client.health()
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# generate() — request shape + response decoding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_sends_full_payload_and_decodes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The /generate request body must match the FastAPI shape spec §8
    documents: prompt + negative_prompt + width/height + steps + guidance
    + seed (nullable). The response's ``image_base64`` must round-trip
    to the bytes the caller passed in."""

    expected_png = _png_bytes()
    payload = _ok_image_payload(png=expected_png, seed=12345)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/generate"
        body = json.loads(request.content)
        assert body == {
            "prompt": "an alchemist's tower at sunset",
            "negative_prompt": "modern objects, watermark",
            "width": 1280,
            "height": 768,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "seed": None,
        }
        return httpx.Response(200, json=payload)

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        png, seed_used = await client.generate(
            "an alchemist's tower at sunset",
            negative_prompt="modern objects, watermark",
            width=1280,
            height=768,
        )
    finally:
        await client.aclose()

    assert png == expected_png
    assert seed_used == 12345


@pytest.mark.asyncio
async def test_generate_passes_explicit_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the worker reuses a seed for reproducibility, the client
    must thread it through unchanged (not coerce ``None``)."""

    captured_body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.content)
        return httpx.Response(200, json=_ok_image_payload(seed=99))

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        _png, seed_used = await client.generate("x", seed=99)
    finally:
        await client.aclose()

    assert captured_body["seed"] == 99
    assert seed_used == 99


# ---------------------------------------------------------------------------
# edit() — base64 encoding of source_png
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_base64_encodes_source_png(monkeypatch: pytest.MonkeyPatch) -> None:
    """The source bytes go in the request body as ``image_base64``;
    callers must not have to encode themselves."""

    source = b"\x89PNG-source-bytes-arbitrary"
    response_png = b"\x89PNG-edited-result"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/edit"
        body = json.loads(request.content)
        assert body["prompt"] == "same character, torchlit crypt"
        assert body["image_base64"] == base64.b64encode(source).decode("ascii")
        assert body["num_inference_steps"] == 28
        assert body["guidance_scale"] == 2.5
        return httpx.Response(200, json=_ok_image_payload(png=response_png, seed=7))

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        png, seed_used = await client.edit(
            "same character, torchlit crypt",
            source,
        )
    finally:
        await client.aclose()

    assert png == response_png
    assert seed_used == 7


# ---------------------------------------------------------------------------
# 503 retry policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_503_retries_with_backoff_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two 503s in a row, then a 200. The wrapper must absorb both
    failures, sleep the configured backoff each time, and surface the
    eventual success transparently."""

    sleeps = _patch_sleep(monkeypatch)
    responses: list[httpx.Response] = [
        httpx.Response(503, text="busy"),
        httpx.Response(503, text="busy"),
        httpx.Response(200, json=_ok_image_payload(seed=4)),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    captured = _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        png, seed_used = await client.generate("anything")
    finally:
        await client.aclose()

    assert seed_used == 4
    assert png == _png_bytes()
    # Three POSTs total (two failures + one success).
    assert len(captured) == 3
    # Backoff schedule executed for the first two retries (5s, 15s);
    # the third attempt succeeded, so 45s sleep is not exercised.
    assert sleeps == [_RETRY_BACKOFF_SECONDS[0], _RETRY_BACKOFF_SECONDS[1]]


@pytest.mark.asyncio
async def test_503_retry_exhaustion_raises_after_full_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four straight 503s exhaust the retry budget. The wrapper must
    raise ``FluxClientError`` after sleeping the full 5/15/45 schedule
    (three sleeps for four attempts)."""

    sleeps = _patch_sleep(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    captured = _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        with pytest.raises(FluxClientError, match="503"):
            await client.generate("anything")
    finally:
        await client.aclose()

    # 1 initial + 3 retries = 4 attempts.
    assert len(captured) == 4
    # Slept the full schedule before each retry; no sleep after the
    # final 503 (we raise instead).
    assert tuple(sleeps) == _RETRY_BACKOFF_SECONDS


@pytest.mark.asyncio
async def test_backoff_schedule_matches_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """The constants are part of the contract: the spec calls out
    5s / 15s / 45s explicitly. A regression that changes the schedule
    silently would shift the tail-latency story significantly."""

    assert _RETRY_BACKOFF_SECONDS == (5.0, 15.0, 45.0)


# ---------------------------------------------------------------------------
# Non-retryable failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_503_status_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 is a caller error — retrying won't help. Raise on the
    first response without sleeping or making further requests."""

    sleeps = _patch_sleep(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad prompt")

    captured = _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        with pytest.raises(FluxClientError, match="400"):
            await client.generate("x")
    finally:
        await client.aclose()

    assert len(captured) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_500_non_oom_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry policy is scoped to 503. A plain 500 (e.g. an
    unhandled exception in the FLUX service) is not retried — the
    odds of a different result on retry are low and we'd rather fail
    fast."""

    sleeps = _patch_sleep(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    captured = _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        with pytest.raises(FluxClientError, match="500"):
            await client.generate("x")
    finally:
        await client.aclose()

    assert len(captured) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_transport_error_raises_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connect / read timeout from httpx is not the same as a 503.
    The wrapper must surface it as ``FluxClientError`` without
    retrying — connection failures usually mean the service is down
    entirely, and the worker treats this case via the watchdog."""

    sleeps = _patch_sleep(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        with pytest.raises(FluxClientError, match="transport"):
            await client.generate("x")
    finally:
        await client.aclose()

    assert sleeps == []


# ---------------------------------------------------------------------------
# Malformed response bodies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_missing_image_base64_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 with the wrong shape should fail loudly — silently
    returning ``b""`` would let the worker write an empty PNG to disk."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"seed_used": 1})

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        with pytest.raises(FluxClientError, match="image_base64"):
            await client.generate("x")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_response_missing_seed_used_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"image_base64": base64.b64encode(b"x").decode()})

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        with pytest.raises(FluxClientError, match="seed_used"):
            await client.generate("x")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_response_invalid_base64_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"image_base64": "not!valid!base64!", "seed_used": 1},
        )

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        with pytest.raises(FluxClientError, match="base64"):
            await client.generate("x")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_response_non_json_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"content-type": "text/plain"})

    _install_transport(monkeypatch, handler)
    client = FluxClient(base_url="http://test")
    try:
        with pytest.raises(FluxClientError, match="non-JSON"):
            await client.generate("x")
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Singleton wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_singleton_returns_same_instance() -> None:
    await reset_for_tests()
    a = get_flux_client()
    b = get_flux_client()
    try:
        assert a is b
    finally:
        await reset_for_tests()


@pytest.mark.asyncio
async def test_reset_for_tests_disposes_singleton() -> None:
    """``reset_for_tests`` must close the held transport so a
    subsequent test gets a fresh one — without this, the file-backed
    singleton would carry over event-loop bindings from the previous
    test."""

    await reset_for_tests()
    first = get_flux_client()
    await reset_for_tests()
    assert flux_module._singleton is None
    second = get_flux_client()
    assert first is not second
    await reset_for_tests()
