"""Tests for app.llm.client — the runaway-token detector and shape of
``stream_dm`` plumbing.

The real vLLM endpoint is not exercised here; the integration test
under ``tests/integration/`` handles that.
"""

from __future__ import annotations

import json as _json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncOpenAI

from app.llm.client import (
    DmClient,
    RunawayTokenError,
    _reasoning_extra_body,
    _watch_for_runaways,
)


def _make_chunk(content: str | None) -> Any:
    """Build a minimal object that quacks like ``ChatCompletionChunk`` —
    ``chunk.choices[0].delta.content`` is the only attribute the detector
    looks at."""

    class _Delta:
        def __init__(self, c: str | None) -> None:
            self.content = c

    class _Choice:
        def __init__(self, c: str | None) -> None:
            self.delta = _Delta(c)

    class _Chunk:
        def __init__(self, c: str | None) -> None:
            self.choices = [_Choice(c)]

    return _Chunk(content)


async def _stream_from(items: list[Any]) -> AsyncIterator[Any]:
    for it in items:
        yield it


@pytest.mark.asyncio
async def test_detector_passes_normal_stream() -> None:
    """Varied content tokens flow through unchanged."""

    chunks = [_make_chunk(c) for c in ("Hello", " there", ", how", " are", " you?")]
    out: list[Any] = []
    async for chunk in _watch_for_runaways(_stream_from(chunks)):
        out.append(chunk)
    assert len(out) == len(chunks)


@pytest.mark.asyncio
async def test_detector_passes_repeated_punctuation_below_threshold() -> None:
    """Up to and including 50 consecutive identical tokens is allowed —
    the detector trips on the 51st repeat (>50)."""

    chunks = [_make_chunk("!") for _ in range(50)]
    out: list[Any] = []
    async for chunk in _watch_for_runaways(_stream_from(chunks)):
        out.append(chunk)
    assert len(out) == 50


@pytest.mark.asyncio
async def test_detector_trips_on_runaway() -> None:
    """The qwen3_coder failure-mode signature: 100+ identical ``!`` tokens.
    The detector raises RunawayTokenError before consuming all of them."""

    chunks = [_make_chunk("!") for _ in range(100)]
    seen = 0
    with pytest.raises(RunawayTokenError):
        async for _ in _watch_for_runaways(_stream_from(chunks)):
            seen += 1
    # Trip point is the 51st-and-beyond identical token; we should have
    # yielded at least the first 50 before the raise.
    assert seen >= 50


@pytest.mark.asyncio
async def test_detector_resets_on_different_token() -> None:
    """A different token between repeats resets the counter — a model
    legitimately emitting many ``!`` separated by other tokens shouldn't
    trip."""

    pattern = ["!", "?", "!", "?"] * 100  # 400 chunks, no run > 1
    out: list[Any] = []
    async for chunk in _watch_for_runaways(_stream_from([_make_chunk(c) for c in pattern])):
        out.append(chunk)
    assert len(out) == 400


@pytest.mark.asyncio
async def test_detector_ignores_empty_chunks() -> None:
    """Tool-call chunks (empty content) and the final stop chunk
    (``content == ''``) flow through and do not affect the run counter."""

    chunks = [
        _make_chunk("Hello"),
        _make_chunk(None),  # tool-call chunk
        _make_chunk(""),  # stop-marker chunk
        _make_chunk("Hello"),  # not adjacent — counter shouldn't be 2
    ]
    out: list[Any] = []
    async for chunk in _watch_for_runaways(_stream_from(chunks)):
        out.append(chunk)
    assert len(out) == 4


# ---------------------------------------------------------------------------
# reasoning_mode plumbing — Phase 5 prep #2
# ---------------------------------------------------------------------------


def test_reasoning_extra_body_full_returns_none() -> None:
    """``"full"`` is Nemotron's default; we don't pass kwargs for it
    so the boot logs stay quiet about the no-op."""

    assert _reasoning_extra_body("full") is None


def test_reasoning_extra_body_low_emits_low_effort_kwargs() -> None:
    """``"low"`` maps to ``enable_thinking=True, low_effort=True`` — the
    canonical compression mode summarisers + fact extractor use."""

    assert _reasoning_extra_body("low") == {
        "chat_template_kwargs": {"enable_thinking": True, "low_effort": True}
    }


def test_reasoning_extra_body_off_disables_thinking() -> None:
    """``"off"`` maps to ``enable_thinking=False`` — reserved knob; no
    current call site uses it but the schema needs to exist."""

    assert _reasoning_extra_body("off") == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_complete_default_full_does_not_set_extra_body() -> None:
    """A ``complete()`` call without ``reasoning_mode`` (= full) must
    not include ``extra_body`` in the OpenAI request — full reasoning
    is Nemotron's default and an explicit empty kwarg dict would still
    show up in audit logs."""

    client = DmClient()
    fake_create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))],
            usage=None,
        )
    )
    client._client.chat.completions.create = fake_create  # type: ignore[method-assign]
    await client.complete([{"role": "user", "content": "hi"}])
    args, kwargs = fake_create.call_args
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_complete_low_passes_chat_template_kwargs() -> None:
    """A ``complete(..., reasoning_mode="low")`` must thread the
    ``chat_template_kwargs`` payload into the OpenAI request's
    ``extra_body`` slot — that's how vLLM forwards them to the chat
    template."""

    client = DmClient()
    fake_create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))],
            usage=None,
        )
    )
    client._client.chat.completions.create = fake_create  # type: ignore[method-assign]
    await client.complete([{"role": "user", "content": "hi"}], reasoning_mode="low")
    args, kwargs = fake_create.call_args
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True, "low_effort": True}
    }


@pytest.mark.asyncio
async def test_stream_dm_default_full_does_not_set_extra_body() -> None:
    """``stream_dm()`` defaults to full reasoning; mirrors the
    ``complete()`` contract so the DM turn loop's payload stays
    minimal."""

    client = DmClient()
    fake_create = AsyncMock(return_value=_stream_from([]))
    client._client.chat.completions.create = fake_create  # type: ignore[method-assign]
    await client.stream_dm([{"role": "user", "content": "hi"}])
    args, kwargs = fake_create.call_args
    assert "extra_body" not in kwargs
    assert kwargs["stream"] is True


@pytest.mark.asyncio
async def test_stream_dm_low_passes_chat_template_kwargs() -> None:
    """``stream_dm(..., reasoning_mode="low")`` threads the kwargs
    through identically. No current call site uses this; it's wired
    for symmetry so a future caller doesn't have to special-case the
    streaming entry point."""

    client = DmClient()
    fake_create = AsyncMock(return_value=_stream_from([]))
    client._client.chat.completions.create = fake_create  # type: ignore[method-assign]
    await client.stream_dm([{"role": "user", "content": "hi"}], reasoning_mode="low")
    args, kwargs = fake_create.call_args
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True, "low_effort": True}
    }


# ---------------------------------------------------------------------------
# HTTP transport boundary — Phase 8 fixup
#
# The _FakeDmClient in test_dm.py mocks at the stream_dm() boundary: it
# validates that the orchestrator passes the right kwargs, but it never
# exercises what stream_dm() actually puts in the HTTP body. These tests
# intercept at the httpx transport layer (below the OpenAI SDK, above the
# network) to verify the wire payload. They catch mis-serialisations that
# the stream_dm-level mock cannot. See AGENTS.md Code Conventions → Tests.
# ---------------------------------------------------------------------------

_SSE_RESPONSE = (
    'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},'
    '"finish_reason":null}]}\n\n'
    'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


class _CapturingTransport(httpx.AsyncBaseTransport):
    """httpx transport that captures the last request body and returns a
    canned SSE response. Wired into AsyncOpenAI via ``http_client=``."""

    def __init__(self) -> None:
        self.captured_body: dict[str, Any] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.captured_body = _json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_SSE_RESPONSE.encode(),
        )


def _client_with_transport(transport: _CapturingTransport) -> DmClient:
    http_client = httpx.AsyncClient(transport=transport)
    oai = AsyncOpenAI(base_url="http://fake/v1", api_key="x", http_client=http_client)
    dm = DmClient()
    dm._client = oai  # type: ignore[assignment]
    return dm


@pytest.mark.asyncio
async def test_stream_dm_sends_max_tokens_in_http_body() -> None:
    """stream_dm() must serialize max_tokens into the JSON body sent to
    vLLM. Intercepted at the httpx transport layer so any SDK-level
    renaming or omission of the field is caught here."""

    transport = _CapturingTransport()
    client = _client_with_transport(transport)

    stream = await client.stream_dm(
        [{"role": "user", "content": "test"}],
        max_tokens=2048,
    )
    async for _ in stream:
        pass

    assert transport.captured_body.get("max_tokens") == 2048


@pytest.mark.asyncio
async def test_stream_dm_full_reasoning_sends_no_extra_body_in_http() -> None:
    """For reasoning_mode=full (the default), no extra_body key must appear
    in the HTTP payload — Nemotron's default is full reasoning and an
    explicit empty dict would add noise to the boot logs."""

    transport = _CapturingTransport()
    client = _client_with_transport(transport)

    stream = await client.stream_dm([{"role": "user", "content": "test"}])
    async for _ in stream:
        pass

    assert "extra_body" not in transport.captured_body


@pytest.mark.asyncio
async def test_stream_dm_low_reasoning_sends_chat_template_kwargs_in_http() -> None:
    """For reasoning_mode=low, chat_template_kwargs must appear in the
    HTTP body via extra_body — that's how vLLM forwards them to the
    chat template."""

    transport = _CapturingTransport()
    client = _client_with_transport(transport)

    stream = await client.stream_dm(
        [{"role": "user", "content": "test"}],
        reasoning_mode="low",
    )
    async for _ in stream:
        pass

    # The OpenAI SDK merges extra_body keys into the top-level request
    # body — there is no nested "extra_body" key on the wire.
    assert transport.captured_body.get("chat_template_kwargs") == {
        "enable_thinking": True,
        "low_effort": True,
    }
