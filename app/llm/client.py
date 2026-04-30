"""vLLM client wrapper.

Async wrapper around the OpenAI SDK pointed at our internal vLLM
endpoint serving Nemotron 3 Super (spec §7).

Two responsibilities the orchestrator depends on:

1. **Streaming chat completions** — ``stream_dm`` wraps
   ``client.chat.completions.create(..., stream=True)`` and returns an
   async iterator of chunks. The runaway-token detector lives here so
   every consumer of this client is protected automatically against the
   ``qwen3_coder`` infinite-``!`` failure mode (spec §7 watch-item, §15
   decision log).

2. **Health + model resolution** — at app startup we call ``health()``
   to confirm the endpoint is reachable and to log the actual model id
   the server reports. The id may differ from spec text (the spec says
   "nemotron-3-super"; the server reports
   ``nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4``). We use whatever
   the server says.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from app.config import get_settings

log = logging.getLogger(__name__)


# Nemotron exposes its reasoning effort via the chat-template-kwargs
# slot. Three modes the call sites pick from:
#
#   "full" (default) — full reasoning trace. Tool-call accuracy depends
#                       on it; the DM turn loop stays here.
#   "low"            — `enable_thinking=True, low_effort=True`. The
#                       canonical compression mode; summarisers run here.
#   "off"            — `enable_thinking=False`. Reserved for cases where
#                       reasoning would actively confuse the output;
#                       no current call site uses it.
#
# vLLM forwards ``extra_body`` keys into the engine's request payload,
# so OpenAI-style ``client.chat.completions.create(..., extra_body={...})``
# is how the kwargs reach the chat template.
ReasoningMode = Literal["full", "low", "off"]


def _reasoning_extra_body(mode: ReasoningMode) -> dict[str, Any] | None:
    """Render a reasoning mode into the ``extra_body`` payload vLLM
    expects, or ``None`` when no kwargs are needed (full reasoning is
    Nemotron's default — passing the flag explicitly is unnecessary noise
    in the boot logs).
    """

    if mode == "full":
        return None
    if mode == "low":
        return {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}
    return {"chat_template_kwargs": {"enable_thinking": False}}


class RunawayTokenError(RuntimeError):
    """Raised when the qwen3_coder parser falls into its repeating-token
    failure mode. The orchestrator should abort the request and surface
    a ``dm_error`` event to the table."""


class DmClientError(RuntimeError):
    """Generic client-side failure (transport, auth, model not found).

    Raised in a unified shape so the orchestrator's error handling
    doesn't have to know about httpx vs openai exception hierarchies.
    """


# Repeating-token threshold from spec §7. Spec says ``>50 consecutive
# identical tokens``; we trigger ON the 51st, so the comparison is
# strictly greater-than 50.
_RUNAWAY_THRESHOLD = 50


class DmClient:
    """Long-lived async client. One instance per process; share via
    ``get_dm_client()``."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        settings = get_settings()
        self._base_url = base_url or f"{settings.vllm_base_url}/v1"
        # Read timeout is generous because the model can take a while to
        # produce the first token under load; we rely on the streaming
        # protocol's heartbeat for liveness, not a tight read deadline.
        self._timeout = timeout or httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key="not-needed",  # vLLM accepts any string
            timeout=self._timeout,
        )
        self._resolved_model: str | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        """Resolved model id (filled by ``health()`` at startup). Falls
        back to a sensible placeholder if ``health()`` was never called."""

        return self._resolved_model or "nemotron-3-super"

    async def health(self) -> dict[str, Any]:
        """GET ``/v1/models``. Caches the first model id as ``self.model``.

        Raises :class:`DmClientError` on any transport / status error.
        """

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as c:
                response = await c.get(f"{self._base_url}/models")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DmClientError(f"vLLM health check failed: {exc}") from exc

        models = payload.get("data") or []
        if not models:
            raise DmClientError(f"vLLM at {self._base_url} returned no models in /v1/models")
        first = models[0]
        self._resolved_model = first["id"]
        # The spec mentions "nemotron-3-super" by name; the actual served id
        # is whatever the operator launched the engine with (e.g.
        # ``nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4``). Log both the id
        # and the context window so a misconfigured deployment is obvious in
        # the boot log.
        max_len = first.get("max_model_len")
        log.info(
            "vLLM resolved model: %s (max_model_len=%s)",
            self._resolved_model,
            max_len if max_len is not None else "unknown",
        )
        result: dict[str, Any] = payload
        return result

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        reasoning_mode: ReasoningMode = "full",
    ) -> str:
        """Non-streaming convenience for archival / structured calls.

        Used by the summariser (session + campaign rolling summaries)
        and the world-fact extractor. These callers want a single string
        back rather than the streaming chunk sequence ``stream_dm``
        returns. Temperature defaults to 0.3 because the work is
        structured / archival, not creative DM narration (which uses
        0.85).

        ``response_format`` is forwarded as-is — pass
        ``{"type": "json_object"}`` to ask the server for guaranteed-
        parseable JSON when the underlying engine supports it. Nemotron
        on vLLM does, but the fact extractor still strips fenced code
        blocks defensively because the parser sometimes wraps even with
        ``json_object`` set.

        ``reasoning_mode`` selects Nemotron's chat-template effort knob;
        see :data:`ReasoningMode`. Default ``"full"`` so an unaware caller
        gets the safe behaviour. Compression-style work (summarisers,
        fact extractor) overrides to ``"low"``.

        Reuses the same ``self._client``; no separate transport.
        """

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        extra_body = _reasoning_extra_body(reasoning_mode)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # openai exceptions don't share a clean base
            raise DmClientError(f"chat.completions.create failed: {exc}") from exc

        if not response.choices:
            raise DmClientError("chat.completions.create returned no choices")
        # Surface usage at debug level — the per-call-site reasoning_mode
        # tuning only pays off if the savings are visible. Field is
        # optional in the OpenAI response shape; guard accordingly.
        usage = getattr(response, "usage", None)
        if usage is not None:
            log.debug(
                "complete: reasoning_mode=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                reasoning_mode,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                getattr(usage, "total_tokens", "?"),
            )
        content = response.choices[0].message.content
        return content or ""

    async def stream_dm(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Literal["auto", "none", "required"] | None = None,
        temperature: float = 0.85,
        max_tokens: int = 1024,
        reasoning_mode: ReasoningMode = "full",
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream a DM response. Wraps the runaway-token detector.

        Yields raw OpenAI chunks; the orchestrator is responsible for
        accumulating ``delta.content`` and ``delta.tool_calls`` into the
        final message.

        ``reasoning_mode`` defaults to ``"full"`` — the DM turn loop's
        tool-call accuracy depends on the full reasoning trace and is the
        canonical caller. No current call site overrides this; future
        Phase 8 module extractor stays at ``"full"`` for the same reason.
        """

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools is not None:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        extra_body = _reasoning_extra_body(reasoning_mode)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # openai exceptions don't share a clean base
            raise DmClientError(f"chat.completions.create failed: {exc}") from exc

        return _watch_for_runaways(stream)

    async def aclose(self) -> None:
        """Dispose of the underlying transport. Called from app lifespan."""

        await self._client.close()


async def _watch_for_runaways(
    stream: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[ChatCompletionChunk]:
    """Wrap a chunk stream and abort on >50 consecutive identical content
    tokens (the qwen3_coder repeating-token failure mode). The detector
    only inspects ``delta.content`` — tool-call chunks are short and
    their content slot is empty, so they pass through unaffected.
    """

    last_token: str | None = None
    repeat_count = 0
    async for chunk in stream:
        if not chunk.choices:
            yield chunk
            continue
        delta = chunk.choices[0].delta
        token = (delta.content or "").strip()
        if token and token == last_token:
            repeat_count += 1
            if repeat_count > _RUNAWAY_THRESHOLD:
                raise RunawayTokenError(
                    f"runaway-token detector tripped: token {token!r} repeated"
                    f" {repeat_count} times in a row"
                )
        else:
            last_token = token if token else last_token
            repeat_count = 0 if token else repeat_count
        yield chunk


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_singleton: DmClient | None = None


def get_dm_client() -> DmClient:
    """Return the process-wide :class:`DmClient`, building on first call.

    The lifespan hook in ``app.main`` calls ``health()`` once at startup
    (and disposes via ``aclose()`` on shutdown).
    """

    global _singleton
    if _singleton is None:
        _singleton = DmClient()
    return _singleton


async def reset_for_tests() -> None:
    """Tear down the singleton — used by tests that swap the endpoint."""

    global _singleton
    if _singleton is not None:
        await _singleton.aclose()
        _singleton = None


__all__ = [
    "DmClient",
    "DmClientError",
    "ReasoningMode",
    "RunawayTokenError",
    "get_dm_client",
    "reset_for_tests",
]
