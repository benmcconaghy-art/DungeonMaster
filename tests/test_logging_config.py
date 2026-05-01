"""Phase 7 structured-logging tests.

Three contracts under test:

* ``JsonFormatter`` produces parseable single-line JSON with the
  canonical keys (ts, level, logger, msg, request_id, user_id).
* ``request_id_var`` and ``user_id_var`` propagate via contextvars
  so log lines emitted from inside a ``bind_request_context`` block
  carry the right ids.
* ``RequestIdMiddleware`` round-trips an inbound ``X-Request-ID``
  header and synthesises one when absent. ``AccessLogMiddleware``
  emits one structured record per request including
  ``request_id``, ``method``, ``path``, ``status``, ``duration_ms``.
"""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest
from httpx import AsyncClient

from app.logging_config import (
    JsonFormatter,
    bind_request_context,
    configure_logging,
    set_user_id,
    user_id_var,
)

# ---------------------------------------------------------------------------
# JsonFormatter shape
# ---------------------------------------------------------------------------


def test_json_formatter_renders_canonical_fields() -> None:
    """A vanilla log record renders with ts/level/logger/msg + null
    request_id and user_id when no context is bound."""

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    line = formatter.format(record)
    payload = json.loads(line)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["msg"] == "hello world"
    assert payload["request_id"] is None
    assert payload["user_id"] is None
    assert "ts" in payload and payload["ts"].endswith("Z")


def test_json_formatter_promotes_extras_to_top_level() -> None:
    """Anything passed via ``logger.info(..., extra={...})`` lands at
    the top level alongside the canonical fields."""

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="llm complete",
        args=(),
        exc_info=None,
    )
    record.model = "nemotron-3-super"
    record.prompt_tokens = 1234
    record.latency_ms = 567

    payload = json.loads(formatter.format(record))

    assert payload["model"] == "nemotron-3-super"
    assert payload["prompt_tokens"] == 1234
    assert payload["latency_ms"] == 567


def test_json_formatter_does_not_let_extras_shadow_canonical_fields() -> None:
    """If an extra collides with ``ts`` / ``level`` / ``request_id`` /
    etc., the formatter's own value wins so the output shape is
    stable across call sites."""

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    record.request_id = "should-not-appear"
    record.level = "DEBUG"

    payload = json.loads(formatter.format(record))
    assert payload["request_id"] is None  # contextvar default wins
    assert payload["level"] == "INFO"  # record.levelname wins


def test_json_formatter_renders_exc_info() -> None:
    """An exception attached to the record produces a string traceback
    under ``exc_info``."""

    formatter = JsonFormatter()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            name="app.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="kaboom",
            args=(),
            exc_info=sys.exc_info(),
        )

    payload = json.loads(formatter.format(record))
    assert "RuntimeError: boom" in payload["exc_info"]


# ---------------------------------------------------------------------------
# Contextvars
# ---------------------------------------------------------------------------


def test_bind_request_context_propagates_to_log_records() -> None:
    """Inside the ``bind_request_context`` block the contextvars carry
    the bound ids; outside they're null again."""

    formatter = JsonFormatter()
    record_template = {
        "name": "app.test",
        "level": logging.INFO,
        "pathname": __file__,
        "lineno": 1,
        "msg": "msg",
        "args": (),
        "exc_info": None,
    }

    with bind_request_context("req-abc", "user-123"):
        rec = logging.LogRecord(**record_template)
        inside = json.loads(formatter.format(rec))
        assert inside["request_id"] == "req-abc"
        assert inside["user_id"] == "user-123"

    rec = logging.LogRecord(**record_template)
    outside = json.loads(formatter.format(rec))
    assert outside["request_id"] is None
    assert outside["user_id"] is None


def test_set_user_id_updates_contextvar() -> None:
    """``set_user_id`` works without a surrounding ``bind_request_context``
    so the auth dep can lazily attach the user once it resolves."""

    assert user_id_var.get() is None
    token = user_id_var.set(None)
    try:
        set_user_id("late-bound-user")
        assert user_id_var.get() == "late-bound-user"
    finally:
        user_id_var.reset(token)


# ---------------------------------------------------------------------------
# configure_logging behaviour
# ---------------------------------------------------------------------------


def test_configure_logging_replaces_existing_handlers() -> None:
    """Calling configure_logging twice doesn't double up the output."""

    root = logging.getLogger()
    # Add two arbitrary handlers, then reconfigure: should end at 1.
    root.addHandler(logging.NullHandler())
    root.addHandler(logging.NullHandler())

    configure_logging()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)


def test_configure_logging_emits_json_to_handler_stream() -> None:
    """Wire a StringIO into a fresh JsonFormatter and confirm the
    rendered line round-trips through json.loads."""

    formatter = JsonFormatter()
    sink = StringIO()
    handler = logging.StreamHandler(sink)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    test_logger = logging.getLogger("app.test_sink")
    test_logger.handlers.clear()
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)
    test_logger.propagate = False

    test_logger.info("hello", extra={"path": "/api/foo"})
    line = sink.getvalue().strip()
    payload = json.loads(line)
    assert payload["msg"] == "hello"
    assert payload["path"] == "/api/foo"


# ---------------------------------------------------------------------------
# Middleware integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_id_synthesised_when_absent(client: AsyncClient) -> None:
    """A request without an inbound X-Request-ID gets a generated one
    on the response."""

    r = await client.get("/health")
    assert r.status_code == 200
    rid = r.headers.get("x-request-id")
    assert rid is not None
    assert 1 < len(rid) <= 128


@pytest.mark.asyncio
async def test_request_id_propagates_inbound_header(client: AsyncClient) -> None:
    """An inbound X-Request-ID is honoured verbatim on the response so
    a load balancer can stitch traces."""

    r = await client.get("/health", headers={"X-Request-ID": "trace-from-lb-12345"})
    assert r.status_code == 200
    assert r.headers.get("x-request-id") == "trace-from-lb-12345"


@pytest.mark.asyncio
async def test_request_id_rejects_oversized_inbound(client: AsyncClient) -> None:
    """A 200-byte X-Request-ID is dropped — the middleware synthesises
    a fresh one rather than letting the client pollute every log line."""

    too_long = "x" * 200
    r = await client.get("/health", headers={"X-Request-ID": too_long})
    assert r.status_code == 200
    assert r.headers["x-request-id"] != too_long


@pytest.mark.asyncio
async def test_access_log_emits_one_structured_record_per_request(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The AccessLogMiddleware emits exactly one INFO record per
    request with the canonical fields."""

    caplog.set_level(logging.INFO, logger="app.access")

    r = await client.get("/health", headers={"X-Request-ID": "trace-access-log"})
    assert r.status_code == 200

    access_records = [rec for rec in caplog.records if rec.name == "app.access"]
    assert len(access_records) == 1
    record = access_records[0]
    assert record.method == "GET"
    assert record.path == "/health"
    assert record.status == 200
    assert isinstance(record.duration_ms, int)
    assert record.duration_ms >= 0


# ---------------------------------------------------------------------------
# Regression: request_id propagation across async-task boundaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_id_propagates_into_fire_and_forget_create_task() -> None:
    """``asyncio.create_task`` copies the parent's contextvars (PEP 567).
    The orchestrator's post-turn fact-extractor and session-summary
    tasks are spawned via ``asyncio.create_task``, so request_id set
    by the request middleware must be visible inside the spawned
    coroutine. Locking this in: a future refactor that switched to
    ``loop.create_task`` (which copies from the loop's default
    context, not the caller's) or to a thread pool without
    ``contextvars.copy_context`` would silently break the
    correlation. This test catches that regression.
    """

    import asyncio

    from app.logging_config import bind_request_context, request_id_var

    captured: dict[str, str | None] = {}

    async def _spawned() -> None:
        # Read the contextvar from inside the fire-and-forget task.
        captured["from_task"] = request_id_var.get()

    with bind_request_context("rid-propagation-check"):
        task = asyncio.create_task(_spawned())
        await task

    assert captured["from_task"] == "rid-propagation-check"


@pytest.mark.asyncio
async def test_request_id_propagates_through_asyncio_to_thread() -> None:
    """``asyncio.to_thread`` (Python 3.9+) wraps the callable in
    ``contextvars.copy_context().run(...)`` so the caller's
    contextvars are visible inside the threaded callable. The
    embedding backend uses ``asyncio.to_thread`` for the
    sentence-transformers ``.encode`` hop; if a future Python or
    refactor regressed this, log lines from inside the threaded
    callable would lose the request_id correlation."""

    import asyncio

    from app.logging_config import bind_request_context, request_id_var

    def _read_contextvar() -> str | None:
        return request_id_var.get()

    with bind_request_context("rid-thread-hop"):
        result = await asyncio.to_thread(_read_contextvar)

    assert result == "rid-thread-hop"
