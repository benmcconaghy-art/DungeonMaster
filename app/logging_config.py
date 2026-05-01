"""Structured JSON logging configuration (Phase 7 hardening).

Stdlib ``logging`` with a custom JSON formatter — no ``structlog`` dep,
which keeps the dependency surface lean. The structured records carry
enough context that an operator can correlate a player's HTTP turn with
its LLM calls and tool dispatches across log streams.

Per-record shape (one JSON object per line, NDJSON-friendly):
- ``ts``: ISO-8601 UTC timestamp with millisecond precision
- ``level``: uppercase severity name
- ``logger``: dotted module name (``app.api.ws`` etc.)
- ``msg``: the formatted log message
- ``request_id``: from a contextvar set by the request-id middleware;
  null when the log line isn't on a request hot path (background
  workers, lifespan setup).
- ``user_id``: from the same contextvar; populated when the auth
  middleware has resolved a user.
- Any keys passed via ``logger.info("...", extra={"k": v})`` land at
  the top level alongside the canonical fields. Reserved keys (the ones
  above plus standard logging dunders) are not overwritten — name
  collisions are silently dropped to keep the JSON shape stable.
- ``exc_info``: rendered traceback string when present.

The configuration is idempotent: calling ``configure_logging`` twice
clears the existing handlers and re-installs them. Tests reset state by
calling it again with their preferred level.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

# Reserved logging-record attributes — anything matching this set is the
# stdlib's own bookkeeping; we either map it explicitly into the JSON
# shape or skip it. Drawn from the Python docs' LogRecord attribute list.
_RESERVED_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

# JSON-output keys we always set ourselves; passing the same key via
# ``extra=`` should NOT be allowed to overwrite them, which would make
# the log shape unstable across call sites.
_FIXED_OUTPUT_KEYS: Final[frozenset[str]] = frozenset(
    {"ts", "level", "logger", "msg", "request_id", "user_id", "exc_info"}
)


# Context variables. Set by the request-id middleware on entry, cleared
# on exit. Any code anywhere on the same async task picks them up via
# the formatter — no need to thread an explicit ``request_id=`` kwarg
# through every log call.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_id", default=None)


@contextmanager
def bind_request_context(request_id: str | None, user_id: str | None = None) -> Iterator[None]:
    """Temporarily bind ``request_id`` (and optionally ``user_id``) to
    the current context. Used by the middleware and by background
    coroutines that want their log lines correlated with a triggering
    HTTP turn (e.g. the orchestrator's post-turn fact-extractor task).
    """

    rid_token = request_id_var.set(request_id)
    uid_token = user_id_var.set(user_id) if user_id is not None else None
    try:
        yield
    finally:
        request_id_var.reset(rid_token)
        if uid_token is not None:
            user_id_var.reset(uid_token)


def set_user_id(user_id: str | None) -> None:
    """Update the per-request user id once the auth middleware has
    resolved it. Called from ``app.deps.require_user`` so subsequent
    logs on the same request carry the user id even if the request-id
    middleware ran before auth.
    """

    user_id_var.set(user_id)


class JsonFormatter(logging.Formatter):
    """Render a ``LogRecord`` as a single line of JSON.

    Optimised for log-shipper consumption (Loki, Vector, journald) and
    for grep-by-request_id workflows. Pretty-printing is intentionally
    not supported — a single line per record keeps the parsers happy.
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S")
        # formatTime returns seconds; append the milliseconds + Z to
        # match the spec §5 timestamp shape (ISO-8601 UTC).
        ts_full = f"{ts}.{int(record.msecs):03d}Z"

        payload: dict[str, Any] = {
            "ts": ts_full,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
        }

        # Promote anything that was passed via ``extra={...}`` to the
        # top level. We can detect those by ranging over the record's
        # __dict__ and skipping the reserved logging attributes.
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS:
                continue
            if key in _FIXED_OUTPUT_KEYS:
                # Don't let extras shadow our canonical fields.
                continue
            payload[key] = _safe_value(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # Last-ditch fallback: drop the non-serialisable extras and
            # ship the canonical shape. Better a degraded log line than
            # a swallowed error.
            return json.dumps(
                {
                    "ts": ts_full,
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                    "request_id": request_id_var.get(),
                    "user_id": user_id_var.get(),
                    "_unserialisable_extras": True,
                },
                ensure_ascii=False,
            )


def _safe_value(value: Any) -> Any:
    """Coerce a structured-log extra into something json.dumps will
    accept without help. Most things pass through; weirdness like
    ``Path``, ``UUID``, datetime get stringified by ``default=str``."""

    return value


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: removes existing handlers before adding the new one so
    repeated calls (e.g. test setup) don't duplicate output. Writes to
    stderr — systemd captures stderr to journald automatically, and a
    syslog redirect lands the lines in /var/log/dungeon-master/app.log
    via the rsyslog config bundled with the deploy. Image worker has
    its own logger configuration; see ``app.images.worker``.
    """

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


__all__ = [
    "JsonFormatter",
    "bind_request_context",
    "configure_logging",
    "request_id_var",
    "set_user_id",
    "user_id_var",
]
