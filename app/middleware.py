"""HTTP middleware for request correlation + access logging.

Phase 7 hardening (spec §13). Pieces:

* :class:`RequestIdMiddleware` — assigns a UUIDv7 request id to every
  HTTP request (or accepts an inbound ``X-Request-ID`` so a load-
  balancer can stitch traces). Sets the
  :data:`app.logging_config.request_id_var` contextvar so every log
  line on the request hot path carries the id automatically. Returns
  the id as ``X-Request-ID`` on the response.

* :class:`AccessLogMiddleware` — emits one structured log record per
  request: ``method``, ``path``, ``status``, ``duration_ms``, plus
  ``request_id`` (auto-attached via the contextvar) and ``user_id``
  (set lazily by :func:`app.logging_config.set_user_id` from
  ``app.deps.require_user`` once auth resolves).

Two distinct middlewares (rather than one combined) so the request_id
binding wraps the auth resolution — auth's log lines correlate cleanly
with the rest of the request — while the access log fires after the
response is fully assembled.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from uuid_extensions import uuid7

from app import metrics
from app.logging_config import request_id_var

log = logging.getLogger("app.access")


_REQUEST_ID_HEADER = "x-request-id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assigns / propagates an X-Request-ID per request.

    A client may supply ``X-Request-ID`` already (e.g. a load balancer
    stitching multi-hop traces). If present, it's accepted verbatim;
    otherwise a UUIDv7 is generated. Either way the value is bound to
    the contextvar before downstream middleware / handlers run, and
    written to the response as ``X-Request-ID`` so the caller can
    cross-reference its own log lines with ours.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Accept an inbound header up to a sensible length; reject
        # anything wild rather than letting a 4 KB id pollute every log
        # line on this request.
        inbound = request.headers.get(_REQUEST_ID_HEADER)
        request_id = inbound if inbound and 1 <= len(inbound) <= 128 else str(uuid7())

        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)

        response.headers[_REQUEST_ID_HEADER] = request_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Emit one structured log record per HTTP request.

    Fields: ``method``, ``path``, ``status``, ``duration_ms``,
    ``client``. The contextvar-driven formatter automatically attaches
    ``request_id`` and ``user_id``, so the access record is fully
    self-correlated with everything else logged during the request.

    A 5xx is logged at WARNING; the rest at INFO. We don't log the
    request body or query string — the spec §13 trusted-LAN posture
    accepts opaque bodies, and structured access logs that include
    arbitrary user input are an exfil surface even on a LAN. Path
    template (rather than the resolved URL) keeps cardinality low for
    the metric counterparts.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            duration_s = time.monotonic() - started
            duration_ms = int(duration_s * 1000)
            # Path label is only valid AFTER routing has run (which
            # happens during call_next). On exception, the route may
            # or may not have been matched; ``_path_label`` falls back
            # to ``<unmatched>`` if not.
            path_label = _path_label(request)
            log.exception(
                "request raised",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "client": _client_ip(request),
                },
            )
            metrics.http_request_duration_seconds.labels(
                method=request.method, path=path_label
            ).observe(duration_s)
            metrics.http_requests_total.labels(
                method=request.method, path=path_label, status="500"
            ).inc()
            raise

        duration_s = time.monotonic() - started
        duration_ms = int(duration_s * 1000)
        # Resolve the path label after call_next so ``scope["route"]``
        # has been populated by the router. Reading it before would
        # always give ``<unmatched>``.
        path_label = _path_label(request)
        level = logging.WARNING if response.status_code >= 500 else logging.INFO
        log.log(
            level,
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "client": _client_ip(request),
            },
        )
        metrics.http_request_duration_seconds.labels(
            method=request.method, path=path_label
        ).observe(duration_s)
        metrics.http_requests_total.labels(
            method=request.method,
            path=path_label,
            status=str(response.status_code),
        ).inc()
        return response


def _path_label(request: Request) -> str:
    """Resolve a low-cardinality path label for HTTP metrics.

    Uses Starlette's matched route template (``/api/campaigns/{id}``)
    rather than the resolved URL (``/api/campaigns/abc-123``) so the
    label cardinality stays bounded by the number of routes, not by
    the number of campaigns. Falls back to the literal path when no
    route matched (e.g. a 404 on an undefined URL) — and bins those
    under the ``"<unmatched>"`` label so a flood of 404s on random
    URLs doesn't blow the metric cardinality.
    """

    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if template:
        return str(template)
    if request.scope.get("endpoint") is None:
        return "<unmatched>"
    return request.url.path


def _client_ip(request: Request) -> str:
    """Best-effort source-IP extraction for the access log.

    Same logic as ``app.ratelimit._client_ip`` — XFF first (nginx
    forwards it), then ``request.client.host``, then ``unknown``.
    Duplicated rather than imported so the access log doesn't pull in
    the rate-limit module's import graph.
    """

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


__all__ = ["AccessLogMiddleware", "RequestIdMiddleware"]
