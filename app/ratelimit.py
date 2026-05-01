"""Rate limiting on auth-shaped endpoints.

Phase 7 hardening (spec §13). The trusted-LAN deployment threat model is
"slow accidents," not "skilled attacker": a misconfigured client retry
loop, a player mashing the login button thinking the page froze. The
limits set here are deliberately generous — set high enough that no
legitimate human use trips them, low enough that runaway loops do.

Storage backend: Valkey at ``settings.redis_url``. Counters live in the
same KV store as pubsub / image queue, so there's no second daemon to
operate. Tests override via ``RATELIMIT_STORAGE_URI=memory://``.

Limits applied (per IP):
- POST /api/auth/login         60/minute   (one per second sustained)
- POST /api/auth/register      30/minute, 200/hour
- POST /api/campaigns/join     60/minute

Game-action endpoints (pc_action over WS, dice rolls, image generation
requests) are intentionally NOT rate-limited — the LLM and FLUX latency
is already the natural throttle.

Implementation note: this module uses the ``limits`` library directly
rather than slowapi's ``@limiter.limit`` decorator. slowapi's wrapper
re-binds the route function's ``__globals__`` to slowapi's module, which
breaks FastAPI's resolution of ``from __future__ import annotations``
ForwardRefs (``RegisterRequest``, ``DbSession``) and produces 422 errors
on the body parameters. A FastAPI ``Depends`` sidesteps that — the route
function's signature stays untouched.
"""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Request, status
from limits import RateLimitItem, parse
from limits.aio.strategies import FixedWindowRateLimiter
from limits.storage import storage_from_string

from app.config import get_settings

log = logging.getLogger(__name__)


# Limit strings — single source of truth for the route dependencies and
# any future code that wants to advertise the policy. Format is the
# slowapi/limits DSL. Calibration per the brief: high enough that a
# legitimate human never trips, low enough to slow runaway loops.
LOGIN_LIMIT = "60/minute"
REGISTER_LIMIT_MINUTE = "30/minute"
REGISTER_LIMIT_HOUR = "200/hour"
JOIN_LIMIT = "60/minute"

# Human message returned in the 429 body. Per the brief, the user-facing
# experience must be a clear sentence, not the raw limits-DSL string.
HUMAN_MESSAGE = "You're trying that too quickly — please wait a moment before retrying."


# Module-level state. Built lazily so tests can flip
# RATELIMIT_STORAGE_URI between cases via ``reset_for_tests``.
_limiter: FixedWindowRateLimiter | None = None


def _resolve_storage_uri() -> str:
    """Pick the storage URI for the rate-limit counters.

    Production: the configured ``redis_url`` (Valkey on localhost). Tests:
    ``memory://`` via the env override so unit tests don't need a real
    Valkey daemon. Setting ``RATELIMIT_STORAGE_URI`` in the systemd unit
    is also an escape hatch if an operator wants to point counters at a
    separate Valkey database.
    """

    override = os.environ.get("RATELIMIT_STORAGE_URI")
    if override:
        return override
    return get_settings().redis_url


def _ensure_limiter() -> FixedWindowRateLimiter:
    """Build (or reuse) the module-level RateLimiter.

    The ``async+`` prefix on the storage URI selects the asyncio
    backend variant — required so ``limiter.hit(...)`` returns an
    awaitable rather than a sync result.
    """

    global _limiter
    if _limiter is None:
        uri = _resolve_storage_uri()
        if not uri.startswith("async+"):
            uri = f"async+{uri}"
        storage = storage_from_string(uri)
        _limiter = FixedWindowRateLimiter(storage)
    return _limiter


def reset_for_tests() -> None:
    """Drop the cached limiter so the next call rebuilds it.

    Tests call this in an autouse fixture so each case starts with
    zero counter state. Without it, the same testserver IP would
    accumulate hits across tests and the limit checks would
    spuriously fire.
    """

    global _limiter
    _limiter = None


def _client_ip(request: Request) -> str:
    """Resolve the per-IP key for rate limiting.

    The trusted-LAN deployment puts nginx in front of the app, so
    ``X-Forwarded-For`` is set to the originating IP. Trust it here —
    if a misbehaving client were spoofing it on the LAN, we'd have
    bigger problems than rate limiting (and the trusted-LAN posture
    explicitly accepts that). Fall back to ``request.client.host`` for
    the no-proxy / dev path.
    """

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # XFF can be a comma-separated chain; the first entry is the
        # originating client per RFC 7239 conventions.
        return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


async def _enforce(request: Request, *limits: RateLimitItem) -> None:
    """Hit each limit; raise 429 with a clear message if any is exhausted.

    All limits are checked. The first one that's already at its cap
    determines the Retry-After header value — that's the one the client
    needs to wait out anyway. Successful checks consume one slot per
    limit; the trade-off is that a legitimate request "uses" both the
    per-minute and per-hour budget on the register endpoint, which is
    the intended semantics (one register attempt = one ticket against
    each window).
    """

    limiter = _ensure_limiter()
    ip = _client_ip(request)
    for item in limits:
        ok = await limiter.hit(item, ip)
        if not ok:
            # Window length in seconds, used to compute Retry-After.
            # GRANULARITY is a (seconds, name) NamedTuple on the limits
            # RateLimitItem class.
            window_seconds = item.GRANULARITY.seconds
            log.info(
                "rate limit tripped",
                extra={
                    "ip": ip,
                    "limit": str(item),
                    "path": request.url.path,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=HUMAN_MESSAGE,
                headers={"Retry-After": str(int(window_seconds))},
            )


# ---------------------------------------------------------------------------
# FastAPI dependencies, one per endpoint
# ---------------------------------------------------------------------------
#
# Pre-parse the limit items at import time. They're plain value objects;
# parsing is cheap but doing it once keeps the per-request hot path tight.
_LOGIN_ITEM = parse(LOGIN_LIMIT)
_REGISTER_ITEMS = (parse(REGISTER_LIMIT_MINUTE), parse(REGISTER_LIMIT_HOUR))
_JOIN_ITEM = parse(JOIN_LIMIT)


async def login_rate_limit(request: Request) -> None:
    """Apply the login endpoint's per-IP limit."""

    await _enforce(request, _LOGIN_ITEM)


async def register_rate_limit(request: Request) -> None:
    """Apply the register endpoint's per-IP limits (minute + hour)."""

    await _enforce(request, *_REGISTER_ITEMS)


async def join_rate_limit(request: Request) -> None:
    """Apply the campaign-join endpoint's per-IP limit."""

    await _enforce(request, _JOIN_ITEM)


__all__ = [
    "HUMAN_MESSAGE",
    "JOIN_LIMIT",
    "LOGIN_LIMIT",
    "REGISTER_LIMIT_HOUR",
    "REGISTER_LIMIT_MINUTE",
    "join_rate_limit",
    "login_rate_limit",
    "register_rate_limit",
    "reset_for_tests",
]
