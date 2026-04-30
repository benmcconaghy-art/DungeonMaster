"""FastAPI dependency providers.

Holds the shared dependencies injected into HTTP handlers and WebSocket
endpoints — the database session, the current authenticated user, and any
per-request collaborators that need the FastAPI dependency-injection
lifecycle.

Implementations land in Phase 1 (database session) and Phase 1 / 2
(current user, once auth is in place).
"""

from __future__ import annotations
