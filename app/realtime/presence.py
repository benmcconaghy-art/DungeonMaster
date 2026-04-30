"""Presence tracking — who is connected to which session right now.

Live in-memory state describing the current set of WebSocket connections per
session. Distinct from ``messages.py`` (which defines message *types*); this
module owns the live *state*. Drives the ``presence`` notifications the hub
broadcasts when players join or leave.

Phase 4 implements this; Phase 0 ships only the placeholder.
"""

from __future__ import annotations
