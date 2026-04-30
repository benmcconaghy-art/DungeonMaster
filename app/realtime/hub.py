"""WebSocket session hub.

Authorises connections to ``/ws/session/{session_id}`` (campaign membership
check), subscribes them to the per-session Redis pub/sub channel, sends an
initial state snapshot, and fans out ``narration_chunk``,
``narration_complete``, ``dice_roll``, ``state_update``, ``image_ready``,
``whisper``, and ``presence`` messages to the connected clients.

Phase 4 implements this; Phase 0 ships only the placeholder.
"""

from __future__ import annotations
