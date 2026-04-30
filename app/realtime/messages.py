"""WebSocket message types — Pydantic models for what gets sent over the wire.

Server → client messages: ``narration_chunk``, ``narration_complete``,
``pc_action``, ``whisper``, ``dice_roll``, ``state_update``,
``image_pending``, ``image_ready``, ``presence``.

Client → server messages: ``pc_action``, ``whisper_to_dm``,
``out_of_band_chat``, ``ping``.

Each is a discriminated-union variant on a ``type`` field so the hub can
parse and route without ad-hoc dict introspection.
"""

from __future__ import annotations
