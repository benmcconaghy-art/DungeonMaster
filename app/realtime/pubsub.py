"""Redis pub/sub adapter for cross-process fan-out.

Wraps the redis-py async client and exposes session-scoped publish /
subscribe primitives used by the WS hub and the image worker. Channel
naming: ``session:{session_id}`` for narration / state, ``image:ready:{id}``
for image-ready notifications.
"""

from __future__ import annotations
