"""SQLAlchemy ORM models.

Every persistent entity in the system maps to a class defined here:
``users``, ``campaigns``, ``characters``, ``sessions``, ``session_messages``,
``world_facts``, ``generated_images``, ``modules``, and the rest of the
schema laid out in ``dungeon-master-spec.md`` §5.

Phase 0 deliberately leaves this empty — the first model lands in Phase 1
when the BFRPG engine starts persisting characters.
"""

from __future__ import annotations
