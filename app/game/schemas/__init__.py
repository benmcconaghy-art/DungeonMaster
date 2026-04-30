"""Pydantic schemas for the YAML content under ``data/bfrpg/``.

Each top-level resource has its own module (``class_def``, ``race_def``,
``spell_def``, ``monster_def``, ``equipment_def``). The schemas use
``ConfigDict(extra='forbid')`` so the CI validator (``app.game.validate_data``)
catches typos and unknown fields the moment a content author introduces
them, rather than silently ignoring them.

The loader modules in ``app.game`` (``classes``, ``races``, ``items``,
``monsters``) consume these schemas and expose lighter dataclasses to
the rest of the engine, keeping Pydantic out of the hot path.
"""

from __future__ import annotations
