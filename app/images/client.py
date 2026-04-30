"""FLUX HTTP client.

Async wrapper around the FLUX.1 [dev] / FLUX.1 Kontext [dev] service running
on ``svrai01:11437``. Exposes ``generate`` (text-to-image) and ``edit``
(instruction-based edit of an existing image, used for character /
NPC consistency). Generous timeouts because cold pipeline load + 28-step
inference can run to a minute (spec §8).

Phase 5 implements this; Phase 0 ships only the placeholder.
"""

from __future__ import annotations
