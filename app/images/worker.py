"""Image worker — async queue consumer.

Single-concurrency worker that pops image jobs off Redis (`images:queue`),
hashes the request to dedupe against ``generated_images.prompt_hash``,
calls the FLUX client (``/generate`` for txt2img, ``/edit`` for Kontext
character / scene edits), persists the resulting PNG under
``/var/lib/dungeon-master/images/<uuid>.png`` and a row in
``generated_images``, then publishes ``image:ready:<id>`` on Redis pub/sub
so the session WS hub can broadcast ``image_ready``.

Runs as its own systemd unit (``dungeon-master-imageworker.service``).
Phase 5 implements this; Phase 0 ships only the placeholder.
"""

from __future__ import annotations
