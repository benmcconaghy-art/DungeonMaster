"""Generated-image serving endpoint (Phase 5 close-out follow-up).

The Phase 5 plan in spec §8 imagined an ``X-Accel-Redirect`` flow:
nginx serves the bytes from an ``internal`` location after FastAPI
authorises the caller. The handler that sets ``X-Accel-Redirect``
was never written, so every ``<img src="/api/images/{id}.png">``
in the templates 404'd through the playthrough.

This module ships the missing route as a plain ``FileResponse``
(via Starlette → Linux ``sendfile()``) rather than wiring the
X-Accel-Redirect dance. Reasons:

* Single-gunicorn-worker deployment with 2-4 concurrent players
  (spec §13). ``sendfile`` is kernel-space; the Python overhead
  is one ``os.stat`` and a Cache-Control header.
* Avoids the dev/prod split — X-Accel-Redirect requires nginx
  in front; FileResponse works identically in both.
* Keeps authorization and serving in one place. The nginx
  ``location /images/`` block can stay (or be removed) without
  affecting correctness.

If image-serving ever becomes a measured bottleneck the optimization
is to add ``X-Accel-Redirect`` here and let nginx ``sendfile`` to
the kernel-buffered FD. For now: do less.

Authorization: caller must be a member of the campaign that owns
the ``generated_images`` row. NPCs that haven't been encountered yet
have canonical portraits the players shouldn't see — campaign-
membership is the right granularity (per spec §8).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.config import get_settings
from app.db import models
from app.deps import CurrentUser, DbSession

router = APIRouter(prefix="/api", tags=["images"])
log = logging.getLogger(__name__)

# Long-lived because the bytes never change once written. ``private``
# rather than ``public`` because the resource is auth-gated — proxies
# / CDNs (nginx in front, future caches) must not serve it to a
# different user. ``immutable`` tells modern browsers they can skip
# revalidation entirely.
_CACHE_CONTROL = "private, max-age=86400, immutable"


@router.get("/images/{image_id}.png")
async def get_image(
    image_id: str,
    user: CurrentUser,
    db: DbSession,
) -> FileResponse:
    """Serve a generated PNG to a campaign member.

    Returns 404 (not 403) for any failure — unknown id, missing
    file on disk, or non-member — so a probe can't distinguish
    "image exists but you can't see it" from "no such image".
    Path-traversal in ``image_id`` is moot: we never touch
    ``image_id`` as a filesystem fragment. The disk path comes
    from the trusted ``GeneratedImage.file_path`` column written
    by the worker, and we still verify it resolves under
    ``image_storage_path`` as defence in depth.
    """

    image = await db.get(models.GeneratedImage, image_id)
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="image not found")

    membership = await db.get(models.CampaignMember, (image.campaign_id, user.id))
    if membership is None:
        # Don't leak existence — same 404 the unknown-id path uses.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="image not found")

    settings = get_settings()
    storage_root = settings.image_storage_path.resolve()
    file_path = Path(image.file_path).resolve()

    # Defence in depth: the worker always writes
    # ``<image_storage_path>/<id>.png`` so a resolved path outside
    # the storage root would be a row-level corruption, not a
    # routing one. 404 keeps the leak surface flat.
    try:
        file_path.relative_to(storage_root)
    except ValueError:
        log.error(
            "get_image: file_path %s outside image_storage_path %s for image %s",
            file_path,
            storage_root,
            image_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="image not found"
        ) from None

    if not file_path.is_file():
        # The DB row exists but the bytes don't — orphaned row, or
        # the operator wiped /var/lib/dungeon-master/images/ without
        # truncating the table. 404 rather than 500 because the
        # caller can't do anything about it; the operator sees this
        # in the access log.
        log.warning(
            "get_image: row %s references missing file %s",
            image_id,
            file_path,
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="image not found")

    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        headers={"Cache-Control": _CACHE_CONTROL},
    )


__all__ = ["router"]
