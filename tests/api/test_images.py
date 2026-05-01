"""Tests for ``GET /api/images/{id}.png`` (Phase 5 close-out).

Real-play surfaced that this route 404'd because Phase 5 planned
an X-Accel-Redirect flow but never wired the FastAPI handler. The
fix ships ``FileResponse`` with campaign-membership authorization;
these tests pin both the happy path and the leak surface
(unknown ids and non-member viewers both return 404 — never 403 —
so a probe can't distinguish "exists but you can't see" from
"doesn't exist").
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

from app.config import get_settings
from app.db import models
from app.deps import get_db

_VALID_PW = "correct horse battery staple"

# Tiny valid PNG (1x1, transparent). Decoded once as bytes so the test
# can write it to disk without an extra dependency.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


async def _register_and_login(client: AsyncClient, username: str = "alice") -> str:
    response = await client.post(
        "/api/auth/register",
        json={"username": username, "password": _VALID_PW},
    )
    assert response.status_code == 201
    return response.json()["id"]  # type: ignore[no-any-return]


async def _create_campaign(client: AsyncClient, name: str = "Borderlands") -> str:
    response = await client.post("/api/campaigns", json={"name": name})
    assert response.status_code == 201
    return response.json()["id"]  # type: ignore[no-any-return]


async def _seed_image(
    client: AsyncClient,
    *,
    campaign_id: str,
    image_id: str,
    storage_dir: Path,
    write_bytes: bool = True,
) -> Path:
    """Insert a ``generated_images`` row pointing at a file we control,
    optionally writing the bytes to disk. Returns the file path."""

    file_path = storage_dir / f"{image_id}.png"
    if write_bytes:
        file_path.write_bytes(_PNG_BYTES)

    # Drive DB writes through the same dependency-overridden session
    # the AsyncClient uses, so the row commits to the same in-memory
    # engine the request handler will read from.
    from app.main import app as fastapi_app

    db_factory = fastapi_app.dependency_overrides[get_db]
    async for db in db_factory():  # type: ignore[union-attr]
        row = models.GeneratedImage(
            id=image_id,
            campaign_id=campaign_id,
            kind="npc",
            prompt="test",
            prompt_hash=f"hash-{image_id}",
            file_path=str(file_path),
        )
        db.add(row)
        await db.commit()
        break
    return file_path


@pytest.fixture
def storage_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the image_storage_path setting at a per-test tmpdir so
    the production /var/lib path is never touched. The setting is read
    inside the handler via ``get_settings()`` so monkeypatching the
    cached singleton's attribute is enough."""

    settings = get_settings()
    monkeypatch.setattr(settings, "image_storage_path", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_image_serves_png_to_member(client: AsyncClient, storage_dir: Path) -> None:
    """A campaign member fetching a known image gets the bytes back
    with image/png content type and a private cache header."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client)
    image_id = "069f0000-0000-7000-8000-000000000001"
    await _seed_image(client, campaign_id=campaign_id, image_id=image_id, storage_dir=storage_dir)

    response = await client.get(f"/api/images/{image_id}.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "private" in response.headers["cache-control"]
    assert "max-age" in response.headers["cache-control"]
    assert response.content == _PNG_BYTES


# ---------------------------------------------------------------------------
# Unknown / missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_image_unknown_id_returns_404(client: AsyncClient, storage_dir: Path) -> None:
    """Unknown image id → 404; no leak via 403 or 500."""

    await _register_and_login(client)
    response = await client.get("/api/images/does-not-exist.png")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_image_row_exists_but_file_missing_returns_404(
    client: AsyncClient, storage_dir: Path
) -> None:
    """Orphaned DB row (file wiped from disk) — 404 rather than 500.
    The operator sees this in the access log + the warning we emit;
    the player just sees a broken image and reloads."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client)
    image_id = "069f0000-0000-7000-8000-000000000002"
    await _seed_image(
        client,
        campaign_id=campaign_id,
        image_id=image_id,
        storage_dir=storage_dir,
        write_bytes=False,
    )

    response = await client.get(f"/api/images/{image_id}.png")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_image_requires_auth(client: AsyncClient, storage_dir: Path) -> None:
    response = await client.get("/api/images/anything.png")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_image_non_member_returns_404(client: AsyncClient, storage_dir: Path) -> None:
    """A user who isn't a campaign member gets 404 (not 403) so they
    can't probe whether the image exists. Spec §8: only campaign
    members see a campaign's images — canonical NPC portraits could
    spoil things otherwise."""

    await _register_and_login(client, "alice")
    campaign_id = await _create_campaign(client, "Alice's table")
    image_id = "069f0000-0000-7000-8000-000000000003"
    await _seed_image(client, campaign_id=campaign_id, image_id=image_id, storage_dir=storage_dir)
    await client.post("/api/auth/logout")

    await _register_and_login(client, "mallory")
    response = await client.get(f"/api/images/{image_id}.png")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Path traversal — defence in depth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_image_traversal_in_id_does_not_match(
    client: AsyncClient, storage_dir: Path
) -> None:
    """``..`` in the URL part can't escape the storage directory:
    the handler looks the id up in ``generated_images`` and hits 404
    on miss. The disk path is taken from ``row.file_path``, never
    constructed from the URL segment."""

    await _register_and_login(client)
    response = await client.get("/api/images/..%2F..%2Fetc%2Fpasswd.png")
    # FastAPI may decode this differently across versions; either a
    # 404 from no-such-row or a 4xx routing rejection is acceptable.
    # What's NOT acceptable is reading bytes off /etc/passwd.
    assert 400 <= response.status_code < 500
    if response.status_code == 200:  # pragma: no cover — would be the bug
        pytest.fail("traversal attempt returned 200 — file was served")


@pytest.mark.asyncio
async def test_get_image_row_file_path_outside_storage_returns_404(
    client: AsyncClient, storage_dir: Path, tmp_path: Path
) -> None:
    """If a corrupted ``generated_images.file_path`` ever pointed at
    ``/etc/passwd`` (which the worker would never write), the handler
    refuses to serve and 404s. Defence in depth — the worker is the
    trusted writer, but a row with an out-of-tree path stays a 404."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client)
    image_id = "069f0000-0000-7000-8000-000000000004"

    # Create a file outside storage_dir, then seed a row whose
    # ``file_path`` points at it.
    rogue = tmp_path / "outside.png"
    rogue.write_bytes(_PNG_BYTES)
    rogue_outside = rogue.resolve()
    # Move outside the per-test storage root.
    out_of_tree = storage_dir.parent / "rogue.png"
    out_of_tree.write_bytes(_PNG_BYTES)
    assert out_of_tree.exists()

    from app.main import app as fastapi_app

    db_factory = fastapi_app.dependency_overrides[get_db]
    async for db in db_factory():  # type: ignore[union-attr]
        row = models.GeneratedImage(
            id=image_id,
            campaign_id=campaign_id,
            kind="npc",
            prompt="rogue",
            prompt_hash=f"hash-{image_id}",
            file_path=str(out_of_tree),
        )
        db.add(row)
        await db.commit()
        break

    response = await client.get(f"/api/images/{image_id}.png")
    assert response.status_code == 404
    assert rogue_outside.exists()  # we didn't delete it
