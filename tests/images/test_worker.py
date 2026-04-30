"""Tests for ``app.images.worker._process_job`` — the per-job
end-to-end flow with FLUX, DB, and Pubsub at the boundaries.

We exercise ``_process_job`` directly rather than ``main()`` because:

- the BLPOP loop is one line of bookkeeping; ``_process_job`` is
  where the dedup, FLUX dispatch, persistence, FK link, and
  failure-flavour logic live;
- a real Valkey would slow the suite to a crawl and brings no test
  signal that ``_process_job`` doesn't already give us.

Real SQLite (file-backed temp DB) so the unique constraint on
``generated_images.prompt_hash`` is honoured. Fake FluxClient that
records calls; fake Pubsub that captures published messages so we
can assert the right ``image_ready`` / ``image_failed`` flavour.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import models
from app.db.base import Base
from app.db.session import create_engine
from app.images.client import FluxClientError
from app.images.queue import ImageJob
from app.images.worker import (
    _KIND_PARAMS,
    _hash_inputs,
    _process_job,
)
from app.realtime.messages import ImageFailed, ImageReady, ServerMessage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Real file-backed SQLite engine + Phase 1 schema."""

    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'worker.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


@pytest.fixture
def storage_dir(tmp_path: Path) -> Path:
    return tmp_path / "images"


class _FakeFlux:
    """Records calls to ``generate`` / ``edit``; returns a deterministic
    PNG and seed. Tests substitute its method behaviour to exercise
    failure paths."""

    def __init__(self) -> None:
        self.generate = AsyncMock(return_value=(b"\x89PNG-fake-bytes", 42))
        self.edit = AsyncMock(return_value=(b"\x89PNG-edited", 7))


class _FakePubsub:
    """Captures everything published. Per-session lists so tests can
    distinguish broadcast targeting."""

    def __init__(self) -> None:
        self.published: list[tuple[str, ServerMessage]] = []

    async def publish(self, session_id: str, message: ServerMessage) -> int:
        self.published.append((session_id, message))
        return 1


async def _seed_campaign(
    factory: async_sessionmaker[AsyncSession],
    *,
    image_style: str | None = None,
    image_negative_prompt: str | None = None,
) -> tuple[str, str]:
    """Insert a user + campaign. Returns ``(user_id, campaign_id)``."""

    async with factory() as db, db.begin():
        user = models.User(username="alice", pwd_hash="x" * 60)
        db.add(user)
        await db.flush()
        campaign = models.Campaign(
            name="Borderlands",
            owner_id=user.id,
            image_style=image_style,
            image_negative_prompt=image_negative_prompt,
        )
        db.add(campaign)
        await db.flush()
        return user.id, campaign.id


async def _seed_session(factory: async_sessionmaker[AsyncSession], campaign_id: str) -> str:
    async with factory() as db, db.begin():
        session = models.Session(campaign_id=campaign_id)
        db.add(session)
        await db.flush()
        return session.id


async def _seed_character(
    factory: async_sessionmaker[AsyncSession], *, user_id: str, campaign_id: str
) -> str:
    async with factory() as db, db.begin():
        character = models.Character(
            user_id=user_id,
            campaign_id=campaign_id,
            name="Tav",
            race="Human",
            class_name="Fighter",
            alignment="neutral",
            level=1,
            hp_current=8,
            hp_max=8,
            ac=14,
            str_score=14,
            int_score=10,
            wis_score=10,
            dex_score=12,
            con_score=14,
            cha_score=10,
            gold=20,
        )
        db.add(character)
        await db.flush()
        return character.id


async def _seed_npc(factory: async_sessionmaker[AsyncSession], *, campaign_id: str) -> str:
    async with factory() as db, db.begin():
        npc = models.Npc(
            campaign_id=campaign_id,
            name="Castellan Thorvald",
            description="An elderly officer.",
        )
        db.add(npc)
        await db.flush()
        return npc.id


async def _seed_existing_image(
    factory: async_sessionmaker[AsyncSession],
    *,
    campaign_id: str,
    prompt: str,
    prompt_hash: str,
    file_path: str,
) -> str:
    async with factory() as db, db.begin():
        row = models.GeneratedImage(
            campaign_id=campaign_id,
            kind="scene",
            prompt=prompt,
            prompt_hash=prompt_hash,
            file_path=file_path,
        )
        db.add(row)
        await db.flush()
        return row.id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scene_job_writes_file_row_and_publishes_ready(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    """A non-deduped scene job hits FLUX once, writes the PNG to disk,
    inserts a generated_images row, and publishes image_ready on the
    session channel."""

    _user_id, campaign_id = await _seed_campaign(factory)
    session_id = await _seed_session(factory, campaign_id)
    flux = _FakeFlux()
    pubsub = _FakePubsub()

    job = ImageJob(
        id="img-1",
        campaign_id=campaign_id,
        session_id=session_id,
        kind="scene",
        prompt="a torchlit crypt",
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )

    # File on disk
    assert (storage_dir / "img-1.png").exists()
    assert (storage_dir / "img-1.png").read_bytes() == b"\x89PNG-fake-bytes"

    # Row in DB
    async with factory() as db:
        row = await db.get(models.GeneratedImage, "img-1")
    assert row is not None
    assert row.kind == "scene"
    assert row.width == _KIND_PARAMS["scene"]["width"]
    assert row.height == _KIND_PARAMS["scene"]["height"]

    # Published ImageReady on the originating session
    assert len(pubsub.published) == 1
    sid, msg = pubsub.published[0]
    assert sid == session_id
    assert isinstance(msg, ImageReady)
    assert msg.image_id == "img-1"
    assert msg.url == "/api/images/img-1.png"

    # FLUX called exactly once with scene parameters
    flux.generate.assert_awaited_once()
    _args, kwargs = flux.generate.call_args
    assert kwargs["width"] == _KIND_PARAMS["scene"]["width"]
    assert kwargs["height"] == _KIND_PARAMS["scene"]["height"]
    assert kwargs["steps"] == _KIND_PARAMS["scene"]["steps"]
    assert kwargs["guidance"] == _KIND_PARAMS["scene"]["guidance"]


@pytest.mark.asyncio
async def test_per_kind_parameters_passed_through(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    """An ``npc`` job uses 768x1024 + 32 steps, not the scene
    defaults. A regression that maps everything to scene parameters
    would silently squash NPC portraits to a wide aspect ratio."""

    _u, campaign_id = await _seed_campaign(factory)
    session_id = await _seed_session(factory, campaign_id)
    flux = _FakeFlux()
    pubsub = _FakePubsub()

    job = ImageJob(
        id="npc-1",
        campaign_id=campaign_id,
        session_id=session_id,
        kind="npc",
        prompt="grizzled half-orc cleric",
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )
    _args, kwargs = flux.generate.call_args
    assert kwargs["width"] == 768
    assert kwargs["height"] == 1024
    assert kwargs["steps"] == 32


@pytest.mark.asyncio
async def test_campaign_style_appended_to_prompt(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    """The campaign-level ``image_style`` suffix gets appended to
    every prompt; ``image_negative_prompt`` flows through to FLUX's
    ``negative_prompt`` slot."""

    _u, campaign_id = await _seed_campaign(
        factory,
        image_style="dark fantasy oil painting, candlelight",
        image_negative_prompt="modern objects, watermark",
    )
    session_id = await _seed_session(factory, campaign_id)
    flux = _FakeFlux()
    pubsub = _FakePubsub()

    job = ImageJob(
        id="img-2",
        campaign_id=campaign_id,
        session_id=session_id,
        kind="scene",
        prompt="a goblin warband on the road",
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )
    args, kwargs = flux.generate.call_args
    assert "a goblin warband on the road" in args[0]
    assert "dark fantasy oil painting, candlelight" in args[0]
    assert kwargs["negative_prompt"] == "modern objects, watermark"


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_hit_skips_flux_and_publishes_existing_id(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    """An identical second request reuses the existing row's id and
    publishes image_ready pointing at it. FLUX is never called.
    Spec §8: hash-based dedup is load-bearing for cost / latency."""

    _u, campaign_id = await _seed_campaign(factory)
    session_id = await _seed_session(factory, campaign_id)

    prompt = "a torchlit crypt"
    expected_hash = _hash_inputs(
        campaign_id=campaign_id,
        kind="scene",
        prompt_with_style=prompt,
        reference_image_id=None,
    )
    existing_id = await _seed_existing_image(
        factory,
        campaign_id=campaign_id,
        prompt=prompt,
        prompt_hash=expected_hash,
        file_path=str(storage_dir / "existing.png"),
    )

    flux = _FakeFlux()
    pubsub = _FakePubsub()
    job = ImageJob(
        id="new-id-but-same-hash",
        campaign_id=campaign_id,
        session_id=session_id,
        kind="scene",
        prompt=prompt,
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )

    flux.generate.assert_not_awaited()
    flux.edit.assert_not_awaited()
    assert len(pubsub.published) == 1
    _sid, msg = pubsub.published[0]
    assert isinstance(msg, ImageReady)
    assert msg.image_id == existing_id  # the cached row, not the job id


@pytest.mark.asyncio
async def test_hash_includes_reference_image_id(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    """An /edit job with a reference image must NOT collide with a
    /generate job that has the same prompt+style. The reference id is
    part of the hash key per spec §8."""

    h_no_ref = _hash_inputs(
        campaign_id="c",
        kind="scene",
        prompt_with_style="p",
        reference_image_id=None,
    )
    h_with_ref = _hash_inputs(
        campaign_id="c",
        kind="scene",
        prompt_with_style="p",
        reference_image_id="canon-1",
    )
    assert h_no_ref != h_with_ref


# ---------------------------------------------------------------------------
# Kontext /edit path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_job_loads_source_and_calls_flux_edit(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    """A job with reference_image_id+edit_instruction reads the source
    PNG off disk and dispatches via ``flux.edit`` rather than
    ``flux.generate``."""

    _u, campaign_id = await _seed_campaign(factory)
    session_id = await _seed_session(factory, campaign_id)

    storage_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = storage_dir / "canonical.png"
    canonical_bytes = b"\x89PNG-canonical-portrait"
    canonical_path.write_bytes(canonical_bytes)
    canon_id = await _seed_existing_image(
        factory,
        campaign_id=campaign_id,
        prompt="a heroic portrait",
        prompt_hash="canonhash",
        file_path=str(canonical_path),
    )

    flux = _FakeFlux()
    pubsub = _FakePubsub()
    job = ImageJob(
        id="edit-1",
        campaign_id=campaign_id,
        session_id=session_id,
        kind="npc",
        prompt="placeholder",
        reference_image_id=canon_id,
        edit_instruction="same character, torchlit crypt",
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )

    flux.generate.assert_not_awaited()
    flux.edit.assert_awaited_once()
    args, _kw = flux.edit.call_args
    assert args[0] == "same character, torchlit crypt"
    assert args[1] == canonical_bytes

    async with factory() as db:
        row = await db.get(models.GeneratedImage, "edit-1")
    assert row is not None
    assert row.source_image_id == canon_id
    assert row.edit_instruction == "same character, torchlit crypt"

    sid, msg = pubsub.published[0]
    assert sid == session_id
    assert isinstance(msg, ImageReady)


@pytest.mark.asyncio
async def test_missing_reference_emits_image_failed(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    _u, campaign_id = await _seed_campaign(factory)
    session_id = await _seed_session(factory, campaign_id)
    flux = _FakeFlux()
    pubsub = _FakePubsub()

    job = ImageJob(
        id="edit-broken",
        campaign_id=campaign_id,
        session_id=session_id,
        kind="npc",
        prompt="x",
        reference_image_id="does-not-exist",
        edit_instruction="anything",
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )
    flux.edit.assert_not_awaited()
    _sid, msg = pubsub.published[0]
    assert isinstance(msg, ImageFailed)
    assert msg.reason == "missing_reference"


@pytest.mark.asyncio
async def test_edit_job_without_instruction_emits_invalid_job(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    """An incomplete edit job (reference set, instruction missing)
    fails fast without touching FLUX or the DB."""

    _u, campaign_id = await _seed_campaign(factory)
    session_id = await _seed_session(factory, campaign_id)
    flux = _FakeFlux()
    pubsub = _FakePubsub()

    job = ImageJob(
        id="edit-bad",
        campaign_id=campaign_id,
        session_id=session_id,
        kind="npc",
        prompt="x",
        reference_image_id="anything",
        edit_instruction=None,
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )
    flux.edit.assert_not_awaited()
    flux.generate.assert_not_awaited()
    _sid, msg = pubsub.published[0]
    assert isinstance(msg, ImageFailed)
    assert msg.reason == "invalid_job"


# ---------------------------------------------------------------------------
# FLUX failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flux_error_emits_image_failed_and_skips_persist(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    """A FluxClientError (e.g. 503 retry exhaustion) becomes an
    image_failed event with reason=flux_unavailable. No row is
    written; the orchestrator's narration is unaffected."""

    _u, campaign_id = await _seed_campaign(factory)
    session_id = await _seed_session(factory, campaign_id)
    flux = _FakeFlux()
    flux.generate = AsyncMock(side_effect=FluxClientError("503 exhausted"))
    pubsub = _FakePubsub()

    job = ImageJob(
        id="img-fail",
        campaign_id=campaign_id,
        session_id=session_id,
        kind="scene",
        prompt="a battle",
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )

    async with factory() as db:
        rows = (await db.scalars(select(models.GeneratedImage))).all()
    assert rows == []  # no row inserted

    sid, msg = pubsub.published[0]
    assert sid == session_id
    assert isinstance(msg, ImageFailed)
    assert msg.image_id == "img-fail"
    assert msg.reason == "flux_unavailable"


# ---------------------------------------------------------------------------
# FK link to characters / npcs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subject_character_id_updates_canonical_image_id(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    """When a portrait job names a ``subject_character_id``, the
    worker links the new image to ``characters.canonical_image_id`` in
    the same transaction as the row insert."""

    user_id, campaign_id = await _seed_campaign(factory)
    session_id = await _seed_session(factory, campaign_id)
    char_id = await _seed_character(factory, user_id=user_id, campaign_id=campaign_id)

    flux = _FakeFlux()
    pubsub = _FakePubsub()
    job = ImageJob(
        id="portrait-1",
        campaign_id=campaign_id,
        session_id=session_id,
        kind="npc",
        prompt="portrait of Tav",
        subject_character_id=char_id,
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )

    async with factory() as db:
        character = await db.get(models.Character, char_id)
    assert character is not None
    assert character.canonical_image_id == "portrait-1"


@pytest.mark.asyncio
async def test_subject_npc_id_updates_canonical_image_id(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    _u, campaign_id = await _seed_campaign(factory)
    session_id = await _seed_session(factory, campaign_id)
    npc_id = await _seed_npc(factory, campaign_id=campaign_id)

    flux = _FakeFlux()
    pubsub = _FakePubsub()
    job = ImageJob(
        id="portrait-npc",
        campaign_id=campaign_id,
        session_id=session_id,
        kind="npc",
        prompt="portrait of Castellan",
        subject_npc_id=npc_id,
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )

    async with factory() as db:
        npc = await db.get(models.Npc, npc_id)
    assert npc is not None
    assert npc.canonical_image_id == "portrait-npc"


# ---------------------------------------------------------------------------
# Session-less broadcast
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_session_id_persists_but_skips_publish(
    factory: async_sessionmaker[AsyncSession], storage_dir: Path
) -> None:
    """Canonical portraits requested outside an active session
    (session_id None) still persist; just no broadcast."""

    user_id, campaign_id = await _seed_campaign(factory)
    char_id = await _seed_character(factory, user_id=user_id, campaign_id=campaign_id)
    flux = _FakeFlux()
    pubsub = _FakePubsub()
    job = ImageJob(
        id="orphan-portrait",
        campaign_id=campaign_id,
        session_id=None,
        kind="npc",
        prompt="portrait of Tav",
        subject_character_id=char_id,
    )
    await _process_job(
        job,
        flux=flux,  # type: ignore[arg-type]
        pubsub=pubsub,  # type: ignore[arg-type]
        factory=factory,
        storage_dir=storage_dir,
    )

    async with factory() as db:
        row = await db.get(models.GeneratedImage, "orphan-portrait")
    assert row is not None
    assert pubsub.published == []  # no broadcast


# ---------------------------------------------------------------------------
# _hash_inputs determinism
# ---------------------------------------------------------------------------


def test_hash_is_deterministic() -> None:
    """SHA-256 hex of identical inputs is stable across calls — the
    enqueuer can pre-compute and the worker will compute the same."""

    a = _hash_inputs(campaign_id="c", kind="scene", prompt_with_style="p", reference_image_id=None)
    b = _hash_inputs(campaign_id="c", kind="scene", prompt_with_style="p", reference_image_id=None)
    assert a == b
    assert a == hashlib.sha256(b"c\x00scene\x00p\x00").hexdigest()
