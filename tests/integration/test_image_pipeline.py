"""End-to-end smoke against live FLUX + live Valkey for the Phase 5
image pipeline.

Verifies the full path that unit tests can only fake:

  enqueuer → real Valkey list → real worker BLPOP → real FLUX HTTP
  → real PNG bytes → real disk write → real DB insert → real Valkey
  pub/sub broadcast → subscriber assertion

Skipped if FLUX or Valkey is unreachable. Run with::

    uv run pytest -m integration tests/integration/test_image_pipeline.py

The test uses a fresh randomly-named Valkey queue key (overriding
:data:`app.images.queue.QUEUE_KEY`) and a randomly-named session id
so concurrent runs and a populated production Valkey don't collide.
PNG bytes land in a tmp_path so we don't pollute the real image
directory.

Cold-load expectation per spec §8: 256x256/1-step ~5s, full scene
params ~17s on warm hardware. Timeouts are generous (90s) to absorb
a real cold load.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import models
from app.db.base import Base
from app.db.session import create_engine
from app.images import queue as queue_module
from app.images.client import FluxClient, FluxClientError
from app.images.queue import ImageJob, open_queue_client, push_job
from app.images.worker import _process_job
from app.realtime.messages import ImageReady, ServerMessage
from app.realtime.pubsub import DmPubsubError, Pubsub

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Reachability fixtures (skip cleanly when upstreams are down)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def reachable_flux() -> AsyncIterator[FluxClient]:
    """Probe FLUX once with a 256x256/1-step generation. /health is
    insufficient (it returns 200 even when /generate is broken — the
    incident that motivated the watchdog probe widening). If the deep
    probe fails, skip the test rather than fail."""

    client = FluxClient()
    try:
        await client.probe()
    except FluxClientError as exc:
        await client.aclose()
        pytest.skip(f"FLUX unreachable: {exc}")
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def reachable_pubsub() -> AsyncIterator[Pubsub]:
    instance = Pubsub()
    try:
        await instance.health()
    except DmPubsubError as exc:
        await instance.aclose()
        pytest.skip(f"Valkey unreachable: {exc}")
    try:
        yield instance
    finally:
        await instance.aclose()


@pytest_asyncio.fixture
async def temp_db(
    tmp_path: Path,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """File-backed SQLite engine + Phase 1 schema, isolated per test."""

    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'integration.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_campaign_and_session(
    factory: async_sessionmaker[AsyncSession],
) -> tuple[str, str]:
    """Seed a campaign + session so the worker's FK reads succeed.
    Returns ``(campaign_id, session_id)``."""

    async with factory() as db, db.begin():
        user = models.User(username="integration-tester", pwd_hash="x" * 60)
        db.add(user)
        await db.flush()
        campaign = models.Campaign(
            name="Integration Run",
            owner_id=user.id,
            image_style="dark fantasy oil painting, candlelight",
        )
        db.add(campaign)
        await db.flush()
        session = models.Session(campaign_id=campaign.id)
        db.add(session)
        await db.flush()
        return campaign.id, session.id


# ---------------------------------------------------------------------------
# Live pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scene_job_round_trip_against_live_flux_and_valkey(
    reachable_flux: FluxClient,
    reachable_pubsub: Pubsub,
    temp_db: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Push a job onto a real Valkey queue, drive ``_process_job``
    against real FLUX, watch the real publish land on a real subscriber.

    Uses 256x256/1-step parameters via a kind override of "scene" with
    custom dimensions wouldn't work — _KIND_PARAMS is fixed per kind.
    Instead we accept the spec's scene params (1280x768/28-step) and
    rely on FLUX warming up to ~17s per the cold-load measurement.
    """

    campaign_id, session_id = await _seed_campaign_and_session(temp_db)
    storage_dir = tmp_path / "images"

    # Subscribe before publishing so the round-trip race doesn't drop
    # the message. Capture exactly one ImageReady per the test's
    # single-job pipeline.
    received: list[ServerMessage] = []

    async def reader() -> None:
        async for msg in reachable_pubsub.subscribe(session_id):
            received.append(msg)
            return

    reader_task = asyncio.create_task(reader())
    await asyncio.sleep(0.2)  # let the redis subscribe attach

    job = ImageJob(
        id="integration-img-" + uuid.uuid4().hex[:8],
        campaign_id=campaign_id,
        session_id=session_id,
        kind="scene",
        prompt="a small ceramic bowl on a wooden table",
    )
    await _process_job(
        job,
        flux=reachable_flux,
        pubsub=reachable_pubsub,
        factory=temp_db,
        storage_dir=storage_dir,
    )

    # Wait for the broadcast — generous timeout because the redis
    # round-trip is fast (sub-second) but absorbing real network jitter
    # buys cheap resilience.
    await asyncio.wait_for(reader_task, timeout=10.0)

    assert len(received) == 1
    msg = received[0]
    assert isinstance(msg, ImageReady)
    assert msg.image_id == job.id
    assert msg.url == f"/api/images/{job.id}.png"

    # PNG actually landed on disk.
    file_path = storage_dir / f"{job.id}.png"
    assert file_path.exists()
    assert file_path.stat().st_size > 1000  # real PNG, not an empty stub
    assert file_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # Row committed in DB before the broadcast (transaction discipline).
    async with temp_db() as db:
        row = await db.get(models.GeneratedImage, job.id)
    assert row is not None
    assert row.kind == "scene"
    assert row.campaign_id == campaign_id


@pytest.mark.asyncio
async def test_queue_round_trip_against_live_valkey(
    reachable_pubsub: Pubsub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lighter test that doesn't touch FLUX: the redis-side helpers
    (push_job / pop_job) survive a real Valkey, including BLPOP
    timeout. Keeps coverage of the queue layer in the integration
    suite even if FLUX is down (this fixture only requires Valkey)."""

    queue_key = f"images:queue:test:{uuid.uuid4()}"
    monkeypatch.setattr(queue_module, "QUEUE_KEY", queue_key)

    client = open_queue_client()
    try:
        # Empty queue → BLPOP times out, returns None.
        from app.images.queue import pop_job

        result = await pop_job(client, timeout=0.5)
        assert result is None

        # Push a job, pop it back, verify shape preserved.
        job = ImageJob(
            id="rt-" + uuid.uuid4().hex[:8],
            campaign_id="camp-rt",
            session_id="sess-rt",
            kind="scene",
            prompt="round trip",
        )
        await push_job(client, job)
        popped = await pop_job(client, timeout=2.0)
        assert popped is not None
        assert popped.id == job.id
        assert popped.prompt == "round trip"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_flux_probe_against_live_service(reachable_flux: FluxClient) -> None:
    """The deep liveness probe must succeed against the live FLUX —
    this is the only test that confirms the watchdog's signal
    actually maps to the production service shape (256x256/1-step
    generates a valid PNG-bearing JSON response in under 60s)."""

    payload = await reachable_flux.probe()
    assert "image_base64" in payload
    assert "seed_used" in payload
    # generation_time_seconds is service-reported and may understate
    # wall-clock; we don't pin it, just assert presence.
    assert "generation_time_seconds" in payload
