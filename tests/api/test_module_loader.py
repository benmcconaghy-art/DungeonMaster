"""Integration tests for POST /api/campaigns/from-module.

Tests the full module-load transaction: campaign + locations + NPCs +
world_facts + module_state. Image enqueue is tested with a fake queue
client.

Design note: these tests use a combined ``client_db`` fixture that yields
both an AsyncClient and an AsyncSession on the *same* in-memory engine, so
we can insert Module rows directly while the HTTP layer still uses the
same database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.base import Base
from app.db.models import Campaign, CampaignMember, Location, Module, Npc, WorldFact
from app.db.session import create_engine
from app.deps import get_db
from app.images.portrait import reset_for_tests as reset_queue_client
from app.images.portrait import set_queue_client_for_tests
from app.main import app as fastapi_app
from sqlalchemy import select

_VALID_PW = "correct horse battery staple"

_FIXTURE_MODULE_CONTENT = {
    "format_version": "1.0",
    "synopsis": "Goblins raid Morgansfort. The party is hired to stop them.",
    "tone": "gritty",
    "image_style": "dark fantasy ink illustration",
    "image_negative_prompt": "modern objects, photographic",
    "level_range": [1, 3],
    "estimated_sessions": 4,
    "starting_hook": "A rider arrives at dusk with a sealed letter.",
    "starting_location_symbol": "loc_keep_gate",
    "locations": [
        {
            "symbol": "loc_keep_gate",
            "name": "Morgansfort Gate",
            "description": "Heavy oak gates studded with iron bolts.",
        },
        {
            "symbol": "loc_common_hall",
            "name": "Common Hall",
            "description": "Smoky long-hall smelling of tallow and old straw.",
            "parent_symbol": "loc_keep_gate",
        },
    ],
    "npcs": [
        {
            "symbol": "npc_castellan",
            "name": "Castellan Thorvald",
            "description": "Greying veteran, missing two fingers.",
            "motivation": "Protect the keep at any cost.",
            "starting_location_symbol": "loc_keep_gate",
            "stats": {"hd": "3", "ac": 14, "hp": 18},
            "sample_dialogue": "The keep has stood two hundred years.",
        },
    ],
    "encounters": [],
    "plot_beats": [
        {
            "symbol": "beat_arrival",
            "title": "Arrival Briefing",
            "trigger_hint": "When the party first speaks with the Castellan.",
            "outcome": "Party understands the threat and is offered payment.",
        },
        {
            "symbol": "beat_caves_found",
            "title": "Goblin Caves Located",
            "trigger_hint": "When the party discovers the cave entrance.",
            "outcome": "The caves are confirmed as the goblin base.",
        },
    ],
    "secrets": [
        {
            "symbol": "sec_old_shame",
            "content": "Thorvald abandoned his post twenty years ago.",
            "reveal_when": "When the party finds the old orders.",
        }
    ],
    "endings": [
        {
            "symbol": "end_clean",
            "trigger": "Party defeats the goblin chief.",
            "outcome": "The keep is safe.",
        }
    ],
    "world_facts": [
        {
            "fact": "Morgansfort has stood for two centuries and never been breached.",
            "tags": ["morgansfort", "history"],
            "importance": 7,
        }
    ],
}


# ---------------------------------------------------------------------------
# Shared fixture: client + db on the same engine
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client_db(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[AsyncClient, AsyncSession]]:
    """Yield ``(client, db)`` sharing the same in-memory test database.

    This lets tests insert Module rows via ``db`` and then exercise
    the HTTP endpoint via ``client`` against the same schema.
    """
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    monkeypatch.setattr("app.orchestrator.dm.SessionLocal", factory)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as s:
            yield s

    fastapi_app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=fastapi_app)
    async with factory() as db:
        try:
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                yield ac, db
        finally:
            fastapi_app.dependency_overrides.pop(get_db, None)

    await engine.dispose()


async def _register_and_login(client: AsyncClient, username: str = "alice") -> str:
    resp = await client.post(
        "/api/auth/register", json={"username": username, "password": _VALID_PW}
    )
    assert resp.status_code == 201
    return resp.json()["id"]


class _FakeQueueClient:
    def __init__(self) -> None:
        self.pushed: list[bytes] = []

    async def rpush(self, key: str, value: bytes) -> int:
        self.pushed.append(value)
        return len(self.pushed)

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_module_creates_campaign_with_locations_and_npcs(
    client_db: tuple[AsyncClient, AsyncSession],
) -> None:
    """Full load: campaign, locations (parent + child), NPC, module_state."""

    client, db = client_db
    user_id = await _register_and_login(client)

    mod = Module(author_id=user_id, name="Morgansfort", content=_FIXTURE_MODULE_CONTENT)
    db.add(mod)
    await db.commit()
    await db.refresh(mod)

    fake_queue = _FakeQueueClient()
    set_queue_client_for_tests(fake_queue)
    try:
        resp = await client.post(
            "/api/campaigns/from-module",
            json={"module_id": mod.id, "name": "Morgansfort Campaign"},
        )
    finally:
        await reset_queue_client()

    assert resp.status_code == 201, resp.text
    body = resp.json()
    campaign_id = body["campaign_id"]
    assert body["locations_created"] == 2
    assert body["npcs_created"] == 1

    # Verify campaign row.
    campaign = await db.get(Campaign, campaign_id)
    assert campaign is not None
    assert campaign.module_id == mod.id
    assert campaign.image_style == "dark fantasy ink illustration"
    assert campaign.image_negative_prompt == "modern objects, photographic"

    # Verify module_state.
    ms = campaign.module_state
    assert ms["module_id"] == mod.id
    assert "beat_arrival" in ms["beats_pending"]
    assert "beat_caves_found" in ms["beats_pending"]
    assert ms["beats_hit"] == []
    assert "sec_old_shame" in ms["symbolic_id_map"]
    assert "loc_keep_gate" in ms["symbolic_id_map"]
    assert "npc_castellan" in ms["symbolic_id_map"]

    # Verify locations.
    locs = list(
        (await db.scalars(select(Location).where(Location.campaign_id == campaign_id))).all()
    )
    assert len(locs) == 2
    loc_names = {loc.name for loc in locs}
    assert "Morgansfort Gate" in loc_names
    assert "Common Hall" in loc_names

    # Verify parent-child relationship.
    child = next(loc for loc in locs if loc.name == "Common Hall")
    parent = next(loc for loc in locs if loc.name == "Morgansfort Gate")
    assert child.parent_id == parent.id

    # Verify NPC.
    npcs = list(
        (await db.scalars(select(Npc).where(Npc.campaign_id == campaign_id))).all()
    )
    assert len(npcs) == 1
    assert npcs[0].name == "Castellan Thorvald"
    assert npcs[0].location_id == parent.id  # starting_location_symbol = loc_keep_gate

    # Verify CampaignMember (owner).
    member = await db.get(CampaignMember, (campaign_id, user_id))
    assert member is not None
    assert member.role == "owner"


@pytest.mark.asyncio
async def test_load_module_module_state_beats_all_pending(
    client_db: tuple[AsyncClient, AsyncSession],
) -> None:
    """All plot_beat symbols land in beats_pending; beats_hit starts empty."""

    client, db = client_db
    user_id = await _register_and_login(client)

    mod = Module(author_id=user_id, name="Test", content=_FIXTURE_MODULE_CONTENT)
    db.add(mod)
    await db.commit()
    await db.refresh(mod)

    fake_queue = _FakeQueueClient()
    set_queue_client_for_tests(fake_queue)
    try:
        resp = await client.post(
            "/api/campaigns/from-module",
            json={"module_id": mod.id, "name": "Beat Test"},
        )
    finally:
        await reset_queue_client()

    assert resp.status_code == 201
    campaign = await db.get(Campaign, resp.json()["campaign_id"])
    assert campaign is not None
    ms = campaign.module_state
    assert set(ms["beats_pending"]) == {"beat_arrival", "beat_caves_found"}
    assert ms["beats_hit"] == []
    assert ms["secrets_revealed"] == []


@pytest.mark.asyncio
async def test_load_module_image_style_override(
    client_db: tuple[AsyncClient, AsyncSession],
) -> None:
    """image_style_override replaces the module's default image_style."""

    client, db = client_db
    user_id = await _register_and_login(client)

    mod = Module(author_id=user_id, name="Test", content=_FIXTURE_MODULE_CONTENT)
    db.add(mod)
    await db.commit()
    await db.refresh(mod)

    fake_queue = _FakeQueueClient()
    set_queue_client_for_tests(fake_queue)
    try:
        resp = await client.post(
            "/api/campaigns/from-module",
            json={
                "module_id": mod.id,
                "name": "Override Test",
                "image_style_override": "watercolour pastel",
            },
        )
    finally:
        await reset_queue_client()

    assert resp.status_code == 201
    campaign = await db.get(Campaign, resp.json()["campaign_id"])
    assert campaign is not None
    assert campaign.image_style == "watercolour pastel"


@pytest.mark.asyncio
async def test_load_module_unknown_module_returns_404(client: AsyncClient) -> None:
    await _register_and_login(client)
    resp = await client.post(
        "/api/campaigns/from-module",
        json={"module_id": "00000000-0000-0000-0000-000000000000", "name": "x"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_load_module_requires_auth(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/campaigns/from-module",
        json={"module_id": "anything", "name": "x"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_load_module_world_facts_embedded(
    client_db: tuple[AsyncClient, AsyncSession],
) -> None:
    """world_facts from the module are inserted with embeddings."""

    client, db = client_db
    user_id = await _register_and_login(client)

    mod = Module(author_id=user_id, name="Test", content=_FIXTURE_MODULE_CONTENT)
    db.add(mod)
    await db.commit()
    await db.refresh(mod)

    fake_queue = _FakeQueueClient()
    set_queue_client_for_tests(fake_queue)
    try:
        resp = await client.post(
            "/api/campaigns/from-module",
            json={"module_id": mod.id, "name": "Facts Test"},
        )
    finally:
        await reset_queue_client()

    assert resp.status_code == 201
    campaign_id = resp.json()["campaign_id"]

    facts = list(
        (
            await db.scalars(
                select(WorldFact).where(WorldFact.campaign_id == campaign_id)
            )
        ).all()
    )
    assert len(facts) == 1
    assert "Morgansfort" in facts[0].fact
    assert facts[0].embedding is not None
    assert len(facts[0].embedding) > 0
    assert facts[0].importance == 7
    assert "history" in facts[0].tags
