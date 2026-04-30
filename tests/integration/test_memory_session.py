"""End-to-end memory exercise across a 25-turn session.

Validates the four memory tiers from spec §7 work in concert against
the real vLLM endpoint and the real (local) embedder:

  - **Verbatim** (N=40 turns) — prompt builder includes recent messages.
  - **Session summary** — populated automatically when the player turn
    count crosses a multiple of 20.
  - **World facts** — extracted post-turn via :func:`extract_and_persist_facts`,
    embedded, and stored in ``world_facts``.
  - **Vector retrieval** — top-K cosine retrieval surfaces earlier facts
    when later turns reference them.

The test runs 25 player turns chosen to be mostly non-combat so wall
clock stays under ~15 minutes — combat turns chain 6-10 tool calls each
which would push the test past 30 min. A named NPC ("Castellan
Thorvald") is introduced at turn 5 so retrieval at turn 25 has a
concrete earlier referent to find.

Run with::

    uv run pytest -m integration tests/integration/test_memory_session.py

Skipped by default. Skips the module if vLLM is unreachable; the
embedder runs locally so that's always available (loads ~1.5GB on first
call — happens before the loop).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.db import models
from app.db.base import Base
from app.db.session import create_engine
from app.game.chargen import AbilityScores, generate_character
from app.llm.client import DmClientError, get_dm_client
from app.llm.client import reset_for_tests as reset_dm_client
from app.llm.embeddings import get_embedder
from app.llm.embeddings import reset_for_tests as reset_embedder
from app.llm.memory import get_world_fact_retriever
from app.orchestrator import dm as orchestrator_dm
from app.orchestrator.dm import take_turn

pytestmark = pytest.mark.integration


# Wall-clock budget per turn including its post-turn background drain.
# Empirical: 5-30s for non-combat dispatch + 5-15s for the fact-extractor
# LLM call. 90s is generous.
_PER_TURN_TIMEOUT_S = 90.0

# Phase 3 raised this from Phase 2's 500ms. The build_dm_prompt path now
# includes a query-embedding call for world-fact retrieval inside the
# autobegun read transaction; even with a warm local embedder that adds
# 100-500ms to legitimate prompt builds. The threshold's real job is
# catching stream-held transactions (5-30 seconds); 1500ms still does
# that with several seconds of margin.
_MAX_TX_OPEN_MS = 1500.0


# Player actions chosen to be mostly non-combat. Castellan Thorvald is
# named at turn 5 (zero-indexed: index 4) so retrieval at turn 25 has a
# specific earlier referent. The final action explicitly references him
# so we can assert on retrieval.
_ACTIONS: list[str] = [
    "I take in the keep's main courtyard. What stands out?",
    "I check my pack — what equipment do I have?",
    "I look for someone in charge.",
    "I head toward the keep's tower.",
    "I find an elderly officer named Castellan Thorvald and introduce myself.",
    "I ask Thorvald about the goblin raids he's been dealing with.",
    "I ask Thorvald what he can pay for help with the goblins.",
    "I ask if there are other adventurers in town.",
    "I take a small purse of silver Thorvald offers as advance.",
    "I leave Thorvald's office and find a tavern.",
    "I order ale and listen to local rumours.",
    "I ask the tavernkeeper about the goblin caves.",
    "I rent a room for the night.",
    "I rest until morning.",
    "I have breakfast in the common room.",
    "I check my equipment one more time before heading out.",
    "I set out for the goblin caves on foot, watching for tracks.",
    "I follow the road north for an hour, eyes on the treeline.",
    "I take a short rest by the side of the road.",
    "I press on; the road bends west into thicker forest.",
    "I notice an old stone marker by the road and stop to inspect it.",
    "I continue toward the caves, slower now in the underbrush.",
    "I scout ahead carefully, watching for goblin sentries.",
    "I find a defensible spot to wait until dusk.",
    (
        "Looking back on this whole expedition, what would Castellan"
        " Thorvald think of my progress so far?"
    ),
]


@pytest.fixture(scope="module")
def vllm_reachable() -> bool:
    """Skip the whole module if vLLM is unreachable."""

    settings = get_settings()
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as c:
            response = c.get(f"{settings.vllm_base_url}/v1/models")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"vLLM unreachable at {settings.vllm_base_url}: {exc}")
    return True


@pytest_asyncio.fixture
async def integration_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[async_sessionmaker[AsyncSession], list[float]]]:
    """File-backed engine with the transaction-duration tripwire and a
    monkey-patch routing the orchestrator's post-turn ``SessionLocal``
    calls at this same engine — so the fact extractor lands in the
    same DB the assertions read.
    """

    db_path = tmp_path / "memory-integration.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    durations_ms: list[float] = []
    open_at: dict[int, float] = {}

    @event.listens_for(engine.sync_engine, "begin")
    def _on_begin(conn: Any) -> None:
        open_at[id(conn)] = time.monotonic()

    def _close(conn: Any) -> None:
        start = open_at.pop(id(conn), None)
        if start is not None:
            durations_ms.append((time.monotonic() - start) * 1000.0)

    event.listens_for(engine.sync_engine, "commit")(_close)
    event.listens_for(engine.sync_engine, "rollback")(_close)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    monkeypatch.setattr("app.orchestrator.dm.SessionLocal", factory)
    yield factory, durations_ms

    await engine.dispose()


@pytest_asyncio.fixture
async def fresh_singletons() -> AsyncIterator[None]:
    """Reset DmClient and embedder singletons around the test."""

    await reset_dm_client()
    await reset_embedder()
    # Pre-warm the embedder so loading its 1.5GB model isn't accounted
    # against the first turn's wall-clock budget.
    await get_embedder().health()
    # Also pre-warm the world-fact retriever's cache namespace.
    get_world_fact_retriever()
    yield
    await reset_dm_client()
    await reset_embedder()


async def _seed_world(
    factory: async_sessionmaker[AsyncSession],
) -> tuple[str, str, str]:
    """Insert a user + campaign + Fighter character + session.

    Returns ``(session_id, character_id, user_id)``.
    """

    import random

    async with factory() as db:
        user = models.User(username="alice", pwd_hash="x" * 60)
        db.add(user)
        await db.flush()

        campaign = models.Campaign(name="The Borderlands", owner_id=user.id)
        db.add(campaign)
        await db.flush()

        db.add(models.CampaignMember(campaign_id=campaign.id, user_id=user.id, role="owner"))

        rolled = generate_character(
            name="Borin Stoneforge",
            race_name="Dwarf",
            class_name="Fighter",
            alignment="lawful",
            rng=random.Random(42),
            method="classic",
            abilities=AbilityScores(
                str_score=16,
                int_score=10,
                wis_score=12,
                dex_score=12,
                con_score=14,
                cha_score=9,
            ),
        )
        character = models.Character(
            user_id=user.id,
            campaign_id=campaign.id,
            name=rolled.name,
            race=rolled.race,
            class_name=rolled.class_name,
            level=rolled.level,
            hp_current=rolled.hp_max,
            hp_max=rolled.hp_max,
            ac=rolled.ac,
            str_score=rolled.abilities.str_score,
            int_score=rolled.abilities.int_score,
            wis_score=rolled.abilities.wis_score,
            dex_score=rolled.abilities.dex_score,
            con_score=rolled.abilities.con_score,
            cha_score=rolled.abilities.cha_score,
            gold=rolled.starting_gold,
            alignment=rolled.alignment,
        )
        db.add(character)
        await db.flush()

        session = models.Session(campaign_id=campaign.id)
        db.add(session)
        await db.commit()

        return session.id, character.id, user.id


async def _drain_background_tasks() -> None:
    """Wait for the orchestrator's fire-and-forget post-turn tasks to
    complete. The test needs them done before asserting on world_facts
    or sessions.summary."""

    pending = list(orchestrator_dm._BACKGROUND_TASKS)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_25_turn_session_grows_memory(
    vllm_reachable: bool,
    integration_db: tuple[async_sessionmaker[AsyncSession], list[float]],
    fresh_singletons: None,
) -> None:
    factory, durations_ms = integration_db
    session_id, character_id, user_id = await _seed_world(factory)

    # Sanity-probe vLLM one more time so a mid-test outage gives a clean
    # skip rather than a confusing assertion failure.
    client = get_dm_client()
    try:
        await client.health()
    except DmClientError as exc:
        pytest.skip(f"vLLM became unreachable mid-test: {exc}")

    # ------- 25-turn loop -----------------------------------------------------

    turn_durations: list[float] = []
    for i, action in enumerate(_ACTIONS, start=1):
        turn_start = time.monotonic()

        async def collect(action: str = action) -> None:
            async with factory() as db:
                async for _event in take_turn(
                    db,
                    session_id=session_id,
                    sender_user_id=user_id,
                    sender_character_id=character_id,
                    content=action,
                ):
                    pass

        try:
            await asyncio.wait_for(collect(), timeout=_PER_TURN_TIMEOUT_S)
        except TimeoutError:
            pytest.fail(f"turn {i} exceeded {_PER_TURN_TIMEOUT_S}s budget")

        # Wait for the post-turn fact-extractor + maybe-summary tasks so
        # subsequent turns see the facts in retrieval.
        await asyncio.wait_for(_drain_background_tasks(), timeout=120.0)

        turn_durations.append(time.monotonic() - turn_start)

    # ------- (a) Session summary populated by turn 20-21 ----------------------

    async with factory() as db:
        session = await db.get(models.Session, session_id)
        assert session is not None
    assert session.summary, (
        "sessions.summary should have been regenerated when the player message"
        " count hit a multiple of 20 — got empty/None after 25 turns."
        " Background tasks may have failed silently; check logs."
    )
    # The summary should mention something concrete from the run, not
    # just be a templated stub.
    assert len(session.summary) > 200, f"summary suspiciously short: {len(session.summary)}"

    # ------- (b) world_facts has entries ------------------------------------

    async with factory() as db:
        facts = list(
            (await db.scalars(select(models.WorldFact).order_by(models.WorldFact.created_at))).all()
        )

    # Lower bound is permissive: even one extracted fact across 25 turns
    # demonstrates the pipeline. Real-world play tends to surface 5-15.
    assert len(facts) >= 1, (
        "fact extractor produced zero rows across 25 turns. Either Nemotron's"
        " JSON parsing is failing every turn (check logs for 'JSON parse failed'"
        " warnings) or the prompt isn't surfacing memorable content."
    )

    # Each fact's embedding must be 1024-dim and L2-normed (invariant #5).
    for fact in facts:
        import numpy as np

        vector = np.frombuffer(fact.embedding, dtype=np.float32)
        assert vector.shape == (1024,), f"unexpected dim: {vector.shape}"
        norm = float(np.linalg.norm(vector))
        assert abs(norm - 1.0) < 1e-3, f"non-unit norm: {norm}"
        assert fact.embedding_dim == 1024
        assert fact.source_session_id == session_id

    # ------- (c) Retrieval surfaces an earlier fact when referenced ---------
    #
    # The final action explicitly mentions Castellan Thorvald. The
    # retriever's top-K=5 should include at least one fact that mentions
    # him by name (introduced at turn 5).

    retriever = get_world_fact_retriever()
    async with factory() as db:
        hits = await retriever.topk(
            db,
            campaign_id=session.campaign_id,
            query="Castellan Thorvald and the goblin raids",
            k=5,
        )

    assert hits, "retriever returned zero hits for a query referencing a turn-5 NPC"
    thorvald_hit = any("thorvald" in h.fact.lower() for h in hits)
    assert thorvald_hit, "no Thorvald-related fact in top-5 retrieval. Hits were:\n" + "\n".join(
        f"  - {h.score:.3f} {h.fact}" for h in hits
    )

    # ------- (d) Prompt token count remains bounded ------------------------
    #
    # Build the prompt for what would be turn 26 and assert its
    # character size stays well within Nemotron's 256k context. This is
    # a coarse proxy for token count (we don't have a tokeniser
    # available); the relationship for English text is ~4 chars/token,
    # so 64k chars ≈ 16k tokens — comfortably below the 256k window even
    # under spec §7's 8.5k token budget plus retrieval expansion.

    from app.llm.prompts import build_dm_prompt

    async with factory() as db:
        prompt_messages = await build_dm_prompt(db, session_id=session_id)
    prompt_chars = sum(len(m.get("content") or "") for m in prompt_messages)
    assert prompt_chars < 64_000, (
        f"prompt grew unexpectedly large: {prompt_chars} chars" f" (expected <64k for ~16k tokens)"
    )
    # Lower bound — the prompt should be substantial after 25 turns.
    assert prompt_chars > 5_000, f"prompt suspiciously small: {prompt_chars} chars"

    # ------- Transaction-discipline tripwire ------------------------------

    over_budget = [d for d in durations_ms if d > _MAX_TX_OPEN_MS]
    assert not over_budget, (
        f"transaction held longer than {_MAX_TX_OPEN_MS}ms during the 25-turn run:"
        f" {over_budget}"
    )

    # Helpful diagnostic when this test passes — prints the wall-clock
    # spread so future runs can spot regressions.
    print(
        f"\n25-turn run: facts={len(facts)} retrieved_thorvald={thorvald_hit}"
        f" summary_len={len(session.summary)}"
        f" prompt_chars={prompt_chars} mean_turn_s={sum(turn_durations) / 25:.1f}"
        f" max_turn_s={max(turn_durations):.1f}"
    )
