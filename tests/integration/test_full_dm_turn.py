"""End-to-end DM turn against the real vLLM endpoint.

Exercises the full Phase 2 path: schema migration → chargen → session
start → ``take_turn`` → real Nemotron stream → tool dispatch → DB
persistence. Specifically validates spec §2's data flow and AGENTS.md
invariants #1, #2, and #4.

Run with::

    uv run pytest -m integration

Skipped by default (the default ``addopts`` in ``pyproject.toml`` is
``-m "not integration"``). The test will skip itself with a clear
message if the vLLM endpoint is unreachable, so a misconfigured dev
machine doesn't fail loudly.

The test asserts:

  - The DM stream produces at least one tool dispatch — Nemotron with
    "I attack" reliably calls ``request_dice_roll`` (and friends).
  - If a ``dice_roll`` event fired, the audit row exists in
    ``dice_rolls`` (engine, not LLM, is the dice source of truth).
  - The player's input always lands in ``session_messages``; the DM
    response lands too iff the turn completed cleanly (combat-heavy
    turns sometimes hit ``iteration_cap`` before a final narration —
    that's a documented Phase 2 limitation, not a wiring failure).
  - The "no write transaction during stream" invariant: the test
    instruments the test engine with ``begin`` / ``commit`` listeners
    and asserts every span is under 500ms (LLM streams take seconds,
    so anything above ~half a second is the canary we care about; 500
    leaves headroom over first-connection PRAGMA setup).
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
from app.llm.client import DmClientError, get_dm_client, reset_for_tests
from app.orchestrator.dm import (
    DiceRollEvent,
    DmError,
    NarrationChunk,
    NarrationComplete,
    ToolDispatched,
    take_turn,
)

# How long any single transaction is allowed to be open. Spec / AGENTS.md
# require zero open transactions during streaming; an LLM stream takes
# seconds, so anything multi-hundred-ms long is the canary we care about.
# 500ms is well above first-connection PRAGMA setup (~250ms cold) and
# well below any stream-held transaction (5-30 seconds).
_MAX_TX_OPEN_MS = 500.0


pytestmark = pytest.mark.integration


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
) -> AsyncIterator[tuple[async_sessionmaker[AsyncSession], list[float]]]:
    """File-backed engine (so WAL mode actually applies) with a
    transaction-duration tripwire attached.

    Yields the session factory and a list of recorded transaction
    durations (ms); the test asserts every entry is under
    ``_MAX_TX_OPEN_MS``.
    """

    db_path = tmp_path / "integration.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    durations_ms: list[float] = []

    # SQLAlchemy raises ``begin`` and ``commit`` events on the sync
    # engine that backs an AsyncEngine. We measure between them per
    # connection.
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
    yield factory, durations_ms

    await engine.dispose()


@pytest_asyncio.fixture
async def fresh_dm_client() -> AsyncIterator[None]:
    """Reset the DmClient singleton around the test so the resolved
    model id is fetched fresh against the real endpoint."""

    await reset_for_tests()
    yield
    await reset_for_tests()


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


@pytest.mark.asyncio
async def test_full_dm_turn_against_real_vllm(
    vllm_reachable: bool,
    integration_db: tuple[async_sessionmaker[AsyncSession], list[float]],
    fresh_dm_client: None,
) -> None:
    factory, durations_ms = integration_db
    session_id, character_id, user_id = await _seed_world(factory)

    # Sanity-probe the DmClient one more time so the test failure message
    # is clear if the endpoint went down between the module-level skip
    # and the actual run.
    client = get_dm_client()
    try:
        await client.health()
    except DmClientError as exc:
        pytest.skip(f"vLLM became unreachable mid-test: {exc}")

    events: list[Any] = []
    async with factory() as db:
        # Generous overall budget — Nemotron's reasoning + tool-call loop
        # can easily run multi-second; 90s leaves room for slow GPU days.
        async def collect() -> None:
            async for event in take_turn(
                db,
                session_id=session_id,
                sender_user_id=user_id,
                sender_character_id=character_id,
                content=(
                    "A goblin is sneering at me from the gateway. I draw my axe and" " attack it."
                ),
            ):
                events.append(event)

        await asyncio.wait_for(collect(), timeout=90.0)

    # ---- Stream-shape assertions -----------------------------------------

    # Two Nemotron-specific known outcomes are acceptable on a
    # combat-heavy Phase 2 turn:
    #
    #   - ``iteration_cap`` — the model chained 10+ tool calls without
    #     pausing for the player (combat round driven end-to-end).
    #   - ``empty_completion`` — the model emitted no content after a
    #     long tool-call loop, having said its piece via the tools.
    #
    # Both leave partial state persisted (player input, dice rolls, HP
    # changes) but no final assistant narration. Either is a Phase 2
    # finding, not a wiring failure. Any other dm_error reason — and
    # any number above one — is a bug we want to know about.
    _ALLOWED_ERROR_REASONS = {"iteration_cap", "empty_completion"}

    errors = [e for e in events if isinstance(e, DmError)]
    completions = [e for e in events if isinstance(e, NarrationComplete)]
    chunks = [e for e in events if isinstance(e, NarrationChunk)]
    dice_events = [e for e in events if isinstance(e, DiceRollEvent)]
    tool_dispatches = [e for e in events if isinstance(e, ToolDispatched)]

    if errors:
        assert len(errors) == 1, f"unexpected number of dm_error events: {errors}"
        assert errors[0].reason in _ALLOWED_ERROR_REASONS, (
            f"unexpected dm_error reason: {errors[0].reason!r}"
            f" (allowed: {_ALLOWED_ERROR_REASONS})"
        )
        assert not completions, "errored turn shouldn't also yield narration_complete"
    else:
        assert len(completions) == 1, "happy-path turn must yield exactly one narration_complete"

    # Streaming actually happened — either content tokens or tool-call
    # tokens (which are silent in narration but consume stream time).
    assert (
        chunks or tool_dispatches
    ), "no narration_chunk and no tool_dispatched — the stream produced nothing"

    # The Phase 2 ask: "at least one tool call (likely request_dice_roll)
    # was dispatched server-side". With "I attack" as the prompt this
    # consistently triggers; if it didn't, the prompt or rules text has
    # drifted and we want to know.
    assert tool_dispatches, "expected at least one tool dispatch on an attack action"

    # ---- DB persistence assertions ----------------------------------------

    async with factory() as db:
        messages = list(
            (
                await db.scalars(
                    select(models.SessionMessage)
                    .where(models.SessionMessage.session_id == session_id)
                    .order_by(models.SessionMessage.created_at)
                )
            ).all()
        )
        rolls = list(
            (
                await db.scalars(
                    select(models.DiceRoll).where(models.DiceRoll.session_id == session_id)
                )
            ).all()
        )

    # Player input must always be persisted (it lands in step 1 of
    # take_turn, before any LLM call).
    player_msgs = [m for m in messages if m.sender_kind == "player"]
    assert len(player_msgs) == 1, f"expected 1 player message, got {len(player_msgs)}"
    assert "goblin" in player_msgs[0].content.lower()

    # If the LLM yielded a dice_roll event, the audit row must exist —
    # AGENTS.md invariant: dice_rolls is the authoritative event log.
    if dice_events:
        assert rolls, "dice event yielded but dice_rolls table is empty"
        assert any(r.expression for r in rolls)

    # DM response: present iff the turn completed cleanly.
    dm_msgs = [m for m in messages if m.sender_kind == "dm"]
    if not errors:
        assert dm_msgs, "happy-path turn must persist a DM response"

    # ---- Transaction-discipline tripwire ---------------------------------

    over_budget = [d for d in durations_ms if d > _MAX_TX_OPEN_MS]
    assert not over_budget, (
        f"transaction held longer than {_MAX_TX_OPEN_MS}ms during DM turn:"
        f" {over_budget} (full set: {durations_ms})"
    )

    assert durations_ms, "tripwire recorded zero transactions; instrumentation broken"

    # Sanity: at least one transaction did occur (player write,
    # tool-dispatch writes, final narration). Empty list would mean the
    # tripwire wasn't actually wired.
    assert durations_ms, "tripwire recorded zero transactions; instrumentation broken"


@pytest.mark.asyncio
async def test_endpoint_health_resolves_model_id(
    vllm_reachable: bool,
    fresh_dm_client: None,
) -> None:
    """Cheap canary: the DmClient successfully reads ``/v1/models`` from
    the real endpoint and stores the resolved model id."""

    client = get_dm_client()
    await client.health()
    assert client.model.startswith("nvidia/")
    assert "Nemotron" in client.model
