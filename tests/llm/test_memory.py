"""Tests for ``app.llm.memory``.

Covers:

  - :func:`recent_turns` — the verbatim retrieval tier (default N=40 in
    Phase 3).
  - :class:`WorldFactRetriever` — cosine math, per-campaign cache,
    invalidation, dimension-mismatch detection.
  - :func:`maybe_regenerate_session_summary` — trigger condition,
    per-session lock collapses concurrent calls into one LLM round-trip.
  - :func:`regenerate_campaign_summary` — explicit-trigger persistence.
  - :func:`extract_and_persist_facts` — JSON parsing (fenced / trailing
    prose / malformed), embed + persist, cache invalidation.

The LLM client and the embedder are mocked at their module-level
factories (``app.llm.memory.get_dm_client`` /
``app.llm.memory.get_embedder``); both have process-wide singletons that
test isolation requires we swap rather than mutate.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import WorldFact
from app.llm.memory import (
    EmbeddingDimensionMismatchError,
    WorldFactRetriever,
    extract_and_persist_facts,
    maybe_regenerate_session_summary,
    recent_turns,
    regenerate_campaign_summary,
)
from tests.orchestrator.factories import (
    make_campaign,
    make_message,
    make_session,
    make_user,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedder(mapping: dict[str, np.ndarray] | None = None, *, dim: int = 4) -> MagicMock:
    """Build a stand-in for an :class:`Embedder`.

    ``mapping`` keys are exact text inputs; the matching value is
    returned as a single (1, dim) row when that text is embedded. Any
    missing key returns a unit vector along axis 0 (deterministic but
    distinct from the seeded inputs).
    """

    mapping = mapping or {}

    async def _embed(texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            if t in mapping:
                vec = mapping[t]
            else:
                vec = np.zeros(dim, dtype=np.float32)
                vec[0] = 1.0
            arr = np.asarray(vec, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            if n > 0:
                arr = arr / n
            rows.append(arr)
        return np.stack(rows).astype(np.float32) if rows else np.zeros((0, dim), dtype=np.float32)

    embedder = MagicMock()
    embedder.dim = dim
    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


def _vec_bytes(vec: np.ndarray) -> bytes:
    """L2-normalise then serialise to ``float32`` bytes for ``WorldFact.embedding``."""

    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n > 0:
        arr = arr / n
    return arr.astype(np.float32).tobytes()


async def _add_world_fact(
    db: Any,
    *,
    campaign_id: str,
    fact: str,
    vec: np.ndarray,
    tags: list[str] | None = None,
    importance: int = 5,
) -> WorldFact:
    arr = np.asarray(vec, dtype=np.float32)
    row = WorldFact(
        campaign_id=campaign_id,
        fact=fact,
        embedding=_vec_bytes(arr),
        embedding_dim=int(arr.shape[0]),
        tags=tags or [],
        importance=importance,
    )
    db.add(row)
    await db.flush()
    return row


# ---------------------------------------------------------------------------
# recent_turns — the existing Phase 2 surface, default bumped to N=40
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_turns_returns_chronological_order(db_session) -> None:  # type: ignore[no-untyped-def]
    """Last N rows in ``created_at`` ascending order."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)

    contents = ["one", "two", "three", "four"]
    for content in contents:
        await make_message(
            db_session,
            session_id=session.id,
            sender_kind="player",
            content=content,
        )
        # SQLite's strftime default is millisecond-precision; an
        # explicit await yields the loop and gives the next row a
        # distinct timestamp. Without this, in-memory tests can land
        # multiple inserts in the same millisecond and the order test
        # becomes flaky.
        await asyncio.sleep(0.005)
    await db_session.commit()

    rows = await recent_turns(db_session, session_id=session.id, n=40)
    assert [r.content for r in rows] == contents


@pytest.mark.asyncio
async def test_recent_turns_respects_limit(db_session) -> None:  # type: ignore[no-untyped-def]
    """``n`` truncates from the *front* of history (oldest dropped)."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)

    for i in range(5):
        await make_message(
            db_session,
            session_id=session.id,
            sender_kind="player",
            content=f"msg-{i}",
        )
        await asyncio.sleep(0.005)
    await db_session.commit()

    rows = await recent_turns(db_session, session_id=session.id, n=3)
    assert [r.content for r in rows] == ["msg-2", "msg-3", "msg-4"]


@pytest.mark.asyncio
async def test_recent_turns_zero_returns_empty(db_session) -> None:  # type: ignore[no-untyped-def]
    """``n=0`` short-circuits without a query — returns empty list."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    rows = await recent_turns(db_session, session_id=session.id, n=0)
    assert rows == []


@pytest.mark.asyncio
async def test_recent_turns_filters_by_session(db_session) -> None:  # type: ignore[no-untyped-def]
    """Messages from a different session must not bleed in."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session_a = await make_session(db_session, campaign_id=campaign.id)
    session_b = await make_session(db_session, campaign_id=campaign.id)

    await make_message(db_session, session_id=session_a.id, sender_kind="player", content="A1")
    await make_message(db_session, session_id=session_b.id, sender_kind="player", content="B1")
    await db_session.commit()

    rows = await recent_turns(db_session, session_id=session_a.id, n=10)
    assert [r.content for r in rows] == ["A1"]


def test_recent_turns_default_is_forty() -> None:
    """Phase 3 bumped the default from 20 (Phase 2 conservative) to 40."""

    import inspect

    sig = inspect.signature(recent_turns)
    assert sig.parameters["n"].default == 40


# ---------------------------------------------------------------------------
# WorldFactRetriever — cosine math, cache, dim mismatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topk_orders_by_cosine_with_known_unit_vectors(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hand-computed expected order for a small fixture matrix.

    Five facts at unit-length vectors in 4-D, query ``[1, 0, 0, 0]``.
    Cosine = dot product (vectors pre-normalised); rank should be by
    first-component magnitude.
    """

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)

    fixtures = [
        ("strong-match", np.array([1.0, 0.0, 0.0, 0.0])),
        ("weak-match", np.array([0.4, 0.9, 0.1, 0.0])),
        ("orthogonal", np.array([0.0, 1.0, 0.0, 0.0])),
        ("near-match", np.array([0.9, 0.4, 0.0, 0.0])),
        ("opposite", np.array([-1.0, 0.0, 0.0, 0.0])),
    ]
    for fact, vec in fixtures:
        await _add_world_fact(db_session, campaign_id=campaign.id, fact=fact, vec=vec)
    await db_session.commit()

    embedder = _make_embedder({"find me": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}, dim=4)
    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: embedder)

    retriever = WorldFactRetriever()
    hits = await retriever.topk(db_session, campaign.id, "find me", k=3)

    # Expected ranking by hand-computed cosine on normalised vectors:
    #   strong-match (1.0) > near-match (~0.913) > weak-match (~0.404)
    #   > orthogonal (0.0) > opposite (-1.0)
    assert [h.fact for h in hits] == ["strong-match", "near-match", "weak-match"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[1].score > hits[2].score
    assert hits[2].score > 0.0


@pytest.mark.asyncio
async def test_topk_returns_all_when_k_exceeds_n(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)

    await _add_world_fact(
        db_session,
        campaign_id=campaign.id,
        fact="only fact",
        vec=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    await db_session.commit()

    embedder = _make_embedder({"q": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}, dim=4)
    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: embedder)

    retriever = WorldFactRetriever()
    hits = await retriever.topk(db_session, campaign.id, "q", k=10)
    assert len(hits) == 1
    assert hits[0].fact == "only fact"


@pytest.mark.asyncio
async def test_topk_empty_campaign_returns_empty(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    await db_session.commit()

    embedder = _make_embedder(dim=4)
    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: embedder)

    retriever = WorldFactRetriever()
    hits = await retriever.topk(db_session, campaign.id, "anything", k=5)
    assert hits == []


@pytest.mark.asyncio
async def test_topk_zero_query_vector_returns_empty(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the embedder returns all-zeros, retrieval declines to rank."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    await _add_world_fact(
        db_session, campaign_id=campaign.id, fact="x", vec=np.array([1.0, 0.0, 0.0, 0.0])
    )
    await db_session.commit()

    class _ZeroEmbedder:
        dim = 4

        async def embed(self, texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: _ZeroEmbedder())

    retriever = WorldFactRetriever()
    hits = await retriever.topk(db_session, campaign.id, "", k=5)
    assert hits == []


@pytest.mark.asyncio
async def test_cache_loaded_once_then_invalidated(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First topk for a campaign hits the DB; second doesn't.
    ``invalidate`` resets the cache."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    await _add_world_fact(
        db_session, campaign_id=campaign.id, fact="a", vec=np.array([1.0, 0.0, 0.0, 0.0])
    )
    await db_session.commit()

    embedder = _make_embedder({"q": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}, dim=4)
    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: embedder)

    retriever = WorldFactRetriever()
    load_calls = 0
    real_load = retriever._load

    async def counting_load(db: Any, campaign_id: str) -> Any:
        nonlocal load_calls
        load_calls += 1
        return await real_load(db, campaign_id)

    retriever._load = counting_load  # type: ignore[method-assign]

    await retriever.topk(db_session, campaign.id, "q", k=3)
    assert load_calls == 1
    await retriever.topk(db_session, campaign.id, "q", k=3)
    assert load_calls == 1, "second call should hit the cache, not _load"

    retriever.invalidate(campaign.id)
    await retriever.topk(db_session, campaign.id, "q", k=3)
    assert load_calls == 2


@pytest.mark.asyncio
async def test_inconsistent_embedding_dim_raises(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed embedding_dim values within one campaign are a hard error.
    Switching the embedding model invalidates every existing fact and we
    refuse to silently produce wrong rankings."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)

    # First row at dim=4, second at dim=8 — this should never happen
    # under normal flow but if it does, we want a loud failure.
    await _add_world_fact(
        db_session, campaign_id=campaign.id, fact="dim4", vec=np.array([1.0, 0.0, 0.0, 0.0])
    )
    arr8 = np.zeros(8, dtype=np.float32)
    arr8[0] = 1.0
    db_session.add(
        WorldFact(
            campaign_id=campaign.id,
            fact="dim8",
            embedding=arr8.tobytes(),
            embedding_dim=8,
            tags=[],
            importance=5,
        )
    )
    await db_session.commit()

    embedder = _make_embedder({"q": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}, dim=4)
    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: embedder)

    retriever = WorldFactRetriever()
    with pytest.raises(EmbeddingDimensionMismatchError):
        await retriever.topk(db_session, campaign.id, "q", k=3)


# ---------------------------------------------------------------------------
# Session summary regeneration
# ---------------------------------------------------------------------------


def _make_dm_client(*responses: str) -> MagicMock:
    """A stand-in :class:`DmClient` whose ``complete`` returns each
    seeded response in order. Tracks call count for assertions."""

    iterator = iter(responses)

    async def _complete(messages: list[dict[str, Any]], **_: Any) -> str:
        try:
            return next(iterator)
        except StopIteration as exc:
            raise AssertionError("DmClient.complete called more times than seeded") from exc

    client = MagicMock()
    client.complete = AsyncMock(side_effect=_complete)
    return client


@pytest.mark.asyncio
async def test_session_summary_no_op_below_trigger(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the trigger boundary, the function returns ``None`` without
    invoking the LLM."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    for i in range(3):
        await make_message(db_session, session_id=session.id, sender_kind="player", content=f"p{i}")
    await db_session.commit()

    client = _make_dm_client()  # zero seeded responses; any call asserts
    monkeypatch.setattr("app.llm.memory.get_dm_client", lambda: client)

    out = await maybe_regenerate_session_summary(
        db_session, session_id=session.id, every_n_turns=20
    )
    assert out is None
    client.complete.assert_not_called()


@pytest.mark.asyncio
async def test_session_summary_regenerates_at_trigger(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the trigger boundary, the LLM is called and the result is
    persisted to ``sessions.summary``."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    for i in range(4):
        await make_message(db_session, session_id=session.id, sender_kind="player", content=f"p{i}")
    await db_session.commit()

    client = _make_dm_client("THE PARTY DELVES.")
    monkeypatch.setattr("app.llm.memory.get_dm_client", lambda: client)

    out = await maybe_regenerate_session_summary(db_session, session_id=session.id, every_n_turns=4)
    assert out == "THE PARTY DELVES."
    await db_session.refresh(session)
    assert session.summary == "THE PARTY DELVES."
    client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_session_summary_per_session_lock_collapses_concurrent_calls(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two coroutines that race the same trigger only invoke the LLM once.

    The lock is per-session-id; both coroutines see the same
    persisted summary, but ``client.complete`` is called exactly once.
    The lock implementation is module-level state, so we reset it via
    the singleton-clear helper before this test.
    """

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    for i in range(2):
        await make_message(db_session, session_id=session.id, sender_kind="player", content=f"p{i}")
    await db_session.commit()

    # Reset the module-level lock dict to ensure no carry-over from a
    # previous test that touched the same session id (shouldn't happen
    # but UUIDv7s are time-ordered and tests share the loop).
    from app.llm import memory as memory_module

    memory_module._session_summary_locks.clear()

    started = asyncio.Event()
    permit = asyncio.Event()

    async def _slow_complete(messages: list[dict[str, Any]], **_: Any) -> str:
        started.set()
        await permit.wait()
        return "SLOW SUMMARY"

    client = MagicMock()
    client.complete = AsyncMock(side_effect=_slow_complete)
    monkeypatch.setattr("app.llm.memory.get_dm_client", lambda: client)

    task_a = asyncio.create_task(
        maybe_regenerate_session_summary(db_session, session_id=session.id, every_n_turns=2)
    )
    # Wait until the first task is parked inside the LLM call.
    await started.wait()

    task_b = asyncio.create_task(
        maybe_regenerate_session_summary(db_session, session_id=session.id, every_n_turns=2)
    )
    # Yield enough times for B to enter and block on the lock.
    for _ in range(5):
        await asyncio.sleep(0)

    permit.set()
    result_a = await task_a
    result_b = await task_b

    assert result_a == "SLOW SUMMARY"
    # B must NOT call complete a second time; it observes A's persisted
    # summary and returns it.
    assert result_b == "SLOW SUMMARY"
    assert client.complete.await_count == 1


# ---------------------------------------------------------------------------
# Campaign summary regeneration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_campaign_summary_persists_and_uses_session_summaries(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id, long_summary="Prior arc.")
    s1 = await make_session(db_session, campaign_id=campaign.id, summary="Session 1 summary.")
    s2 = await make_session(db_session, campaign_id=campaign.id, summary="Session 2 summary.")
    await db_session.commit()
    assert s1 is not None and s2 is not None  # silence linter

    captured: list[list[dict[str, Any]]] = []

    async def _complete(messages: list[dict[str, Any]], **_: Any) -> str:
        captured.append(messages)
        return "NEW LONG SUMMARY"

    client = MagicMock()
    client.complete = AsyncMock(side_effect=_complete)
    monkeypatch.setattr("app.llm.memory.get_dm_client", lambda: client)

    new_long = await regenerate_campaign_summary(db_session, campaign_id=campaign.id)
    assert new_long == "NEW LONG SUMMARY"
    await db_session.refresh(campaign)
    assert campaign.long_summary == "NEW LONG SUMMARY"

    # The user message contains both the prior long summary and each
    # session's summary, in chronological order.
    assert len(captured) == 1
    user_text = captured[0][1]["content"]
    assert "Prior arc." in user_text
    assert "Session 1 summary." in user_text
    assert "Session 2 summary." in user_text


# ---------------------------------------------------------------------------
# Fact extractor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fact_extractor_persists_and_invalidates_cache(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: parses JSON, embeds each fact, persists, invalidates
    the per-campaign cache."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    payload = json.dumps(
        [
            {"fact": "The barkeep owes the party 5gp.", "tags": ["npc"], "importance": 4},
            {
                "fact": "There is a hidden door behind the bar.",
                "tags": ["location"],
                "importance": 7,
            },
        ]
    )
    client = _make_dm_client(payload)
    monkeypatch.setattr("app.llm.memory.get_dm_client", lambda: client)

    embedder = _make_embedder(dim=4)
    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: embedder)

    invalidated: list[str] = []

    class _FakeRetriever:
        def invalidate(self, cid: str) -> None:
            invalidated.append(cid)

    monkeypatch.setattr("app.llm.memory.get_world_fact_retriever", lambda: _FakeRetriever())

    persisted = await extract_and_persist_facts(
        db_session,
        session_id=session.id,
        player_action="I tip the barkeep heavily.",
        dm_response="He nods, grateful, and mutters about a back room.",
    )

    assert len(persisted) == 2
    assert "The barkeep owes the party 5gp." in persisted

    count_stmt = select(func.count(WorldFact.id)).where(WorldFact.campaign_id == campaign.id)
    count = int((await db_session.execute(count_stmt)).scalar_one())
    assert count == 2

    rows = list(
        (
            await db_session.scalars(select(WorldFact).where(WorldFact.campaign_id == campaign.id))
        ).all()
    )
    for row in rows:
        # Embedding bytes are float32 of length dim*4.
        assert len(row.embedding) == row.embedding_dim * 4
        # And L2-normalised: re-decode and check the norm.
        decoded = np.frombuffer(row.embedding, dtype=np.float32)
        assert np.linalg.norm(decoded) == pytest.approx(1.0, abs=1e-5)
        assert row.source_session_id == session.id

    assert invalidated == [campaign.id]


@pytest.mark.asyncio
async def test_fact_extractor_handles_fenced_json(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nemotron sometimes wraps the JSON in a ```json fence; the parser
    strips it cleanly."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    payload = (
        "Sure, here's the JSON:\n"
        "```json\n"
        '[{"fact": "A wraith lurks in the keep.", "tags": ["lore"], "importance": 8}]\n'
        "```"
    )
    client = _make_dm_client(payload)
    monkeypatch.setattr("app.llm.memory.get_dm_client", lambda: client)
    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: _make_embedder(dim=4))
    monkeypatch.setattr("app.llm.memory.get_world_fact_retriever", lambda: MagicMock())

    persisted = await extract_and_persist_facts(
        db_session,
        session_id=session.id,
        player_action="I open the door.",
        dm_response="A cold wind rushes out.",
    )
    assert persisted == ["A wraith lurks in the keep."]


@pytest.mark.asyncio
async def test_fact_extractor_handles_trailing_prose(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nemotron sometimes appends commentary after the JSON; the parser
    isolates the array via first-``[`` / last-``]`` heuristic."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    payload = (
        '[{"fact": "Found a silver key.", "tags": ["item"], "importance": 6}]'
        "\n\nLet me know if that captures everything!"
    )
    client = _make_dm_client(payload)
    monkeypatch.setattr("app.llm.memory.get_dm_client", lambda: client)
    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: _make_embedder(dim=4))
    monkeypatch.setattr("app.llm.memory.get_world_fact_retriever", lambda: MagicMock())

    persisted = await extract_and_persist_facts(
        db_session,
        session_id=session.id,
        player_action="I search the corpse.",
        dm_response="You find a silver key on a thin chain.",
    )
    assert persisted == ["Found a silver key."]


@pytest.mark.asyncio
async def test_fact_extractor_malformed_json_logs_and_returns(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the response is unparseable, log a warning and persist nothing —
    the caller (a fire-and-forget background task) must NEVER break the
    next player turn."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    client = _make_dm_client("not json at all, just words and prose")
    monkeypatch.setattr("app.llm.memory.get_dm_client", lambda: client)
    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: _make_embedder(dim=4))

    invalidated: list[str] = []

    class _FakeRetriever:
        def invalidate(self, cid: str) -> None:
            invalidated.append(cid)

    monkeypatch.setattr("app.llm.memory.get_world_fact_retriever", lambda: _FakeRetriever())

    with caplog.at_level("WARNING", logger="app.llm.memory"):
        persisted = await extract_and_persist_facts(
            db_session,
            session_id=session.id,
            player_action="I do nothing notable.",
            dm_response="Nothing happens.",
        )

    assert persisted == []
    count_stmt = select(func.count(WorldFact.id)).where(WorldFact.campaign_id == campaign.id)
    assert int((await db_session.execute(count_stmt)).scalar_one()) == 0
    assert any("JSON parse failed" in rec.message for rec in caplog.records)
    # No persistence -> no cache invalidation either.
    assert invalidated == []


@pytest.mark.asyncio
async def test_fact_extractor_empty_array_is_a_no_op(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``[]`` is the documented "nothing notable happened" response."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    client = _make_dm_client("[]")
    monkeypatch.setattr("app.llm.memory.get_dm_client", lambda: client)
    embedder = _make_embedder(dim=4)
    monkeypatch.setattr("app.llm.memory.get_embedder", lambda: embedder)
    monkeypatch.setattr("app.llm.memory.get_world_fact_retriever", lambda: MagicMock())

    persisted = await extract_and_persist_facts(
        db_session,
        session_id=session.id,
        player_action="I sit by the fire.",
        dm_response="The fire crackles.",
    )
    assert persisted == []
    embedder.embed.assert_not_called()
    count_stmt = select(func.count(WorldFact.id)).where(WorldFact.campaign_id == campaign.id)
    assert int((await db_session.execute(count_stmt)).scalar_one()) == 0
