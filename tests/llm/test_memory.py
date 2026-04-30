"""Tests for ``app.llm.memory.recent_turns``."""

from __future__ import annotations

import asyncio

import pytest

from app.llm.memory import recent_turns
from tests.orchestrator.factories import (
    make_campaign,
    make_message,
    make_session,
    make_user,
)


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

    rows = await recent_turns(db_session, session_id=session.id, n=20)
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
