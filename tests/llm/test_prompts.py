"""Tests for ``app.llm.prompts.build_dm_prompt``."""

from __future__ import annotations

import asyncio

import pytest

from app.llm.prompts import build_dm_prompt
from tests.orchestrator.factories import (
    make_campaign,
    make_character,
    make_encounter,
    make_location,
    make_message,
    make_session,
    make_user,
)


@pytest.mark.asyncio
async def test_build_dm_prompt_layered_shape(db_session) -> None:  # type: ignore[no-untyped-def]
    """All ten layered sections appear in the system message body."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)

    assert messages[0]["role"] == "system"
    body = messages[0]["content"]
    for tag in (
        "[ROLE]",
        "[RULES SUMMARY]",
        "[HOUSE RULES]",
        "[CAMPAIGN]",
        "[CURRENT LOCATION]",
        "[ACTIVE PCs]",
        "[ACTIVE NPCs IN SCENE]",
        "[RECENT TURNS]",
        "[SESSION SO FAR]",
        "[RELEVANT WORLD FACTS]",
        "[ACTIVE ENCOUNTER]",
        "[MODULE]",
    ):
        assert tag in body, f"missing section {tag}"


@pytest.mark.asyncio
async def test_build_dm_prompt_renders_empty_sections(db_session) -> None:  # type: ignore[no-untyped-def]
    """Empty sections render as ``(none)`` rather than being skipped."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    body = messages[0]["content"]

    # Location (none), encounters (none), characters (none),
    # session-summary (none) all should be present.
    assert "[CURRENT LOCATION]\n(none)" in body
    assert "[ACTIVE PCs]\n(none)" in body
    assert "[ACTIVE ENCOUNTER]\n(none)" in body
    assert "[ACTIVE NPCs IN SCENE]\n(none)" in body


@pytest.mark.asyncio
async def test_build_dm_prompt_includes_active_pcs(db_session) -> None:  # type: ignore[no-untyped-def]
    """Active PCs are surfaced; dead ones are filtered out."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await make_character(
        db_session,
        user_id=user.id,
        campaign_id=campaign.id,
        name="Eira",
        status="alive",
    )
    await make_character(
        db_session,
        user_id=user.id,
        campaign_id=campaign.id,
        name="Mort",
        status="dead",
    )
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    body = messages[0]["content"]

    assert "Eira" in body
    assert "Mort" not in body  # dead characters filtered


@pytest.mark.asyncio
async def test_build_dm_prompt_includes_house_rules_defaults(db_session) -> None:  # type: ignore[no-untyped-def]
    """Spec §4 default house rules render even when the JSON column is empty."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id, house_rules={})
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    body = messages[0]["content"]

    assert "Death and Dismemberment" in body
    assert "XP for treasure" in body
    assert "Variable weapon damage" in body


@pytest.mark.asyncio
async def test_build_dm_prompt_recent_turns_alternate(db_session) -> None:  # type: ignore[no-untyped-def]
    """Recent turns render as alternating user/assistant messages, not text."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await make_message(db_session, session_id=session.id, sender_kind="player", content="hello")
    await asyncio.sleep(0.005)
    await make_message(db_session, session_id=session.id, sender_kind="dm", content="hi back")
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    # First is system, then alternating user/assistant.
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "hello"}
    assert messages[2] == {"role": "assistant", "content": "hi back"}


@pytest.mark.asyncio
async def test_build_dm_prompt_recent_turns_respects_limit(db_session) -> None:  # type: ignore[no-untyped-def]
    """``recent_turns_n`` truncates the verbatim tail."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    for i in range(5):
        await make_message(
            db_session,
            session_id=session.id,
            sender_kind="player",
            content=f"line-{i}",
        )
        await asyncio.sleep(0.005)
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id, recent_turns_n=2)
    user_messages = [m for m in messages[1:] if m["role"] == "user"]
    assert [m["content"] for m in user_messages] == ["line-3", "line-4"]


@pytest.mark.asyncio
async def test_build_dm_prompt_with_active_encounter(db_session) -> None:  # type: ignore[no-untyped-def]
    """Active encounter monsters and round are surfaced (DM-only)."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await make_encounter(db_session, session_id=session.id, name="Goblin ambush")
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    body = messages[0]["content"]
    assert "Goblin ambush" in body
    assert "round 1" in body


@pytest.mark.asyncio
async def test_build_dm_prompt_uses_current_location(db_session) -> None:  # type: ignore[no-untyped-def]
    """Session.current_location_id resolves to a location block."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    location = await make_location(
        db_session, campaign_id=campaign.id, name="Black Cave", description="dripping"
    )
    session = await make_session(
        db_session, campaign_id=campaign.id, current_location_id=location.id
    )
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    body = messages[0]["content"]
    assert "Black Cave" in body
    assert "dripping" in body


@pytest.mark.asyncio
async def test_build_dm_prompt_unknown_session_raises(db_session) -> None:  # type: ignore[no-untyped-def]
    """Missing session_id raises ValueError, not a hidden None traversal."""

    with pytest.raises(ValueError, match="unknown session_id"):
        await build_dm_prompt(db_session, session_id="nonexistent")
