"""Tests for ``app.llm.prompts.build_dm_prompt`` and section renderers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.llm.prompts import _render_characters, build_dm_prompt
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
async def test_opening_directive_renders_as_system_context(db_session) -> None:  # type: ignore[no-untyped-def]
    """Phase 6.8 Bug 3: an opening turn persists a ``sender_kind='system'``
    pseudo-message carrying the bootstrapping directive. The prompt
    builder must render that as a system-role message in the
    chat-completions list — NOT as a user-role message — so the DM
    treats it as engine context rather than something the player
    just said.
    """

    from app.db.models import SessionMessage

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    # Persist exactly the synthetic directive shape that take_turn(opening=True) writes.
    db_session.add(
        SessionMessage(
            session_id=session.id,
            sender_kind="system",
            sender_id=None,
            audience=[],
            content="[Session begins — set the opening scene for the party.]",
        )
    )
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)

    # The directive must appear as a system-role message in the
    # post-system-block tail (where recent turns get rendered).
    tail = messages[1:]
    directive_messages = [m for m in tail if "Session begins" in m.get("content", "")]
    assert directive_messages, "opening directive missing from prompt tail"
    for m in directive_messages:
        assert (
            m["role"] == "system"
        ), f"opening directive must render as system role, got {m['role']!r}"

    # And it must NOT appear as a user-role message anywhere.
    user_directives = [
        m for m in messages if m.get("role") == "user" and "Session begins" in m.get("content", "")
    ]
    assert (
        not user_directives
    ), "opening directive leaked as a user message — the DM would 'respond' to it"


@pytest.mark.asyncio
async def test_role_block_forbids_asking_player_for_ids(db_session) -> None:  # type: ignore[no-untyped-def]
    """Phase 6.8 Bug 4 fix: the DM must never ask the player for
    location_id, character_id, or any schema parameter — that's a
    fourth-wall break. The rule lands inside the [ROLE] block so it
    rides on every turn's prompt regardless of campaign state."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    assert messages[0]["role"] == "system"
    body = messages[0]["content"]

    # The role block must contain the discipline rule. Phrase-level
    # check — the test pins on the engine-rather-than-fiction wording
    # so a casual prose tweak doesn't silently lose the rule.
    role_start = body.index("[ROLE]")
    rules_start = body.index("[RULES SUMMARY]")
    role_block = body[role_start:rules_start]
    assert "Never ask the player for ids" in role_block
    assert "fourth-wall break" in role_block
    assert "engine resolves names" in role_block


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


# ---------------------------------------------------------------------------
# Phase 6.10 — speaker attribution (AGENTS.md invariant #17)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_role_block_explains_attribution_prefix(db_session) -> None:  # type: ignore[no-untyped-def]
    """Phase 6.10: the [ROLE] block tells the model how to interpret the
    ``[Name, Class]:`` prefix on player messages — otherwise the model
    would treat the brackets as part of the player's words.
    """

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    body = messages[0]["content"]

    role_start = body.index("[ROLE]")
    rules_start = body.index("[RULES SUMMARY]")
    role_block = body[role_start:rules_start]

    # The instruction must (a) name the prefix shape, (b) flag it as
    # engine metadata not fiction, (c) tell the model to address the
    # named character.
    assert "PLAYER ATTRIBUTION" in role_block
    assert "[Character Name, Class]" in role_block
    assert "engine metadata" in role_block
    assert "not fiction" in role_block.lower() or "NOT part of" in role_block


@pytest.mark.asyncio
async def test_player_message_carries_attribution_prefix(db_session) -> None:  # type: ignore[no-untyped-def]
    """Phase 6.10: a player message persisted with sender_id=character_id
    is rendered with ``[Name, Class]:`` prepended in the chat-completions
    list. Without this prefix the DM cannot disambiguate speakers in a
    multi-PC party."""

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    char = await make_character(
        db_session,
        user_id=user.id,
        campaign_id=campaign.id,
        name="Slowhand",
        class_name="Fighter",
    )
    session = await make_session(db_session, campaign_id=campaign.id)
    await make_message(
        db_session,
        session_id=session.id,
        sender_kind="player",
        sender_id=char.id,
        content="I draw my axe and charge the goblin.",
    )
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    user_messages = [m for m in messages[1:] if m["role"] == "user"]
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == (
        "[Slowhand, Fighter]: I draw my axe and charge the goblin."
    )


@pytest.mark.asyncio
async def test_attribution_disambiguates_multi_character_party(db_session) -> None:  # type: ignore[no-untyped-def]
    """Phase 6.10: the canonical multi-PC case — Slowhand and Lila both
    speak in turn, and each user message must carry the right
    attribution. This is the bug that motivated the fix.
    """

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    slowhand = await make_character(
        db_session,
        user_id=user.id,
        campaign_id=campaign.id,
        name="Slowhand",
        class_name="Fighter",
    )
    lila = await make_character(
        db_session,
        user_id=user.id,
        campaign_id=campaign.id,
        name="Lila",
        class_name="Magic-User",
    )
    session = await make_session(db_session, campaign_id=campaign.id)
    await make_message(
        db_session,
        session_id=session.id,
        sender_kind="player",
        sender_id=slowhand.id,
        content="I need help, I got beat up bad.",
    )
    await asyncio.sleep(0.005)
    await make_message(
        db_session,
        session_id=session.id,
        sender_kind="player",
        sender_id=lila.id,
        content="I rummage in my pack for a healing potion.",
    )
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    user_msgs = [m for m in messages[1:] if m["role"] == "user"]
    assert len(user_msgs) == 2
    assert user_msgs[0]["content"] == ("[Slowhand, Fighter]: I need help, I got beat up bad.")
    assert user_msgs[1]["content"] == (
        "[Lila, Magic-User]: I rummage in my pack for a healing potion."
    )


@pytest.mark.asyncio
async def test_attribution_resolves_dead_character_from_history(db_session) -> None:  # type: ignore[no-untyped-def]
    """Phase 6.10: a recently-deceased PC's earlier messages still carry
    their attribution. The character index includes all statuses so
    the prompt history stays consistent across the death event.
    """

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    mort = await make_character(
        db_session,
        user_id=user.id,
        campaign_id=campaign.id,
        name="Mort",
        class_name="Cleric",
        status="dead",
    )
    session = await make_session(db_session, campaign_id=campaign.id)
    await make_message(
        db_session,
        session_id=session.id,
        sender_kind="player",
        sender_id=mort.id,
        content="I cast cure light wounds on myself.",
    )
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    user_msgs = [m for m in messages[1:] if m["role"] == "user"]
    assert user_msgs[0]["content"] == ("[Mort, Cleric]: I cast cure light wounds on myself.")


@pytest.mark.asyncio
async def test_attribution_skipped_when_sender_id_unresolvable(db_session) -> None:  # type: ignore[no-untyped-def]
    """Phase 6.10: a player row whose sender_id doesn't resolve (legacy
    data, missing character) degrades gracefully to a bare user message
    rather than emitting a half-formed prefix. Better silent than wrong.
    """

    user = await make_user(db_session)
    campaign = await make_campaign(db_session, owner_id=user.id)
    session = await make_session(db_session, campaign_id=campaign.id)
    # No sender_id at all — same as legacy Phase 2 fixtures.
    await make_message(
        db_session,
        session_id=session.id,
        sender_kind="player",
        content="I look around.",
    )
    # And a sender_id that doesn't reference any character row.
    await make_message(
        db_session,
        session_id=session.id,
        sender_kind="player",
        sender_id="no-such-character-id",
        content="I keep looking.",
    )
    await db_session.commit()

    messages = await build_dm_prompt(db_session, session_id=session.id)
    user_msgs = [m for m in messages[1:] if m["role"] == "user"]
    # No prefix on either — graceful degradation, not a malformed
    # ``[None, None]:`` shape.
    assert all("[" not in m["content"][:1] for m in user_msgs)
    assert [m["content"] for m in user_msgs] == ["I look around.", "I keep looking."]


# ---------------------------------------------------------------------------
# Phase 6.13: _render_characters — pronouns and description rendering
# ---------------------------------------------------------------------------


def _make_char(**kwargs) -> SimpleNamespace:  # type: ignore[type-arg]
    """Minimal fake Character object for _render_characters unit tests.

    Only the fields _render_characters actually accesses are required;
    everything else gets a sensible default so the function doesn't blow up.
    """
    defaults = {
        "id": "test-id-0001",
        "name": "Brunhild",
        "race": "Human",
        "class_name": "Fighter",
        "level": 1,
        "hp_current": 8,
        "hp_max": 8,
        "ac": 14,
        "str_score": 14,
        "dex_score": 12,
        "con_score": 12,
        "int_score": 10,
        "wis_score": 10,
        "cha_score": 10,
        "status": "alive",
        "status_effects": [],
        "pronouns": None,
        "description": None,
    }
    return SimpleNamespace(**(defaults | kwargs))


def test_render_characters_with_pronouns_and_description() -> None:
    """Both pronouns and description appear in the rendered block."""

    ch = _make_char(
        pronouns="she/her",
        description="Dark braided hair, scar above left eye",
    )
    out = _render_characters([ch])
    assert "pronouns: she/her" in out
    assert "Appearance: Dark braided hair, scar above left eye" in out


def test_render_characters_null_presentation_omits_both() -> None:
    """Neither pronouns nor Appearance line renders when both are None."""

    ch = _make_char(pronouns=None, description=None)
    out = _render_characters([ch])
    assert "pronouns" not in out
    assert "Appearance" not in out


def test_render_characters_pronouns_only() -> None:
    """pronouns renders in the parenthetical; no Appearance line when description is None."""

    ch = _make_char(pronouns="they/them", description=None)
    out = _render_characters([ch])
    assert "pronouns: they/them" in out
    assert "Appearance" not in out


def test_render_characters_description_only() -> None:
    """Appearance line renders; no pronouns when pronouns is None."""

    ch = _make_char(pronouns=None, description="Tall with copper-red hair")
    out = _render_characters([ch])
    assert "pronouns" not in out
    assert "Appearance: Tall with copper-red hair" in out


def test_render_characters_empty_list_returns_none_placeholder() -> None:
    """Empty character list renders the canonical (none) placeholder."""

    out = _render_characters([])
    assert out == "(none)"


def test_render_characters_multiple_pcs_each_rendered() -> None:
    """All characters in the list appear in the output."""

    alice = _make_char(name="Alice", pronouns="she/her", description="Short and quick")
    bob = _make_char(name="Bob", pronouns=None, description=None)
    out = _render_characters([alice, bob])
    assert "Alice" in out
    assert "pronouns: she/her" in out
    assert "Appearance: Short and quick" in out
    assert "Bob" in out
