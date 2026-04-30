"""Tests for ``app.orchestrator.handlers.*``.

Each handler gets exercised with the in-memory ``db_session``
fixture. The handlers expect to be called inside a dispatch context
(``app.orchestrator.context.with_dispatch_context``) — tests bind one
explicitly. The orchestrator's transaction discipline is tested in
``test_dm.py``; here we exercise the per-handler logic.

Per ``.claude/agents/test-writer.md``: every state-mutating handler
gets a "LLM tried to lie" test that proves the handler reads from the
database, not from any LLM-supplied current state.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.db.models import Character, DiceRoll, Encounter, SessionMessage
from app.llm.tools import (
    ApplyDamage,
    DiceTarget,
    EncounterMonster,
    EndEncounter,
    Heal,
    RequestDiceRoll,
    StartEncounter,
    TransitionLocation,
    Whisper,
    get_handler,
)
from app.orchestrator.context import DispatchContext, with_dispatch_context
from tests.orchestrator.factories import (
    make_campaign,
    make_character,
    make_encounter,
    make_location,
    make_session,
    make_user,
)


def _ctx(session_id: str, *, character_id: str | None = None) -> DispatchContext:
    return DispatchContext(
        session_id=session_id,
        sender_user_id="test-user",
        sender_character_id=character_id,
    )


# ---------------------------------------------------------------------------
# request_dice_roll
# ---------------------------------------------------------------------------


class TestRequestDiceRoll:
    @pytest.mark.asyncio
    async def test_persists_dice_roll(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("request_dice_roll")
        assert handler is not None

        args = RequestDiceRoll(
            expression="1d20+3",
            purpose="climb the wall",
            actor="dm",
            target=DiceTarget(kind="dc", value=12),
        )

        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        rows = list((await db_session.scalars(select(DiceRoll))).all())
        assert len(rows) == 1
        roll = rows[0]
        assert roll.session_id == session.id
        assert roll.expression == "1d20+3"
        assert roll.purpose == "climb the wall"
        assert roll.actor_kind == "dm"
        # Result content is human-readable and cites the result.
        assert "1d20+3" in result.content
        assert "climb the wall" in result.content
        # Side-effects record carries structured details.
        assert result.side_effects["kind"] == "dice_roll"
        assert result.side_effects["expression"] == "1d20+3"

    @pytest.mark.asyncio
    async def test_actor_character_persists_actor_id(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(db_session, user_id=user.id, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("request_dice_roll")
        assert handler is not None
        args = RequestDiceRoll(
            expression="1d20",
            purpose="attack",
            actor=char.id,
        )
        with with_dispatch_context(_ctx(session.id)):
            await handler.fn(db_session, args)
        await db_session.commit()

        roll = (await db_session.scalars(select(DiceRoll))).one()
        assert roll.actor_kind == "character"
        assert roll.actor_id == char.id


# ---------------------------------------------------------------------------
# apply_damage
# ---------------------------------------------------------------------------


class TestApplyDamage:
    @pytest.mark.asyncio
    async def test_reads_hp_from_db_and_persists(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(
            db_session,
            user_id=user.id,
            campaign_id=campaign.id,
            hp_current=10,
            hp_max=10,
        )
        await db_session.commit()

        handler = get_handler("apply_damage")
        assert handler is not None
        args = ApplyDamage(target_id=char.id, amount=3, source="goblin scimitar")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.hp_current == 7
        assert "took 3 damage" in result.content.lower() or "took 3" in result.content
        assert result.side_effects["kind"] == "state_update"

    @pytest.mark.asyncio
    async def test_llm_tried_to_lie_reads_from_db(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """The args don't carry current HP — but if the LLM also narrated
        that the source was 'a fall from 100ft' or similar, the handler
        must still subtract from the real hp_current=10, never hp=100.

        We simulate this by checking that even if the LLM picks an
        ``amount`` value that would imply a different starting HP, the
        handler computes ``hp_current - amount`` rigorously.
        """

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(
            db_session,
            user_id=user.id,
            campaign_id=campaign.id,
            hp_current=10,
            hp_max=20,
        )
        await db_session.commit()

        handler = get_handler("apply_damage")
        assert handler is not None
        # The LLM has narrated that the character is at 100 HP and takes 3
        # damage. The handler must ignore the narrative claim and use the
        # DB's hp_current=10.
        args = ApplyDamage(
            target_id=char.id,
            amount=3,
            source="LLM-claimed 100ft fall",
        )
        with with_dispatch_context(_ctx(session.id)):
            await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.hp_current == 7  # 10 - 3, not 100 - 3

    @pytest.mark.asyncio
    async def test_zero_hp_triggers_death_table(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(
            db_session,
            user_id=user.id,
            campaign_id=campaign.id,
            hp_current=2,
            hp_max=20,
        )
        await db_session.commit()

        handler = get_handler("apply_damage")
        assert handler is not None
        args = ApplyDamage(target_id=char.id, amount=5, source="critical hit")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        # Death table side-effect record must be present.
        assert result.side_effects["dropped_to_zero"] is True
        assert "death_outcome" in result.side_effects
        # HP either 0, 1, or None depending on outcome.
        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.hp_current in (0, 1)
        if result.side_effects["death_outcome"] == "dead":
            assert refreshed.status == "dead"

    @pytest.mark.asyncio
    async def test_unknown_target_returns_clean_error(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("apply_damage")
        assert handler is not None
        args = ApplyDamage(target_id="nonexistent", amount=1, source="x")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        assert result.side_effects.get("kind") == "error"
        assert "no character" in result.content.lower()


# ---------------------------------------------------------------------------
# heal
# ---------------------------------------------------------------------------


class TestHeal:
    @pytest.mark.asyncio
    async def test_caps_at_hp_max(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Healing past hp_max clamps to hp_max."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(
            db_session,
            user_id=user.id,
            campaign_id=campaign.id,
            hp_current=8,
            hp_max=10,
        )
        await db_session.commit()

        handler = get_handler("heal")
        assert handler is not None
        args = Heal(target_id=char.id, amount=100, source="potion")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.hp_current == 10  # capped, not 108
        assert result.side_effects["new"] == 10

    @pytest.mark.asyncio
    async def test_llm_tried_to_lie_about_hp_max(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """If the LLM narrates ``"healed back to full at 200/200"``, the
        handler still respects the DB's hp_max and caps at that value."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(
            db_session,
            user_id=user.id,
            campaign_id=campaign.id,
            hp_current=5,
            hp_max=10,
        )
        await db_session.commit()

        handler = get_handler("heal")
        assert handler is not None
        # The LLM might claim character has hp_max=200; the args carry
        # only target_id and amount — the handler reads hp_max from
        # the DB. Set ``amount`` huge to make this concrete.
        args = Heal(target_id=char.id, amount=999, source="LLM-claimed full restore")
        with with_dispatch_context(_ctx(session.id)):
            await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.hp_current == 10  # DB's hp_max=10, not 999

    @pytest.mark.asyncio
    async def test_refuses_to_heal_downed(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(
            db_session,
            user_id=user.id,
            campaign_id=campaign.id,
            hp_current=0,
            hp_max=10,
        )
        await db_session.commit()

        handler = get_handler("heal")
        assert handler is not None
        args = Heal(target_id=char.id, amount=5)
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        # Not mutated.
        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.hp_current == 0
        assert result.side_effects.get("kind") == "error"


# ---------------------------------------------------------------------------
# transition_location
# ---------------------------------------------------------------------------


class TestTransitionLocation:
    @pytest.mark.asyncio
    async def test_updates_session_and_logs_message(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        loc = await make_location(db_session, campaign_id=campaign.id, name="The Crypt")
        await db_session.commit()

        handler = get_handler("transition_location")
        assert handler is not None
        args = TransitionLocation(location_id=loc.id, description="You descend the cracked stairs.")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        from app.db.models import Session as DmSession

        refreshed = await db_session.get(DmSession, session.id)
        assert refreshed is not None
        assert refreshed.current_location_id == loc.id

        msgs = list(
            (
                await db_session.scalars(
                    select(SessionMessage).where(SessionMessage.session_id == session.id)
                )
            ).all()
        )
        assert any(m.sender_kind == "system" and "The Crypt" in m.content for m in msgs)
        assert "The Crypt" in result.content

    @pytest.mark.asyncio
    async def test_unknown_location_refused(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("transition_location")
        assert handler is not None
        args = TransitionLocation(location_id="bogus-id", description="x")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        assert result.side_effects.get("kind") == "error"
        assert "does not exist" in result.content

    @pytest.mark.asyncio
    async def test_cross_campaign_location_refused(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """A location in a different campaign must be refused."""

        user = await make_user(db_session)
        campaign_a = await make_campaign(db_session, owner_id=user.id, name="A")
        campaign_b = await make_campaign(db_session, owner_id=user.id, name="B")
        session = await make_session(db_session, campaign_id=campaign_a.id)
        loc_b = await make_location(db_session, campaign_id=campaign_b.id)
        await db_session.commit()

        handler = get_handler("transition_location")
        assert handler is not None
        args = TransitionLocation(location_id=loc_b.id, description="x")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        assert result.side_effects.get("kind") == "error"


# ---------------------------------------------------------------------------
# whisper
# ---------------------------------------------------------------------------


class TestWhisper:
    @pytest.mark.asyncio
    async def test_persists_audience(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(db_session, user_id=user.id, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("whisper")
        assert handler is not None
        args = Whisper(character_id=char.id, content="A figure shadows you.")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        msg = (
            await db_session.scalars(
                select(SessionMessage).where(SessionMessage.sender_kind == "dm")
            )
        ).one()
        assert msg.audience == [char.id]
        assert msg.content == "A figure shadows you."
        assert result.side_effects["kind"] == "whisper"


# ---------------------------------------------------------------------------
# start_encounter / end_encounter
# ---------------------------------------------------------------------------


class TestEncounters:
    @pytest.mark.asyncio
    async def test_start_encounter_rolls_initiative(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("start_encounter")
        assert handler is not None
        args = StartEncounter(
            name="Goblins!",
            monsters=[EncounterMonster(name="goblin", count=3, hp=5)],
        )
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        enc = (await db_session.scalars(select(Encounter))).one()
        assert enc.name == "Goblins!"
        assert enc.status == "active"
        # Initiative populated for all 3 goblins.
        assert len(enc.initiative) == 3
        assert all("initiative" in entry for entry in enc.initiative)
        assert "Goblins!" in result.content

    @pytest.mark.asyncio
    async def test_start_encounter_merges_alive_pcs_into_initiative(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Phase 4: alive PCs in the campaign join initiative
        automatically. The PC's ``participant_id`` is the character row
        id so the WS hub's gate can match a player's character_id
        against ``current_turn``.
        """

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        # Two alive PCs and one dead PC — only alive ones should join.
        alive1 = await make_character(
            db_session, user_id=user.id, campaign_id=campaign.id, name="Alive Alice"
        )
        alive2 = await make_character(
            db_session, user_id=user.id, campaign_id=campaign.id, name="Alive Bob"
        )
        await make_character(
            db_session,
            user_id=user.id,
            campaign_id=campaign.id,
            name="Dead Don",
            status="dead",
        )
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("start_encounter")
        assert handler is not None
        args = StartEncounter(
            name="Goblins!",
            monsters=[EncounterMonster(name="goblin", count=2, hp=5)],
        )
        with with_dispatch_context(_ctx(session.id)):
            await handler.fn(db_session, args)
        await db_session.commit()

        enc = (await db_session.scalars(select(Encounter))).one()
        # 2 goblins + 2 alive PCs + 0 dead PCs = 4 entries.
        assert len(enc.initiative) == 4
        ids = {e["participant_id"] for e in enc.initiative}
        assert alive1.id in ids
        assert alive2.id in ids
        # Player flag matches the participant kind.
        for entry in enc.initiative:
            if entry["participant_id"] in {alive1.id, alive2.id}:
                assert entry["is_player"] is True
            else:
                assert entry["is_player"] is False

    @pytest.mark.asyncio
    async def test_end_encounter_flips_status(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        enc = await make_encounter(db_session, session_id=session.id)
        await db_session.commit()

        handler = get_handler("end_encounter")
        assert handler is not None
        args = EndEncounter(encounter_id=enc.id, outcome="victory", summary="Slew them all.")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Encounter, enc.id)
        assert refreshed is not None
        assert refreshed.status == "victory"
        assert "victory" in result.content

    @pytest.mark.asyncio
    async def test_end_encounter_rejects_cross_session(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """An encounter from a different session can't be ended through
        this dispatch — defensive against tool-arg confusion."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session_a = await make_session(db_session, campaign_id=campaign.id)
        session_b = await make_session(db_session, campaign_id=campaign.id)
        enc = await make_encounter(db_session, session_id=session_b.id)
        await db_session.commit()

        handler = get_handler("end_encounter")
        assert handler is not None
        args = EndEncounter(encounter_id=enc.id, outcome="victory")
        with with_dispatch_context(_ctx(session_a.id)):
            result = await handler.fn(db_session, args)
        assert result.side_effects.get("kind") == "error"
