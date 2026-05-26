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

from app.db.models import (
    Campaign,
    Character,
    DiceRoll,
    Encounter,
    GeneratedImage,
    Module,
    Npc,
    SessionMessage,
)
from app.images.portrait import reset_for_tests as reset_queue_client
from app.images.portrait import set_queue_client_for_tests
from app.llm.tools import (
    ApplyDamage,
    ApplyRevival,
    ApplyStatusEffect,
    ClearStatusEffect,
    DiceTarget,
    EncounterMonster,
    EndEncounter,
    GenerateSceneImage,
    Heal,
    MarkBeat,
    RequestDiceRoll,
    RevealSecret,
    SpawnNpc,
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

    @pytest.mark.asyncio
    async def test_name_match_resolves_to_existing_location(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """A name passed to the tool that closely matches an existing
        campaign location resolves to that location rather than
        creating a duplicate. This is the Bug 4 fix path: the DM
        references places by name, the engine resolves them."""

        from app.db.models import Location

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        existing = await make_location(db_session, campaign_id=campaign.id, name="Jeb's Smithy")
        await db_session.commit()

        handler = get_handler("transition_location")
        assert handler is not None
        # Slightly different casing to exercise normalised matching.
        args = TransitionLocation(name="jeb's smithy", description="Smoke and hammer.")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        # Same row, no duplicate created.
        rows = list((await db_session.scalars(select(Location))).all())
        assert len(rows) == 1
        assert rows[0].id == existing.id

        from app.db.models import Session as DmSession

        refreshed = await db_session.get(DmSession, session.id)
        assert refreshed is not None
        assert refreshed.current_location_id == existing.id
        assert result.side_effects["resolution"] == "name_match"

    @pytest.mark.asyncio
    async def test_name_create_inserts_new_location(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """A name with no close match in the campaign creates a fresh
        Location row with the supplied description and transitions to
        it. The DM never has to surface an id to the player; this is
        the no-existing-place path."""

        from app.db.models import Location

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        # Seed an unrelated location so the candidate set isn't empty.
        await make_location(db_session, campaign_id=campaign.id, name="The Keep Gate")
        await db_session.commit()

        handler = get_handler("transition_location")
        assert handler is not None
        args = TransitionLocation(
            name="The Old Crypt",
            description="Cold air, lichen, a single black door.",
        )
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        rows = list((await db_session.scalars(select(Location))).all())
        assert len(rows) == 2
        crypt = next(r for r in rows if r.name == "The Old Crypt")
        assert crypt.campaign_id == campaign.id
        assert crypt.description == "Cold air, lichen, a single black door."

        from app.db.models import Session as DmSession

        refreshed = await db_session.get(DmSession, session.id)
        assert refreshed is not None
        assert refreshed.current_location_id == crypt.id
        assert result.side_effects["resolution"] == "name_create"

    @pytest.mark.asyncio
    async def test_name_match_only_within_campaign(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """A location with a matching name in a *different* campaign
        must not be reused — the resolver creates a fresh row scoped
        to the active campaign."""

        from app.db.models import Location

        user = await make_user(db_session)
        campaign_a = await make_campaign(db_session, owner_id=user.id, name="A")
        campaign_b = await make_campaign(db_session, owner_id=user.id, name="B")
        session_a = await make_session(db_session, campaign_id=campaign_a.id)
        # Same name, different campaign — must not be reused.
        await make_location(db_session, campaign_id=campaign_b.id, name="The Tavern")
        await db_session.commit()

        handler = get_handler("transition_location")
        assert handler is not None
        args = TransitionLocation(name="The Tavern", description="Smoky.")
        with with_dispatch_context(_ctx(session_a.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        rows = list(
            (
                await db_session.scalars(
                    select(Location).where(Location.campaign_id == campaign_a.id)
                )
            ).all()
        )
        assert len(rows) == 1
        assert rows[0].name == "The Tavern"
        assert result.side_effects["resolution"] == "name_create"

    @pytest.mark.asyncio
    async def test_neither_id_nor_name_rejected_at_validation(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """A call with neither location_id nor name must fail Pydantic
        validation — the orchestrator's ``parse_tool_args`` is what
        the LLM hits, and a half-formed call should be a clean reject
        before the handler runs."""

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TransitionLocation(description="x")


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


# ---------------------------------------------------------------------------
# spawn_npc
# ---------------------------------------------------------------------------


class _FakeQueueClient:
    """Captures rpush calls so spawn_npc tests can assert (or assert
    absence of) a portrait enqueue without speaking to a real Valkey."""

    def __init__(self, *, raise_on_push: bool = False) -> None:
        self.pushed: list[tuple[str, bytes]] = []
        self.raise_on_push = raise_on_push

    async def rpush(self, key: str, value: bytes) -> int:
        if self.raise_on_push:
            raise RuntimeError("simulated valkey transport error")
        self.pushed.append((key, value))
        return len(self.pushed)

    async def aclose(self) -> None:
        return None


class TestSpawnNpc:
    @pytest.mark.asyncio
    async def test_creates_npc_row_and_enqueues_portrait(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Default ``auto_portrait=True``: row lands, portrait job
        appears on the queue with the NPC id as the subject FK target."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        fake_queue = _FakeQueueClient()
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("spawn_npc")
            assert handler is not None
            args = SpawnNpc(
                name="Castellan Thorvald",
                description="Greying veteran, missing two fingers.",
            )
            with with_dispatch_context(_ctx(session.id)):
                result = await handler.fn(db_session, args)
            await db_session.commit()
        finally:
            await reset_queue_client()

        # NPC row landed in the campaign.
        rows = list((await db_session.scalars(select(Npc))).all())
        assert len(rows) == 1
        npc = rows[0]
        assert npc.name == "Castellan Thorvald"
        assert npc.campaign_id == campaign.id
        assert npc.description and "Greying veteran" in npc.description

        # Portrait job pushed with subject_npc_id pointing at this NPC.
        assert len(fake_queue.pushed) == 1
        import json

        payload = json.loads(fake_queue.pushed[0][1])
        assert payload["subject_npc_id"] == npc.id
        assert payload["kind"] == "npc"
        assert payload["session_id"] == session.id

        # Side effects expose the new ids so the WS layer can render
        # an image_pending placeholder card.
        assert result.side_effects["kind"] == "npc_spawned"
        assert result.side_effects["npc_id"] == npc.id
        assert "portrait_image_id" in result.side_effects
        # Phase 8 Commit 2: brief carries the description so npc_introduced
        # WS message is self-contained (no DB round-trip from the client).
        assert result.side_effects["brief"] == "Greying veteran, missing two fingers."

    @pytest.mark.asyncio
    async def test_auto_portrait_false_skips_enqueue(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """The LLM should be able to opt out for transient walk-ons
        (a guard, a stable hand) where 17s of FLUX time on a portrait
        is wasted. ``auto_portrait=False`` must not push to the queue."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        fake_queue = _FakeQueueClient()
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("spawn_npc")
            assert handler is not None
            args = SpawnNpc(name="Stable Hand", auto_portrait=False)
            with with_dispatch_context(_ctx(session.id)):
                result = await handler.fn(db_session, args)
            await db_session.commit()
        finally:
            await reset_queue_client()

        # NPC row still lands.
        rows = list((await db_session.scalars(select(Npc))).all())
        assert len(rows) == 1

        # No portrait job pushed.
        assert fake_queue.pushed == []
        assert "portrait_image_id" not in result.side_effects

    @pytest.mark.asyncio
    async def test_queue_failure_keeps_npc_row(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """If the queue push fails (Valkey down, transport error),
        the NPC row must still survive — the LLM has already narrated
        the introduction. The portrait can be re-requested via the API
        endpoint after the operator fixes the transport.

        Critical: a queue-side failure must not roll back the DB row.
        That would be a worse outcome than a missing portrait."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        fake_queue = _FakeQueueClient(raise_on_push=True)
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("spawn_npc")
            assert handler is not None
            args = SpawnNpc(name="Lyra")
            with with_dispatch_context(_ctx(session.id)):
                result = await handler.fn(db_session, args)
            await db_session.commit()
        finally:
            await reset_queue_client()

        # NPC row still landed despite the queue failure.
        rows = list((await db_session.scalars(select(Npc))).all())
        assert len(rows) == 1
        assert rows[0].name == "Lyra"

        # Side effects don't claim a portrait when none was queued.
        assert "portrait_image_id" not in result.side_effects
        assert result.side_effects["kind"] == "npc_spawned"

    @pytest.mark.asyncio
    async def test_unknown_session_returns_clean_error(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Defensive: if the dispatch context names a session that
        doesn't exist, return a structured error rather than crashing.
        The orchestrator validates session before dispatch, so this is
        belt-and-braces."""

        fake_queue = _FakeQueueClient()
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("spawn_npc")
            assert handler is not None
            args = SpawnNpc(name="X")
            with with_dispatch_context(_ctx("does-not-exist")):
                result = await handler.fn(db_session, args)
        finally:
            await reset_queue_client()

        assert result.side_effects.get("kind") == "error"
        assert result.side_effects.get("reason") == "unknown_session"
        # No NPC row inserted, no portrait pushed.
        rows = list((await db_session.scalars(select(Npc))).all())
        assert rows == []
        assert fake_queue.pushed == []


# ---------------------------------------------------------------------------
# generate_scene_image
# ---------------------------------------------------------------------------


async def _make_canonical_image(db_session, *, campaign_id: str) -> str:  # type: ignore[no-untyped-def]
    """Insert a canonical image row that a character/NPC can FK to.
    Returns the image_id."""

    img = GeneratedImage(
        campaign_id=campaign_id,
        kind="npc",
        prompt="canon",
        prompt_hash="canon-hash-" + campaign_id[:8],
        file_path="/tmp/canon.png",
    )
    db_session.add(img)
    await db_session.flush()
    return img.id


class TestGenerateSceneImage:
    @pytest.mark.asyncio
    async def test_no_reference_enqueues_generate_job(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Without any reference, the job lands as a plain /generate
        (no reference_image_id, no edit_instruction)."""

        import json

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        fake_queue = _FakeQueueClient()
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("generate_scene_image")
            assert handler is not None
            args = GenerateSceneImage(prompt="a torchlit crypt", kind="scene")
            with with_dispatch_context(_ctx(session.id)):
                result = await handler.fn(db_session, args)
        finally:
            await reset_queue_client()

        assert len(fake_queue.pushed) == 1
        payload = json.loads(fake_queue.pushed[0][1])
        assert payload["kind"] == "scene"
        assert payload["prompt"] == "a torchlit crypt"
        assert payload["reference_image_id"] is None
        assert payload["edit_instruction"] is None

        assert result.side_effects["mode"] == "generate"
        assert result.side_effects["kind"] == "image_queued"

    @pytest.mark.asyncio
    async def test_character_reference_dispatches_via_edit(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """A scene with reference_character_id and a canonical
        portrait should dispatch through Kontext /edit — the queued
        job carries reference_image_id pointing at the canonical
        portrait, and edit_instruction = the prompt."""

        import json

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        canon_id = await _make_canonical_image(db_session, campaign_id=campaign.id)
        char = await make_character(
            db_session, user_id=user.id, campaign_id=campaign.id, canonical_image_id=canon_id
        )
        await db_session.commit()

        fake_queue = _FakeQueueClient()
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("generate_scene_image")
            assert handler is not None
            args = GenerateSceneImage(
                prompt="same character, kneeling beside a fallen companion",
                kind="scene",
                reference_character_id=char.id,
            )
            with with_dispatch_context(_ctx(session.id)):
                result = await handler.fn(db_session, args)
        finally:
            await reset_queue_client()

        payload = json.loads(fake_queue.pushed[0][1])
        assert payload["reference_image_id"] == canon_id
        assert payload["edit_instruction"] == args.prompt
        assert result.side_effects["mode"] == "edit"
        assert result.side_effects["reference_kind"] == "character"
        assert result.side_effects["reference_id"] == char.id

    @pytest.mark.asyncio
    async def test_character_reference_without_canonical_falls_back_to_generate(  # type: ignore[no-untyped-def]
        self, db_session
    ) -> None:
        """If the referenced character exists but has no canonical
        portrait yet, fall back to /generate rather than refuse —
        a scene without identity preservation is still better than no
        scene. A separate portrait request can fix the consistency
        story afterwards."""

        import json

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(db_session, user_id=user.id, campaign_id=campaign.id)
        # Note: no canonical_image_id set on this character.
        await db_session.commit()

        fake_queue = _FakeQueueClient()
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("generate_scene_image")
            assert handler is not None
            args = GenerateSceneImage(
                prompt="brunhild fighting a goblin",
                reference_character_id=char.id,
            )
            with with_dispatch_context(_ctx(session.id)):
                result = await handler.fn(db_session, args)
        finally:
            await reset_queue_client()

        payload = json.loads(fake_queue.pushed[0][1])
        assert payload["reference_image_id"] is None
        assert payload["edit_instruction"] is None
        assert result.side_effects["mode"] == "generate"

    @pytest.mark.asyncio
    async def test_npc_reference_dispatches_via_edit(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Same path as the character reference test, but for NPCs."""

        import json

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        canon_id = await _make_canonical_image(db_session, campaign_id=campaign.id)
        npc = Npc(
            campaign_id=campaign.id,
            name="Castellan Thorvald",
            canonical_image_id=canon_id,
        )
        db_session.add(npc)
        await db_session.commit()

        fake_queue = _FakeQueueClient()
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("generate_scene_image")
            assert handler is not None
            args = GenerateSceneImage(
                prompt="same NPC, leaning against the fortress wall",
                reference_npc_id=npc.id,
            )
            with with_dispatch_context(_ctx(session.id)):
                result = await handler.fn(db_session, args)
        finally:
            await reset_queue_client()

        payload = json.loads(fake_queue.pushed[0][1])
        assert payload["reference_image_id"] == canon_id
        assert result.side_effects["reference_kind"] == "npc"

    @pytest.mark.asyncio
    async def test_unknown_reference_returns_clean_error(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """An ID the LLM invented (or one from a different campaign)
        must produce a structured error, not enqueue a job that the
        worker can't process. Cross-campaign reads are also rejected
        so a campaign can't request portraits of another's PCs."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        fake_queue = _FakeQueueClient()
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("generate_scene_image")
            assert handler is not None
            args = GenerateSceneImage(
                prompt="scene",
                reference_character_id="char-that-does-not-exist",
            )
            with with_dispatch_context(_ctx(session.id)):
                result = await handler.fn(db_session, args)
        finally:
            await reset_queue_client()

        assert result.side_effects.get("kind") == "error"
        assert result.side_effects.get("reason") == "unknown_reference"
        assert fake_queue.pushed == []

    @pytest.mark.asyncio
    async def test_both_references_set_rejected(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Setting both reference_character_id and reference_npc_id is
        ambiguous — refuse rather than guess which one the LLM meant."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        fake_queue = _FakeQueueClient()
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("generate_scene_image")
            assert handler is not None
            args = GenerateSceneImage(
                prompt="x",
                reference_character_id="a",
                reference_npc_id="b",
            )
            with with_dispatch_context(_ctx(session.id)):
                result = await handler.fn(db_session, args)
        finally:
            await reset_queue_client()

        assert result.side_effects.get("kind") == "error"
        assert result.side_effects.get("reason") == "invalid_args"
        assert fake_queue.pushed == []

    @pytest.mark.asyncio
    async def test_queue_failure_returns_error(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Queue push raises → tool returns a structured queue_unavailable
        error rather than crashing the dispatch."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        fake_queue = _FakeQueueClient(raise_on_push=True)
        set_queue_client_for_tests(fake_queue)
        try:
            handler = get_handler("generate_scene_image")
            assert handler is not None
            args = GenerateSceneImage(prompt="a battle")
            with with_dispatch_context(_ctx(session.id)):
                result = await handler.fn(db_session, args)
        finally:
            await reset_queue_client()

        assert result.side_effects.get("kind") == "error"
        assert result.side_effects.get("reason") == "queue_unavailable"


# ---------------------------------------------------------------------------
# apply_revival (Phase 6.12)
# ---------------------------------------------------------------------------


class TestApplyRevival:
    @pytest.mark.asyncio
    async def test_revives_downed_character(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """A character at hp_current=0 (survivable Death-table outcome) is
        revived to hp_current=1 and status='alive'."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(
            db_session,
            user_id=user.id,
            campaign_id=campaign.id,
            hp_current=0,
            hp_max=10,
            status="alive",
        )
        await db_session.commit()

        handler = get_handler("apply_revival")
        assert handler is not None
        args = ApplyRevival(character_id=char.id, source="Mother Serra's prayer")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.hp_current == 1
        assert refreshed.status == "alive"
        assert result.side_effects["kind"] == "state_update"
        assert result.side_effects["new"] == 1
        assert "Serra" in result.content

    @pytest.mark.asyncio
    async def test_revives_dead_character(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """A character with status='dead' (Death-table fatal outcome) is
        also revivable — divine intervention or a resurrection spell can
        override even the death outcome."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(
            db_session,
            user_id=user.id,
            campaign_id=campaign.id,
            hp_current=0,
            hp_max=8,
            status="dead",
        )
        await db_session.commit()

        handler = get_handler("apply_revival")
        assert handler is not None
        args = ApplyRevival(character_id=char.id, source="resurrection ritual")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.hp_current == 1
        assert refreshed.status == "alive"
        assert result.side_effects["previous_status"] == "dead"
        assert result.side_effects["new_status"] == "alive"

    @pytest.mark.asyncio
    async def test_clears_dying_effects_on_revival(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """apply_revival must clear dying/stable/unconscious effects automatically
        so the DM doesn't need to follow up with clear_status_effect."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(
            db_session,
            user_id=user.id,
            campaign_id=campaign.id,
            hp_current=0,
            hp_max=10,
            status="alive",
        )
        # Manually set effects as if the model called apply_status_effect earlier.
        char.status_effects = ["dying", "poisoned"]
        await db_session.commit()

        handler = get_handler("apply_revival")
        assert handler is not None
        args = ApplyRevival(character_id=char.id, source="healing potion")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.hp_current == 1
        # "dying" cleared automatically; "poisoned" should survive.
        assert "dying" not in refreshed.status_effects
        assert "poisoned" in refreshed.status_effects
        assert "dying" in result.side_effects["cleared_effects"]

    @pytest.mark.asyncio
    async def test_refuses_already_alive(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """apply_revival on a character that is alive (hp_current > 0) must
        return a structured error and leave HP unchanged."""

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

        handler = get_handler("apply_revival")
        assert handler is not None
        args = ApplyRevival(character_id=char.id)
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.hp_current == 5  # unchanged
        assert result.side_effects.get("kind") == "error"
        assert result.side_effects.get("reason") == "target_not_downed"

    @pytest.mark.asyncio
    async def test_unknown_target_returns_error(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("apply_revival")
        assert handler is not None
        args = ApplyRevival(character_id="nonexistent")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        assert result.side_effects.get("kind") == "error"
        assert result.side_effects.get("reason") == "unknown_target"


# ---------------------------------------------------------------------------
# apply_status_effect (Phase 6.12)
# ---------------------------------------------------------------------------


class TestApplyStatusEffect:
    @pytest.mark.asyncio
    async def test_adds_effect_to_list(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(db_session, user_id=user.id, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("apply_status_effect")
        assert handler is not None
        args = ApplyStatusEffect(
            character_id=char.id,
            effect="poisoned",
            duration_hint="until cured",
        )
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert "poisoned" in refreshed.status_effects
        assert result.side_effects["kind"] == "state_update"
        assert result.side_effects["effect"] == "poisoned"
        assert result.side_effects["already_present"] is False
        assert "poisoned" in result.content

    @pytest.mark.asyncio
    async def test_idempotent_if_already_present(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Applying an effect that is already present must not duplicate it
        in the list — idempotent semantics prevent accidental stacking."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(db_session, user_id=user.id, campaign_id=campaign.id)
        char.status_effects = ["poisoned"]
        await db_session.commit()

        handler = get_handler("apply_status_effect")
        assert handler is not None
        args = ApplyStatusEffect(character_id=char.id, effect="poisoned")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.status_effects.count("poisoned") == 1  # not duplicated
        assert result.side_effects["already_present"] is True

    @pytest.mark.asyncio
    async def test_module_specific_effect(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Module-specific effects like 'cursed by the shrine' must be
        accepted — the field is free-form, not an enum."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(db_session, user_id=user.id, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("apply_status_effect")
        assert handler is not None
        args = ApplyStatusEffect(character_id=char.id, effect="cursed by the shrine")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert "cursed by the shrine" in refreshed.status_effects
        assert result.side_effects["kind"] == "state_update"

    @pytest.mark.asyncio
    async def test_unknown_target_returns_error(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("apply_status_effect")
        assert handler is not None
        args = ApplyStatusEffect(character_id="nonexistent", effect="poisoned")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        assert result.side_effects.get("kind") == "error"
        assert result.side_effects.get("reason") == "unknown_target"


# ---------------------------------------------------------------------------
# clear_status_effect (Phase 6.12)
# ---------------------------------------------------------------------------


class TestClearStatusEffect:
    @pytest.mark.asyncio
    async def test_removes_effect(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(db_session, user_id=user.id, campaign_id=campaign.id)
        char.status_effects = ["poisoned", "blessed"]
        await db_session.commit()

        handler = get_handler("clear_status_effect")
        assert handler is not None
        args = ClearStatusEffect(character_id=char.id, effect="poisoned")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert "poisoned" not in refreshed.status_effects
        assert "blessed" in refreshed.status_effects  # unaffected
        assert result.side_effects["kind"] == "state_update"
        assert result.side_effects["was_present"] is True
        assert "no longer" in result.content

    @pytest.mark.asyncio
    async def test_no_op_if_not_present(self, db_session) -> None:  # type: ignore[no-untyped-def]
        """Clearing an effect that was never applied must succeed (no-op),
        not error — the DM shouldn't be penalised for a safe cleanup call."""

        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        char = await make_character(db_session, user_id=user.id, campaign_id=campaign.id)
        char.status_effects = ["blessed"]
        await db_session.commit()

        handler = get_handler("clear_status_effect")
        assert handler is not None
        args = ClearStatusEffect(character_id=char.id, effect="poisoned")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Character, char.id)
        assert refreshed is not None
        assert refreshed.status_effects == ["blessed"]  # unchanged
        assert result.side_effects["kind"] == "state_update"
        assert result.side_effects["was_present"] is False
        # Must not be an error — just a note.
        assert result.side_effects.get("reason") != "unknown_target"

    @pytest.mark.asyncio
    async def test_unknown_target_returns_error(self, db_session) -> None:  # type: ignore[no-untyped-def]
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("clear_status_effect")
        assert handler is not None
        args = ClearStatusEffect(character_id="nonexistent", effect="poisoned")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        assert result.side_effects.get("kind") == "error"
        assert result.side_effects.get("reason") == "unknown_target"


# ---------------------------------------------------------------------------
# Helpers for module-backed campaign tests
# ---------------------------------------------------------------------------

_MINIMAL_MODULE_CONTENT = {
    "format_version": "1.0",
    "synopsis": "Test synopsis",
    "tone": "gritty",
    "starting_hook": "A stranger arrives.",
    "starting_location_symbol": "loc_keep",
    "locations": [{"symbol": "loc_keep", "name": "The Keep", "description": "Stone walls."}],
    "npcs": [],
    "encounters": [],
    "plot_beats": [
        {
            "symbol": "beat_arrival",
            "title": "Arrival Briefing",
            "trigger_hint": "When the party first meets the Castellan.",
            "outcome": "The party is hired.",
        },
        {
            "symbol": "beat_confrontation",
            "title": "Final Confrontation",
            "trigger_hint": "When the party faces the enemy.",
            "outcome": "The enemy is defeated.",
        },
    ],
    "secrets": [
        {
            "symbol": "sec_dark_past",
            "content": "The Castellan has a dark past.",
            "reveal_when": "When the party finds the old journal.",
        }
    ],
    "endings": [],
    "world_facts": [],
}


async def _make_module(db, *, author_id: str) -> Module:
    mod = Module(
        author_id=author_id,
        name="Test Module",
        content=_MINIMAL_MODULE_CONTENT,
    )
    db.add(mod)
    await db.flush()
    return mod


def _module_state(module_id: str) -> dict:
    return {
        "module_id": module_id,
        "symbolic_id_map": {
            "loc_keep": "uuid-loc-keep",
            "beat_arrival": "uuid-beat-arrival",
            "beat_confrontation": "uuid-beat-confrontation",
            "sec_dark_past": "uuid-sec-dark-past",
        },
        "beats_hit": [],
        "beats_pending": ["beat_arrival", "beat_confrontation"],
        "secrets_revealed": [],
        "encounters_run": [],
        "endings_reached": [],
    }


# ---------------------------------------------------------------------------
# mark_beat
# ---------------------------------------------------------------------------


class TestMarkBeat:
    @pytest.mark.asyncio
    async def test_moves_beat_pending_to_hit(self, db_session) -> None:
        user = await make_user(db_session)
        mod = await _make_module(db_session, author_id=user.id)
        campaign = await make_campaign(
            db_session,
            owner_id=user.id,
            module_id=mod.id,
            module_state=_module_state(mod.id),
        )
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("mark_beat")
        assert handler is not None
        args = MarkBeat(beat_id="beat_arrival", summary="Party hired by Castellan.")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Campaign, campaign.id)
        assert refreshed is not None
        assert "beat_arrival" in refreshed.module_state["beats_hit"]
        assert "beat_arrival" not in refreshed.module_state["beats_pending"]
        assert result.side_effects["kind"] == "beat_marked"
        assert result.side_effects["beat_id"] == "beat_arrival"

    @pytest.mark.asyncio
    async def test_already_hit_is_noop(self, db_session) -> None:
        user = await make_user(db_session)
        mod = await _make_module(db_session, author_id=user.id)
        state = _module_state(mod.id)
        state["beats_hit"] = ["beat_arrival"]
        state["beats_pending"] = ["beat_confrontation"]
        campaign = await make_campaign(
            db_session,
            owner_id=user.id,
            module_id=mod.id,
            module_state=state,
        )
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("mark_beat")
        assert handler is not None
        args = MarkBeat(beat_id="beat_arrival")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)

        assert result.side_effects["kind"] == "beat_already_hit"

    @pytest.mark.asyncio
    async def test_unknown_beat_returns_error(self, db_session) -> None:
        user = await make_user(db_session)
        mod = await _make_module(db_session, author_id=user.id)
        campaign = await make_campaign(
            db_session,
            owner_id=user.id,
            module_id=mod.id,
            module_state=_module_state(mod.id),
        )
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("mark_beat")
        assert handler is not None
        args = MarkBeat(beat_id="beat_nonexistent")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)

        assert result.side_effects["kind"] == "error"
        assert result.side_effects["reason"] == "unknown_beat"

    @pytest.mark.asyncio
    async def test_no_module_returns_error(self, db_session) -> None:
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("mark_beat")
        assert handler is not None
        args = MarkBeat(beat_id="beat_arrival")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)

        assert result.side_effects["kind"] == "error"
        assert result.side_effects["reason"] == "no_module"


# ---------------------------------------------------------------------------
# reveal_secret
# ---------------------------------------------------------------------------


class TestRevealSecret:
    @pytest.mark.asyncio
    async def test_moves_secret_to_revealed(self, db_session) -> None:
        user = await make_user(db_session)
        mod = await _make_module(db_session, author_id=user.id)
        campaign = await make_campaign(
            db_session,
            owner_id=user.id,
            module_id=mod.id,
            module_state=_module_state(mod.id),
        )
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("reveal_secret")
        assert handler is not None
        args = RevealSecret(secret_id="sec_dark_past")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)
        await db_session.commit()

        refreshed = await db_session.get(Campaign, campaign.id)
        assert refreshed is not None
        assert "sec_dark_past" in refreshed.module_state["secrets_revealed"]
        assert result.side_effects["kind"] == "secret_revealed"
        assert result.side_effects["secret_id"] == "sec_dark_past"

    @pytest.mark.asyncio
    async def test_already_revealed_is_noop(self, db_session) -> None:
        user = await make_user(db_session)
        mod = await _make_module(db_session, author_id=user.id)
        state = _module_state(mod.id)
        state["secrets_revealed"] = ["sec_dark_past"]
        campaign = await make_campaign(
            db_session,
            owner_id=user.id,
            module_id=mod.id,
            module_state=state,
        )
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("reveal_secret")
        assert handler is not None
        args = RevealSecret(secret_id="sec_dark_past")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)

        assert result.side_effects["kind"] == "secret_already_revealed"

    @pytest.mark.asyncio
    async def test_unknown_secret_returns_error(self, db_session) -> None:
        user = await make_user(db_session)
        mod = await _make_module(db_session, author_id=user.id)
        campaign = await make_campaign(
            db_session,
            owner_id=user.id,
            module_id=mod.id,
            module_state=_module_state(mod.id),
        )
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("reveal_secret")
        assert handler is not None
        args = RevealSecret(secret_id="sec_nonexistent")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)

        assert result.side_effects["kind"] == "error"
        assert result.side_effects["reason"] == "unknown_secret"

    @pytest.mark.asyncio
    async def test_no_module_returns_error(self, db_session) -> None:
        user = await make_user(db_session)
        campaign = await make_campaign(db_session, owner_id=user.id)
        session = await make_session(db_session, campaign_id=campaign.id)
        await db_session.commit()

        handler = get_handler("reveal_secret")
        assert handler is not None
        args = RevealSecret(secret_id="sec_dark_past")
        with with_dispatch_context(_ctx(session.id)):
            result = await handler.fn(db_session, args)

        assert result.side_effects["kind"] == "error"
        assert result.side_effects["reason"] == "no_module"
