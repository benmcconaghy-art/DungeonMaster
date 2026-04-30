"""Tests for ``app.orchestrator.dm.take_turn``.

The vLLM client is mocked at the boundary: ``get_dm_client()`` returns
a :class:`_FakeDmClient` whose ``stream_dm`` yields canned chunks.
The rest of the path — prompt build, handler dispatch, persistence
— is exercised against the in-memory SQLite fixture.

The transaction-discipline invariant (AGENTS.md #2) is exercised
implicitly: if the orchestrator held a write transaction across the
stream, the in-stream handler dispatch — which itself opens
``async with db.begin()`` — would deadlock. Tests therefore *assert
the absence of deadlock* by passing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.db.models import DiceRoll, SessionMessage
from app.llm.client import RunawayTokenError
from app.orchestrator.dm import (
    DiceRollEvent,
    DmError,
    NarrationChunk,
    NarrationComplete,
    StateUpdate,
    ToolDispatched,
    take_turn,
)
from tests.orchestrator.factories import (
    make_campaign,
    make_character,
    make_session,
    make_user,
)

# ---------------------------------------------------------------------------
# Fake chunk constructors — quack like ``ChatCompletionChunk``
# ---------------------------------------------------------------------------


def _content_chunk(text: str) -> Any:
    """A chunk whose ``delta.content`` carries ``text`` and no tool calls."""

    delta = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def _tool_call_chunk(
    *,
    index: int,
    id: str | None,
    name: str | None,
    arguments: str | None,
) -> Any:
    """A chunk that carries a tool-call fragment.

    The OpenAI streaming protocol fragments a tool call across chunks:
    one ``id`` + ``name`` opener, one or more ``arguments`` continuations.
    Tests that need that pattern build multiple of these in sequence.
    """

    function = SimpleNamespace(name=name, arguments=arguments)
    fragment = SimpleNamespace(index=index, id=id, type="function", function=function)
    delta = SimpleNamespace(content=None, tool_calls=[fragment])
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# Fake DM client
# ---------------------------------------------------------------------------


class _FakeDmClient:
    """Stand-in for :class:`DmClient` that returns canned chunk streams.

    Construct with a list of streams — one per expected ``stream_dm``
    invocation. The orchestrator's tool-call loop calls the client
    multiple times in one turn, so tests that exercise tool-call
    iteration provide multiple streams.

    The sentinel ``RAISE_RUNAWAY`` makes a stream raise
    :class:`RunawayTokenError` on first iteration; used by the runaway
    test.
    """

    RAISE_RUNAWAY = object()

    def __init__(self, streams: list[list[Any] | object]) -> None:
        self._streams = streams
        self.call_count = 0

    async def stream_dm(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        if self.call_count >= len(self._streams):
            raise AssertionError(
                f"_FakeDmClient: stream_dm called {self.call_count + 1} times"
                f" but only {len(self._streams)} streams were prepared"
            )
        chunks = self._streams[self.call_count]
        self.call_count += 1

        async def _gen(
            chunks_or_sentinel: list[Any] | object,
        ) -> AsyncIterator[Any]:
            if chunks_or_sentinel is _FakeDmClient.RAISE_RUNAWAY:
                raise RunawayTokenError("simulated runaway")
            assert isinstance(chunks_or_sentinel, list)
            for chunk in chunks_or_sentinel:
                yield chunk

        return _gen(chunks)


@pytest.fixture
def patch_client(monkeypatch):  # type: ignore[no-untyped-def]
    """Replace ``app.orchestrator.dm.get_dm_client`` with a builder.

    Tests call ``patch_client(streams=...)`` and the orchestrator's
    ``get_dm_client()`` returns the configured fake.
    """

    def _install(streams: list[list[Any] | object]) -> _FakeDmClient:
        fake = _FakeDmClient(streams)
        monkeypatch.setattr("app.orchestrator.dm.get_dm_client", lambda: fake)
        return fake

    return _install


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


async def _setup_session(
    db_session: AsyncSession,
) -> tuple[models.User, models.Campaign, models.Session, models.Character]:
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
    return user, campaign, session, char


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plain_narration_no_tool_calls(db_session, patch_client) -> None:  # type: ignore[no-untyped-def]
    """The DM produces narration only — no tool calls."""

    _, _, session, _ = await _setup_session(db_session)
    patch_client([[_content_chunk("You "), _content_chunk("see "), _content_chunk("a door.")]])

    events = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="I look around.",
    ):
        events.append(event)

    chunks = [e for e in events if isinstance(e, NarrationChunk)]
    completes = [e for e in events if isinstance(e, NarrationComplete)]
    assert [c.content for c in chunks] == ["You ", "see ", "a door."]
    assert len(completes) == 1
    assert completes[0].content == "You see a door."

    # Player message + DM message persisted.
    msgs = list(
        (await db_session.scalars(select(SessionMessage).order_by(SessionMessage.created_at))).all()
    )
    kinds = [m.sender_kind for m in msgs]
    assert "player" in kinds
    assert "dm" in kinds
    dm_msg = next(m for m in msgs if m.sender_kind == "dm")
    assert dm_msg.content == "You see a door."


@pytest.mark.asyncio
async def test_one_tool_call_request_dice_roll(  # type: ignore[no-untyped-def]
    db_session, patch_client
) -> None:
    """First stream emits a tool call, second stream emits the narration."""

    _, _, session, _ = await _setup_session(db_session)

    arguments = json.dumps(
        {
            "expression": "1d20+2",
            "purpose": "spot the trap",
            "actor": "dm",
            "target": {"kind": "dc", "value": 12},
        }
    )

    streams: list[list[Any] | object] = [
        [
            _tool_call_chunk(index=0, id="call-1", name="request_dice_roll", arguments=None),
            _tool_call_chunk(index=0, id=None, name=None, arguments=arguments),
        ],
        [
            _content_chunk("You spot the tripwire."),
        ],
    ]
    patch_client(streams)

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="I check the room for traps.",
    ):
        events.append(event)

    # A dice_roll row was persisted between streams.
    dice_rows = list((await db_session.scalars(select(DiceRoll))).all())
    assert len(dice_rows) == 1
    assert dice_rows[0].expression == "1d20+2"
    assert dice_rows[0].session_id == session.id

    # Tool dispatched event + the dice_roll convenience event.
    assert any(isinstance(e, ToolDispatched) for e in events)
    assert any(isinstance(e, DiceRollEvent) for e in events)
    assert any(isinstance(e, NarrationComplete) and "tripwire" in e.content for e in events)


@pytest.mark.asyncio
async def test_apply_damage_in_stream(db_session, patch_client) -> None:  # type: ignore[no-untyped-def]
    """Tool call writes through the handler; HP drops in DB between streams."""

    _, _, session, char = await _setup_session(db_session)

    streams: list[list[Any] | object] = [
        [
            _tool_call_chunk(
                index=0,
                id="call-1",
                name="apply_damage",
                arguments=json.dumps({"target_id": char.id, "amount": 4, "source": "goblin"}),
            ),
        ],
        [_content_chunk("Goblin's blade bites deep.")],
    ]
    patch_client(streams)

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="Defend!",
    ):
        events.append(event)

    # State-update event present.
    assert any(isinstance(e, StateUpdate) for e in events)
    # HP went 10 -> 6.
    from app.db.models import Character

    refreshed = await db_session.get(Character, char.id)
    assert refreshed is not None
    assert refreshed.hp_current == 6


@pytest.mark.asyncio
async def test_json_block_fallback_parser(db_session, patch_client) -> None:  # type: ignore[no-untyped-def]
    """No native tool_calls + a fenced ``json`` block in content -> dispatched."""

    _, _, session, _ = await _setup_session(db_session)

    fallback_block = (
        "I'd like to roll a d20 here.\n"
        "```json\n"
        + json.dumps(
            {
                "name": "request_dice_roll",
                "arguments": {
                    "expression": "1d20",
                    "purpose": "perception",
                    "actor": "dm",
                },
            }
        )
        + "\n```"
    )

    streams: list[list[Any] | object] = [
        [_content_chunk(fallback_block)],
        [_content_chunk("You hear footsteps.")],
    ]
    patch_client(streams)

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="Listen at the door.",
    ):
        events.append(event)

    # Dispatched the fallback tool call -> dice row persisted.
    dice_rows = list((await db_session.scalars(select(DiceRoll))).all())
    assert len(dice_rows) == 1
    assert dice_rows[0].purpose == "perception"
    assert any(isinstance(e, ToolDispatched) for e in events)


@pytest.mark.asyncio
async def test_iteration_cap_breaks_runaway(db_session, patch_client) -> None:  # type: ignore[no-untyped-def]
    """Ten tool calls in a row -> orchestrator gives up at 5 with dm_error."""

    _, _, session, _ = await _setup_session(db_session)

    def _roll_stream(idx: int) -> list[Any]:
        return [
            _tool_call_chunk(
                index=0,
                id=f"call-{idx}",
                name="request_dice_roll",
                arguments=json.dumps(
                    {
                        "expression": "1d20",
                        "purpose": f"loop-{idx}",
                        "actor": "dm",
                    }
                ),
            ),
        ]

    # 10 streams of tool calls. The orchestrator should hit the iteration
    # cap (5) before running out of streams.
    streams: list[list[Any] | object] = [_roll_stream(i) for i in range(10)]
    patch_client(streams)

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="Loop forever.",
    ):
        events.append(event)

    errors = [e for e in events if isinstance(e, DmError)]
    assert errors, "expected at least one DmError event"
    assert any(e.reason == "iteration_cap" for e in errors)


@pytest.mark.asyncio
async def test_runaway_token_error_surfaces(db_session, patch_client) -> None:  # type: ignore[no-untyped-def]
    """A RunawayTokenError raised mid-stream -> DmError event, no crash."""

    _, _, session, _ = await _setup_session(db_session)
    patch_client([_FakeDmClient.RAISE_RUNAWAY])

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="anything",
    ):
        events.append(event)

    errors = [e for e in events if isinstance(e, DmError)]
    assert any(e.reason == "runaway_token" for e in errors)
    # No NarrationComplete was emitted.
    assert not any(isinstance(e, NarrationComplete) for e in events)


@pytest.mark.asyncio
async def test_empty_completion_yields_dm_error(db_session, patch_client) -> None:  # type: ignore[no-untyped-def]
    """An assistant message with neither content nor tool_calls -> dm_error."""

    _, _, session, _ = await _setup_session(db_session)
    patch_client([[_content_chunk("")]])

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="hi",
    ):
        events.append(event)

    errors = [e for e in events if isinstance(e, DmError)]
    assert any(e.reason == "empty_completion" for e in errors)


@pytest.mark.asyncio
async def test_player_input_persisted_independently_of_completion(  # type: ignore[no-untyped-def]
    db_session, patch_client
) -> None:
    """Input persistence happens before streaming, in its own committed
    transaction. We prove this by triggering an empty-completion error
    (no further DB writes) and then asserting the player message is
    visible — if input persistence depended on the completion writing
    cleanly, the player row would have been rolled back."""

    _, _, session, _ = await _setup_session(db_session)
    patch_client([[_content_chunk("")]])  # empty completion -> dm_error

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="player-input-must-be-persisted",
    ):
        events.append(event)

    # Empty completion → DmError, but the player message survived.
    assert any(isinstance(e, DmError) for e in events)

    # Need to clear any autobegun transaction state before we re-read.
    await db_session.commit()
    rows = list(
        (
            await db_session.scalars(
                select(SessionMessage).where(SessionMessage.session_id == session.id)
            )
        ).all()
    )
    assert any(
        r.sender_kind == "player" and "player-input-must-be-persisted" in r.content for r in rows
    )
