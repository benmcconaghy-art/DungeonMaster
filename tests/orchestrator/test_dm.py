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
        # One snapshot of ``messages`` per ``stream_dm`` invocation. Tests
        # inspect this to verify the orchestrator's prompt-history hygiene
        # (e.g. malformed tool-call ``arguments`` strings must never
        # appear in a subsequent call's messages list — Phase 6.9 fix).
        self.received_messages_log: list[list[dict[str, Any]]] = []

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
        # Snapshot a deep-ish copy: the orchestrator may mutate the list
        # between calls (appends), and we want the per-call view.
        self.received_messages_log.append([dict(m) for m in messages])
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
    """Ten tool calls in a row -> orchestrator gives up at the cap with
    dm_error. The cap is :data:`_MAX_TOOL_ITERATIONS` (currently 10;
    Phase 4 prep #1 re-evaluated and held it). The test feeds 10
    streams of tool calls and just asserts the cap-error fires, so the
    exact number can drift without breaking this test."""

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
async def test_opening_turn_persists_system_not_player_message(  # type: ignore[no-untyped-def]
    db_session, patch_client
) -> None:
    """Phase 6.8 Bug 3: ``take_turn(opening=True)`` writes the leading
    message as ``sender_kind='system'`` so the prompt builder surfaces
    it as engine context, not as a player utterance the DM is
    expected to respond to. The DM message lands as usual."""

    _, _, session, _ = await _setup_session(db_session)
    patch_client([[_content_chunk("The wind howls. You stand at the gate.")]])

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="[opening directive]",
        opening=True,
    ):
        events.append(event)

    msgs = list(
        (
            await db_session.scalars(
                select(SessionMessage).where(SessionMessage.session_id == session.id)
            )
        ).all()
    )
    kinds = [m.sender_kind for m in msgs]
    # No 'player' row was persisted for the synthetic directive.
    assert "player" not in kinds
    # The directive landed as 'system' (engine context).
    assert "system" in kinds
    leading = next(m for m in msgs if m.sender_kind == "system")
    assert leading.content == "[opening directive]"
    # The DM still spoke and got persisted as 'dm'.
    assert "dm" in kinds


@pytest.mark.asyncio
async def test_stream_id_is_per_iteration(  # type: ignore[no-untyped-def]
    db_session, patch_client
) -> None:
    """Phase 6.8 Bug 1: each orchestrator iteration mints a fresh
    stream_id so the client can render per-iteration bubbles. A turn
    that emits a chunk → tool call → more chunks must carry two
    distinct stream_ids on the chunk frames, and the trailing
    NarrationComplete pairs with the *final* iteration's id."""

    _, _, session, _ = await _setup_session(db_session)

    streams: list[list[Any] | object] = [
        [
            _content_chunk("Before tool: "),
            _tool_call_chunk(
                index=0,
                id="call-1",
                name="request_dice_roll",
                arguments=json.dumps(
                    {"expression": "1d20", "purpose": "perception", "actor": "dm"}
                ),
            ),
        ],
        [_content_chunk("After tool, scene continues.")],
    ]
    patch_client(streams)

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="Look around.",
    ):
        events.append(event)

    chunk_events = [e for e in events if isinstance(e, NarrationChunk)]
    # First iteration produced one chunk before the tool call;
    # second iteration produced one chunk after the tool result.
    assert len(chunk_events) == 2
    sids = [c.stream_id for c in chunk_events]
    assert sids[0] != sids[1], (
        "post-tool chunks must carry a different stream_id from pre-tool chunks "
        "so the client renders them as a discrete bubble"
    )
    # Per-iteration also means each id is a non-empty string.
    assert all(s for s in sids)

    # NarrationComplete pairs with the final iteration.
    completes = [e for e in events if isinstance(e, NarrationComplete)]
    assert len(completes) == 1
    assert completes[0].stream_id == sids[1]


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


# ---------------------------------------------------------------------------
# Phase 6.9: tool-error history hygiene
#
# When Nemotron emits a tool call whose ``arguments`` aren't valid JSON
# (or whose tool isn't registered, etc.), the orchestrator must not let
# that malformed call propagate to the next vLLM request. The
# OpenAI-shaped assistant message would embed the raw bad ``arguments``
# string in ``tool_calls[].function.arguments``; vLLM's chat-template
# rendering trips an HTTP 400 ("Expecting property name…") on the next
# call and wedges the session.
#
# Real-traffic evidence: 2026-05-03 playthrough, captured in
# ``deploy/PLAYTHROUGH_2026-05-03.md``.
# ---------------------------------------------------------------------------


from app.orchestrator.dm import (  # noqa: E402  module-late-imports for test grouping
    _TOOL_REJECTION_RECOVERY_NOTE,
    _AccumulatedToolCall,
    _classify_tool_call,
    _safe_arguments_string,
)


def _malformed_args_str() -> str:
    """A Python-dict-literal string. ``json.loads`` rejects it with
    "Expecting property name enclosed in double quotes", which is the
    exact shape observed in the 2026-05-03 playthrough."""

    return "{name: 'request_dice_roll', expression: '1d20'}"


@pytest.mark.asyncio
async def test_malformed_tool_args_does_not_poison_next_prompt(  # type: ignore[no-untyped-def]
    db_session, patch_client
) -> None:
    """A malformed-args tool call must NOT appear in the messages list
    that the orchestrator sends on the next iteration. The classification
    gate in :func:`_classify_tool_call` drops it from history and
    appends only a sanitised system note.
    """

    _, _, session, _ = await _setup_session(db_session)

    bad_args = _malformed_args_str()

    streams: list[list[Any] | object] = [
        # Iteration 1: model emits a tool call with malformed arguments.
        [
            _tool_call_chunk(
                index=0,
                id="bad-call-1",
                name="request_dice_roll",
                arguments=bad_args,
            ),
        ],
        # Iteration 2: model recovers with plain narration.
        [_content_chunk("You spot nothing of note.")],
    ]
    fake = patch_client(streams)

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="Look around.",
    ):
        events.append(event)

    # The orchestrator emitted the rejection event and then completed
    # cleanly with narration on the next pass.
    errors = [e for e in events if isinstance(e, DmError)]
    assert any(
        e.reason == "invalid_tool_args" for e in errors
    ), "expected an invalid_tool_args DmError for the malformed call"
    assert any(
        isinstance(e, NarrationComplete) for e in events
    ), "expected NarrationComplete after the recovery iteration"

    # Both stream_dm calls happened.
    assert fake.call_count == 2
    second_call_messages = fake.received_messages_log[1]

    # The malformed args string must not appear anywhere in the next
    # request's messages list — not in any field, not as a substring.
    serialised = json.dumps(second_call_messages)
    assert bad_args not in serialised, (
        "malformed arguments string leaked into the next prompt; vLLM "
        "would reject this with HTTP 400"
    )

    # Every assistant message in the next request must have either no
    # ``tool_calls`` or only entries whose ``arguments`` are valid JSON
    # object strings.
    for msg in second_call_messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            args_str = tc["function"]["arguments"]
            parsed = json.loads(args_str)  # must not raise
            assert isinstance(parsed, dict)

    # The recovery system note is present so the model has a clear
    # signal that a call didn't land.
    assert any(
        msg.get("role") == "system" and msg.get("content") == _TOOL_REJECTION_RECOVERY_NOTE
        for msg in second_call_messages
    ), "expected the sanitised recovery system note in the next prompt"


@pytest.mark.asyncio
async def test_unknown_tool_does_not_poison_next_prompt(  # type: ignore[no-untyped-def]
    db_session, patch_client
) -> None:
    """The class of "tool call we cannot honour" includes unknown tool
    names too. Same guarantees as the malformed-args case: rejected
    from history, recovery note appended.
    """

    _, _, session, _ = await _setup_session(db_session)

    streams: list[list[Any] | object] = [
        [
            _tool_call_chunk(
                index=0,
                id="unknown-1",
                name="this_tool_does_not_exist",
                arguments=json.dumps({"foo": "bar"}),
            ),
        ],
        [_content_chunk("The room is silent.")],
    ]
    fake = patch_client(streams)

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="hello",
    ):
        events.append(event)

    errors = [e for e in events if isinstance(e, DmError)]
    assert any(e.reason == "unknown_tool" for e in errors)

    assert fake.call_count == 2
    second_call_messages = fake.received_messages_log[1]

    # No assistant message in the next prompt references the unknown
    # tool by name in a tool_calls slot.
    for msg in second_call_messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            assert tc["function"]["name"] != "this_tool_does_not_exist"

    # Recovery note present.
    assert any(
        msg.get("role") == "system" and msg.get("content") == _TOOL_REJECTION_RECOVERY_NOTE
        for msg in second_call_messages
    )


@pytest.mark.asyncio
async def test_mixed_honourable_and_rejected_calls(  # type: ignore[no-untyped-def]
    db_session, patch_client
) -> None:
    """When the same iteration emits one honourable call and one
    rejected call, the assistant audit must include only the honourable
    one. The rejected call's tool message is dropped; the honourable
    call's tool message is preserved.
    """

    _, _, session, _ = await _setup_session(db_session)

    good_args = json.dumps({"expression": "1d20", "purpose": "perception", "actor": "dm"})
    bad_args = _malformed_args_str()

    streams: list[list[Any] | object] = [
        [
            _tool_call_chunk(index=0, id="good-1", name="request_dice_roll", arguments=good_args),
            _tool_call_chunk(index=1, id="bad-1", name="request_dice_roll", arguments=bad_args),
        ],
        [_content_chunk("You scan the room.")],
    ]
    fake = patch_client(streams)

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="Look.",
    ):
        events.append(event)

    # The good call dispatched (dice row was persisted).
    dice_rows = list((await db_session.scalars(select(DiceRoll))).all())
    assert len(dice_rows) == 1

    # The bad call surfaced as a rejection event.
    errors = [e for e in events if isinstance(e, DmError)]
    assert any(e.reason == "invalid_tool_args" for e in errors)

    # The next prompt contains the assistant audit with the GOOD call,
    # not the bad one.
    second_call_messages = fake.received_messages_log[1]
    assistant_msgs = [m for m in second_call_messages if m.get("role") == "assistant"]
    audit_msgs = [m for m in assistant_msgs if m.get("tool_calls")]
    assert audit_msgs, "expected an assistant audit message with tool_calls"

    # Exactly one tool_calls entry, and it's the good one.
    audit = audit_msgs[-1]
    tcs = audit["tool_calls"]
    assert len(tcs) == 1
    assert tcs[0]["id"] == "good-1"
    json.loads(tcs[0]["function"]["arguments"])  # valid JSON

    # The malformed args string never leaks into the next prompt.
    serialised = json.dumps(second_call_messages)
    assert bad_args not in serialised


@pytest.mark.asyncio
async def test_all_rejected_with_narration_preserves_prose(  # type: ignore[no-untyped-def]
    db_session, patch_client
) -> None:
    """If every tool call this iteration is rejected but the model also
    narrated, the narration prose should be preserved in the next
    prompt as an assistant message with no ``tool_calls`` field. The
    model's intent isn't lost on retry.
    """

    _, _, session, _ = await _setup_session(db_session)

    bad_args = _malformed_args_str()

    streams: list[list[Any] | object] = [
        [
            _content_chunk("You raise your lantern. "),
            _tool_call_chunk(index=0, id="bad-1", name="request_dice_roll", arguments=bad_args),
        ],
        [_content_chunk("The shadows recoil.")],
    ]
    fake = patch_client(streams)

    events: list[Any] = []
    async for event in take_turn(
        db_session,
        session_id=session.id,
        sender_user_id="test-user",
        sender_character_id=None,
        content="Light it.",
    ):
        events.append(event)

    second_call_messages = fake.received_messages_log[1]

    # An assistant message with the original prose is present, but no
    # tool_calls field on it.
    matching = [
        m
        for m in second_call_messages
        if m.get("role") == "assistant" and "raise your lantern" in (m.get("content") or "")
    ]
    assert matching, "expected the narration prose to survive"
    for m in matching:
        assert not m.get(
            "tool_calls"
        ), "narration-only assistant message must not carry a tool_calls field"

    # And the recovery note is there.
    assert any(
        m.get("role") == "system" and m.get("content") == _TOOL_REJECTION_RECOVERY_NOTE
        for m in second_call_messages
    )


def test_classify_tool_call_accepts_valid_call() -> None:
    """A complete, JSON-valid, schema-valid, handler-registered call
    classifies as honourable (returns ``None``)."""

    tc = _AccumulatedToolCall(index=0)
    tc.id = "ok-1"
    tc.type = "function"
    tc.name = "request_dice_roll"
    tc.arguments = json.dumps({"expression": "1d20", "purpose": "perception", "actor": "dm"})
    assert _classify_tool_call(tc) is None


def test_classify_tool_call_rejects_malformed_json() -> None:
    """Python-dict-literal arguments classify as ``invalid_tool_args``."""

    tc = _AccumulatedToolCall(index=0)
    tc.id = "bad-1"
    tc.type = "function"
    tc.name = "request_dice_roll"
    tc.arguments = _malformed_args_str()
    rejection = _classify_tool_call(tc)
    assert rejection is not None
    reason, message = rejection
    assert reason == "invalid_tool_args"
    assert "invalid arguments JSON" in message


def test_classify_tool_call_rejects_unknown_tool() -> None:
    tc = _AccumulatedToolCall(index=0)
    tc.id = "u-1"
    tc.type = "function"
    tc.name = "no_such_tool_exists"
    tc.arguments = json.dumps({"x": 1})
    rejection = _classify_tool_call(tc)
    assert rejection is not None
    assert rejection[0] == "unknown_tool"


def test_classify_tool_call_rejects_incomplete() -> None:
    """Empty name / arguments fragment classifies as
    ``incomplete_tool_call``."""

    tc = _AccumulatedToolCall(index=0)
    tc.id = "inc-1"
    # Leave name and arguments empty — never finished assembling.
    rejection = _classify_tool_call(tc)
    assert rejection is not None
    assert rejection[0] == "incomplete_tool_call"


def test_safe_arguments_string_passes_valid_json() -> None:
    valid = json.dumps({"a": 1, "b": "two"})
    assert _safe_arguments_string(valid) == valid


def test_safe_arguments_string_substitutes_for_invalid() -> None:
    """A malformed string is replaced with ``"{}"`` so the assistant
    audit message stays vLLM-valid."""

    assert _safe_arguments_string(_malformed_args_str()) == "{}"
    assert _safe_arguments_string("") == "{}"
    # JSON-valid but not an object (e.g. a list or a number).
    assert _safe_arguments_string("[1, 2, 3]") == "{}"
    assert _safe_arguments_string("42") == "{}"


@pytest.mark.asyncio
async def test_recovery_message_text_is_constant() -> None:
    """Pin the exact recovery-note text. Future edits to the wording
    should be deliberate (the note ships into the model's prompt and
    its tone shapes the recovery iteration's narration)."""

    assert _TOOL_REJECTION_RECOVERY_NOTE == (
        "[engine: an attempted tool call was rejected as malformed and discarded;"
        " describe what you intended in narration or try again]"
    )
