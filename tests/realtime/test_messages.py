"""Tests for the WS wire-message contracts.

The hub depends on these for round-tripping every server-emitted event
and every client-submitted intent through Valkey. The discriminated
unions are the most error-prone part — a missing variant or a renamed
``type`` value silently drops messages on the wire — so each variant
gets a serialise-and-parse round trip.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from app.realtime.messages import (
    ClientMessage,
    ClientOutOfBandChat,
    ClientPcAction,
    ClientPing,
    ClientWhisperToDm,
    CurrentActor,
    DiceRoll,
    DmError,
    ImagePending,
    ImageReady,
    NarrationChunk,
    NarrationComplete,
    PcAction,
    Pong,
    Presence,
    PresenceEntry,
    ServerMessage,
    Snapshot,
    SnapshotMessage,
    StateUpdate,
    Whisper,
)

_SERVER_ADAPTER: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)
_CLIENT_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def _server_round_trip(msg: ServerMessage) -> ServerMessage:
    """Helper: serialise via the union TypeAdapter and parse back."""

    raw = _SERVER_ADAPTER.dump_json(msg)
    return _SERVER_ADAPTER.validate_json(raw)


def _client_round_trip(msg: ClientMessage) -> ClientMessage:
    raw = _CLIENT_ADAPTER.dump_json(msg)
    return _CLIENT_ADAPTER.validate_json(raw)


# ---------------------------------------------------------------------------
# ServerMessage variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        NarrationChunk(content="The goblin hisses."),
        NarrationComplete(message_id="msg-1", content="The goblin falls."),
        PcAction(character_id="ch-1", user_id="u-1", content="I attack."),
        Whisper(tool_call_id="tc-1", audience=["ch-2"], content="A note slips into your hand."),
        DiceRoll(
            tool_call_id="tc-2",
            expression="1d20+3",
            total=18,
            individual=[15],
            purpose="to-hit",
            target={"kind": "ac", "value": 14},
        ),
        StateUpdate(tool_call_id="tc-3", side_effects={"kind": "hp", "delta": -3}),
        ImagePending(image_id="img-1", placeholder="loading…"),
        ImageReady(image_id="img-1", url="/static/img/img-1.png"),
        Presence(
            connected=[
                PresenceEntry(
                    user_id="u-1", username="alice", character_id="ch-1", character_name="Tav"
                )
            ]
        ),
        DmError(reason="empty_completion", message="nothing came back"),
        Pong(nonce="abc"),
        Snapshot(
            session_id="s-1",
            current_location_id=None,
            current_actor=None,
            messages=[],
            connected=[],
        ),
    ],
)
def test_server_message_round_trip(msg: ServerMessage) -> None:
    """Every server-side variant survives a JSON round trip with type
    discriminator intact."""

    parsed = _server_round_trip(msg)
    assert type(parsed) is type(msg)
    assert parsed == msg


def test_server_snapshot_round_trip_with_actor_and_messages() -> None:
    """Snapshot with a populated current_actor and a history."""

    snapshot = Snapshot(
        session_id="s-1",
        current_location_id="loc-keep",
        current_actor=CurrentActor(
            encounter_id="enc-1",
            participant_id="goblin#1",
            name="goblin",
            is_player=False,
            round_number=1,
        ),
        messages=[
            SnapshotMessage(
                id="m-1",
                sender_kind="dm",
                sender_id=None,
                audience=[],
                content="The keep looms.",
                created_at="2026-04-30T12:00:00.000Z",
            ),
            SnapshotMessage(
                id="m-2",
                sender_kind="player",
                sender_id="ch-1",
                audience=[],
                content="I look around.",
                created_at="2026-04-30T12:00:01.000Z",
            ),
        ],
        connected=[
            PresenceEntry(
                user_id="u-1", username="alice", character_id="ch-1", character_name="Tav"
            ),
        ],
    )
    parsed = _server_round_trip(snapshot)
    assert isinstance(parsed, Snapshot)
    assert parsed.current_actor is not None
    assert parsed.current_actor.participant_id == "goblin#1"
    assert len(parsed.messages) == 2
    assert parsed.messages[1].sender_id == "ch-1"


def test_server_message_rejects_unknown_type() -> None:
    """Bad ``type`` discriminator surfaces a validation error rather than
    silently parsing into the first variant."""

    with pytest.raises(ValidationError):
        _SERVER_ADAPTER.validate_python({"type": "not_a_real_kind", "content": "x"})


def test_server_message_rejects_extra_fields() -> None:
    """``extra='forbid'`` keeps the wire shape strict — a stray field is a
    contract bug, not silently ignored."""

    with pytest.raises(ValidationError):
        _SERVER_ADAPTER.validate_python(
            {"type": "narration_chunk", "content": "ok", "stowaway": "bad"}
        )


# ---------------------------------------------------------------------------
# ClientMessage variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        ClientPcAction(character_id="ch-1", content="I attack the goblin.", kind="combat"),
        ClientPcAction(character_id=None, content="I look around.", kind="look"),
        ClientWhisperToDm(character_id="ch-1", content="Can I check the door?"),
        ClientOutOfBandChat(content="ooc: brb"),
        ClientPing(nonce="42"),
    ],
)
def test_client_message_round_trip(msg: ClientMessage) -> None:
    parsed = _client_round_trip(msg)
    assert type(parsed) is type(msg)
    assert parsed == msg


def test_client_pc_action_kind_default_is_other() -> None:
    """A pc_action without an explicit ``kind`` defaults to ``other`` — that
    keeps the gate from rejecting non-combat actions for not declaring."""

    msg = _CLIENT_ADAPTER.validate_python({"type": "pc_action", "content": "I look."})
    assert isinstance(msg, ClientPcAction)
    assert msg.kind == "other"


def test_client_pc_action_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        _CLIENT_ADAPTER.validate_python({"type": "pc_action", "content": ""})


def test_client_pc_action_rejects_bad_kind() -> None:
    with pytest.raises(ValidationError):
        _CLIENT_ADAPTER.validate_python(
            {"type": "pc_action", "content": "x", "kind": "telekinesis"}
        )


def test_whisper_requires_at_least_one_audience() -> None:
    """Whisper without an audience would be a public message — refuse it."""

    with pytest.raises(ValidationError):
        Whisper(tool_call_id="tc-x", audience=[], content="oops")


# ---------------------------------------------------------------------------
# Cross-shape sanity
# ---------------------------------------------------------------------------


def test_pc_action_appears_in_both_unions() -> None:
    """Spec §9 lists ``pc_action`` as a server → client message; the
    client → server side also uses ``pc_action`` for player input. They
    are different shapes (server includes ``user_id``; client includes
    ``kind``) but share the discriminator value — confirm both unions
    parse their own shape and reject the other."""

    server_pc = {"type": "pc_action", "user_id": "u-1", "character_id": "ch-1", "content": "x"}
    client_pc = {"type": "pc_action", "character_id": "ch-1", "content": "x", "kind": "combat"}

    server_parsed = _SERVER_ADAPTER.validate_python(server_pc)
    assert isinstance(server_parsed, PcAction)

    client_parsed = _CLIENT_ADAPTER.validate_python(client_pc)
    assert isinstance(client_parsed, ClientPcAction)
    assert client_parsed.kind == "combat"

    # ``server`` shape has ``user_id`` which the client variant forbids;
    # ``client`` shape has ``kind`` which the server variant forbids. Each
    # union should reject the other's payload (extra='forbid' enforces it).
    with pytest.raises(ValidationError):
        _SERVER_ADAPTER.validate_python(client_pc)
    with pytest.raises(ValidationError):
        _CLIENT_ADAPTER.validate_python(server_pc)


def test_dump_json_is_bytes() -> None:
    """The pubsub layer publishes bytes-on-the-wire; confirm the
    TypeAdapter's dump_json returns bytes (not str) so callers don't
    need to encode again."""

    raw: Any = _SERVER_ADAPTER.dump_json(NarrationChunk(content="hi"))
    assert isinstance(raw, bytes)
