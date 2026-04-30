"""WebSocket message types for the multiplayer hub (spec §9).

Discriminated unions on a ``type`` field for both directions:

* **Server → client** carries the orchestrator-emitted events plus
  presence and snapshot frames. The hub fans these out to every WS
  attached to ``session:{session_id}`` (whisper variants filtered to
  the addressed audience at the broadcast layer, never at storage).
* **Client → server** carries player intent. The hub validates by
  ``type`` and dispatches: ``pc_action`` triggers the orchestrator;
  ``ping`` round-trips for keepalive; the rest are reserved.

Each variant carries its own ``Literal[type]`` discriminator so the
``ServerMessage`` / ``ClientMessage`` unions parse without ad-hoc dict
introspection. Bytes-on-the-wire is JSON; ``model_dump_json`` /
``model_validate_json`` handle round-tripping.

Spec §9 enumerates: ``narration_chunk``, ``narration_complete``,
``pc_action``, ``whisper``, ``dice_roll``, ``state_update``,
``image_pending``, ``image_ready``, ``presence``. Phase 4 implements
all except the two image kinds (Phase 5 ships handlers for those —
the wire shape is locked here so the frontend is forward-compatible).

Adding a ``dm_error`` variant the spec doesn't enumerate but the
orchestrator emits — the table needs to know when something went wrong
server-side. A ``snapshot`` variant ships the on-connect catch-up
payload (recent messages, active encounter, current actor, presence).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _BaseMessage(BaseModel):
    """Shared config for every WS message variant."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Helpers used inside snapshots and presence broadcasts
# ---------------------------------------------------------------------------


class PresenceEntry(_BaseMessage):
    """One connected (user, character) pair within a session."""

    user_id: str
    username: str
    character_id: str | None
    character_name: str | None


class CurrentActor(_BaseMessage):
    """The active initiative entry — drives the client-side highlight
    and matches the server-side initiative gate."""

    encounter_id: str
    participant_id: str
    name: str
    is_player: bool
    round_number: int


class SnapshotMessage(_BaseMessage):
    """One historical session message rendered into a snapshot.

    The hub filters whispers the receiving user can't see before
    serialising; a snapshot for player B never includes a whisper
    addressed only to player A.
    """

    id: str
    sender_kind: str
    sender_id: str | None
    audience: list[str]
    content: str
    created_at: str


# ---------------------------------------------------------------------------
# Server → client
# ---------------------------------------------------------------------------


class NarrationChunk(_BaseMessage):
    """One streamed fragment from the DM. The client appends to the
    currently-rendering DM message; the canonical full text arrives in
    the trailing :class:`NarrationComplete`."""

    type: Literal["narration_chunk"] = "narration_chunk"
    content: str


class NarrationComplete(_BaseMessage):
    """End-of-narration marker carrying the persisted message id and the
    full assistant text (so reconnecting clients can render in one go
    rather than reassembling chunks)."""

    type: Literal["narration_complete"] = "narration_complete"
    message_id: str
    content: str


class PcAction(_BaseMessage):
    """A player action surfaced to other clients in the session.

    Echoed by the hub to every connection except the originator so each
    player sees what other tables have just declared (the originating
    client already rendered its own input optimistically).
    """

    type: Literal["pc_action"] = "pc_action"
    character_id: str | None
    user_id: str
    content: str


class Whisper(_BaseMessage):
    """Private DM whisper to one character. The hub filters at broadcast
    so only the addressed character's connections receive it; the full
    content lives in ``session_messages`` for the DM's prompt history
    (invariant from spec §9 — never redact at storage).
    """

    type: Literal["whisper"] = "whisper"
    tool_call_id: str
    audience: list[str] = Field(min_length=1)
    content: str


class DiceRoll(_BaseMessage):
    """A roll the engine performed. The table renders it in the dice
    history sidebar and in the narration when the orchestrator says so."""

    type: Literal["dice_roll"] = "dice_roll"
    tool_call_id: str
    expression: str
    total: int
    individual: list[int]
    purpose: str
    target: dict[str, Any] | None = None


class StateUpdate(_BaseMessage):
    """A persisted state mutation (HP delta, location change, encounter
    start/end). Side effects are passed through verbatim from the
    handler so the client can refresh the affected widget."""

    type: Literal["state_update"] = "state_update"
    tool_call_id: str
    side_effects: dict[str, Any]


class ImagePending(_BaseMessage):
    """Phase 5 — placeholder card. Wire shape only in Phase 4."""

    type: Literal["image_pending"] = "image_pending"
    image_id: str
    placeholder: str = ""


class ImageReady(_BaseMessage):
    """Phase 5 — image generation complete. Wire shape only in Phase 4."""

    type: Literal["image_ready"] = "image_ready"
    image_id: str
    url: str


class Presence(_BaseMessage):
    """Who is currently connected to this session.

    ``connected`` is the full list (after a join/leave delta) so a
    reconnecting client doesn't need to track diffs to converge — the
    hub is the source of truth.
    """

    type: Literal["presence"] = "presence"
    connected: list[PresenceEntry]


class DmError(_BaseMessage):
    """Server-side failure (orchestrator crash, validation reject, etc.).

    ``reason`` is short and machine-readable; ``message`` is the
    human-readable copy the client surfaces.
    """

    type: Literal["dm_error"] = "dm_error"
    reason: str
    message: str


class Pong(_BaseMessage):
    """Server-side reply to :class:`ClientPing`. Carries the same nonce
    so the client can match request/response timing."""

    type: Literal["pong"] = "pong"
    nonce: str = ""


class Snapshot(_BaseMessage):
    """Catch-up payload sent on connect (and on reconnect-with-snapshot).

    ``messages`` is the last 50 ``session_messages`` chronologically.
    ``current_actor`` reflects the active initiative slot when the
    session has an active encounter — ``None`` means out of combat or
    no active encounter. ``connected`` is the current presence roster.
    """

    type: Literal["snapshot"] = "snapshot"
    session_id: str
    current_location_id: str | None
    current_actor: CurrentActor | None
    messages: list[SnapshotMessage]
    connected: list[PresenceEntry]


ServerMessage = Annotated[
    NarrationChunk
    | NarrationComplete
    | PcAction
    | Whisper
    | DiceRoll
    | StateUpdate
    | ImagePending
    | ImageReady
    | Presence
    | DmError
    | Pong
    | Snapshot,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Client → server
# ---------------------------------------------------------------------------


class ClientPcAction(_BaseMessage):
    """A player declares an action.

    The server validates campaign membership and (during combat) the
    initiative gate before forwarding to the orchestrator. ``kind``
    distinguishes combat actions (which go through the gate) from
    non-combat ones (talk, look, other) which everyone can submit at
    any time.
    """

    type: Literal["pc_action"] = "pc_action"
    character_id: str | None = None
    content: str = Field(min_length=1, max_length=2000)
    kind: Literal["combat", "talk", "look", "other"] = "other"


class ClientWhisperToDm(_BaseMessage):
    """A player whispers privately to the DM (different from a DM-to-PC
    whisper — that's the server-side ``whisper`` tool). Phase 4 stub:
    accepted, persisted with audience=['dm'], not surfaced to other
    players; not currently triggering a DM turn."""

    type: Literal["whisper_to_dm"] = "whisper_to_dm"
    character_id: str | None = None
    content: str = Field(min_length=1, max_length=2000)


class ClientOutOfBandChat(_BaseMessage):
    """Table-talk that isn't a declared action. Visible to all connected
    players; doesn't trigger a DM turn."""

    type: Literal["out_of_band_chat"] = "out_of_band_chat"
    content: str = Field(min_length=1, max_length=2000)


class ClientPing(_BaseMessage):
    """Keepalive. The hub replies with :class:`Pong` carrying the same
    nonce."""

    type: Literal["ping"] = "ping"
    nonce: str = ""


ClientMessage = Annotated[
    ClientPcAction | ClientWhisperToDm | ClientOutOfBandChat | ClientPing,
    Field(discriminator="type"),
]


__all__ = [
    "ClientMessage",
    "ClientOutOfBandChat",
    "ClientPcAction",
    "ClientPing",
    "ClientWhisperToDm",
    "CurrentActor",
    "DiceRoll",
    "DmError",
    "ImagePending",
    "ImageReady",
    "NarrationChunk",
    "NarrationComplete",
    "PcAction",
    "Pong",
    "Presence",
    "PresenceEntry",
    "ServerMessage",
    "Snapshot",
    "SnapshotMessage",
    "StateUpdate",
    "Whisper",
]
