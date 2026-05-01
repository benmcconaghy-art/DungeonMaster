"""WebSocket endpoint for the multiplayer table.

WSS ``/ws/session/{session_id}`` — every connected client subscribes to
the per-session Valkey channel and receives narration, dice rolls,
state updates, whispers (filtered to addressed audience), presence,
and errors. Players submit actions, whispers, out-of-band chat, and
keepalive pings on the same socket.

Lifecycle:

1. **Auth and membership.** Cookie-session ``user_id`` resolves to a
   :class:`~app.db.models.User`. The session must exist and the user
   must be a campaign member. Anything else -> 4401 close.
2. **Accept.** Build an in-memory ``conn_id``; record the connection in
   :mod:`app.realtime.presence`; broadcast a ``presence`` frame to the
   session.
3. **Snapshot.** Send the current state catch-up to the new client
   (recent messages, current location, active actor if combat,
   roster). Whispers in the message history are pre-filtered before
   serialisation — a client never sees a whisper not addressed to its
   user.
4. **Subscriber task.** Spawn an inner task that pulls every
   :class:`~app.realtime.messages.ServerMessage` from
   :func:`Pubsub.subscribe`; the task forwards each frame to the
   socket, dropping whispers the receiving user shouldn't see.
5. **Receive loop.** Parse incoming JSON as a
   :class:`~app.realtime.messages.ClientMessage`; dispatch by type:

   - ``ping`` → reply with :class:`~app.realtime.messages.Pong` carrying
     the same nonce.
   - ``pc_action`` → echo a :class:`~app.realtime.messages.PcAction`
     to the session, then trigger the orchestrator's :func:`take_turn`
     in a background task that publishes each event through Valkey.
     Combat-kind actions go through :func:`_check_initiative_gate`
     server-side: out of combat any action passes; during combat the
     submitting character must match the active initiative slot or
     the hub returns a :class:`~app.realtime.messages.DmError` with
     ``reason="not_your_turn"``. The gate is server-side; a UI that
     skips its visual hint and sends a combat action anyway still
     gets rejected.
   - ``whisper_to_dm`` → persist with audience=``["dm"]``; not
     surfaced to other players, no immediate orchestrator turn (the
     whisper context flows in via the next prompt).
   - ``out_of_band_chat`` → broadcast verbatim, no orchestrator turn.
6. **Disconnect.** Cancel the subscriber task; remove from presence;
   broadcast the new roster.

Concurrency: a per-session ``asyncio.Lock`` serialises orchestrator
turns within a single worker (spec §13 single gunicorn worker). Two
players acting concurrently queue; the second player's action waits
for the first's turn to finish. Without this, simultaneous tool calls
race against the same SQLAlchemy session and the LLM sees partially-
stale prompts.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select

from app.db import models
from app.db.session import SessionLocal
from app.orchestrator.dm import take_turn
from app.realtime import messages as ws_msgs
from app.realtime.bridge import orchestrator_event_to_ws
from app.realtime.presence import get_presence_registry
from app.realtime.pubsub import DmPubsubError, get_pubsub

log = logging.getLogger(__name__)

router = APIRouter()

# Discriminated-union adapter for inbound client frames. Cached so we
# don't rebuild on every message.
_CLIENT_MSG_ADAPTER: TypeAdapter[ws_msgs.ClientMessage] = TypeAdapter(ws_msgs.ClientMessage)

# Per-session lock for the orchestrator. Two pc_actions on the same
# session serialise; turn N+1 waits for turn N to finish before
# building its prompt.
_session_turn_locks: dict[str, asyncio.Lock] = {}


def _session_lock(session_id: str) -> asyncio.Lock:
    """Return (or lazily create) the per-session orchestrator lock."""

    lock = _session_turn_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_turn_locks[session_id] = lock
    return lock


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


_SNAPSHOT_MSG_LIMIT = 50
_SNAPSHOT_IMAGE_LIMIT = 20


def _image_url(image_id: str) -> str:
    """Mirror of ``app.images.worker._image_url`` so the snapshot
    builds the same URL shape as the live ``image_ready`` broadcast.
    Kept private here to avoid a worker → ws cross-import; the worker's
    function is process-local and we don't want a pretend dependency
    edge in either direction."""

    return f"/api/images/{image_id}.png"


async def _build_snapshot(
    db: Any,
    *,
    session_id: str,
    visible_character_ids: set[str],
) -> ws_msgs.Snapshot:
    """Compose the on-connect catch-up payload.

    ``visible_character_ids`` is the receiving user's character set,
    used to filter whispers in the message history. The DM still sees
    every whisper in its own prompt history; the snapshot is what the
    *client* renders, so a whisper to PC A never appears in PC B's
    snapshot.
    """

    session = await db.get(models.Session, session_id)
    if session is None:
        raise ValueError(f"unknown session_id: {session_id!r}")

    msg_stmt = (
        select(models.SessionMessage)
        .where(models.SessionMessage.session_id == session_id)
        .order_by(models.SessionMessage.created_at.desc())
        .limit(_SNAPSHOT_MSG_LIMIT)
    )
    raw_msgs = list((await db.scalars(msg_stmt)).all())
    raw_msgs.reverse()

    snapshot_msgs: list[ws_msgs.SnapshotMessage] = []
    for m in raw_msgs:
        # Whisper filtering at the snapshot layer: only show the message
        # if the audience is empty (public) or includes any of this
        # user's character ids.
        if m.audience and not any(cid in visible_character_ids for cid in m.audience):
            continue
        snapshot_msgs.append(
            ws_msgs.SnapshotMessage(
                id=m.id,
                sender_kind=m.sender_kind,
                sender_id=m.sender_id,
                audience=list(m.audience),
                content=m.content,
                created_at=m.created_at,
            )
        )

    image_events = await _build_image_events(
        db, session_id=session_id, message_window_start=_window_start(snapshot_msgs)
    )

    current_actor = await _resolve_current_actor(db, session_id=session_id)
    roster = await get_presence_registry().roster(session_id)

    return ws_msgs.Snapshot(
        session_id=session_id,
        current_location_id=session.current_location_id,
        current_actor=current_actor,
        messages=snapshot_msgs,
        image_events=image_events,
        connected=roster,
    )


def _window_start(snapshot_msgs: list[ws_msgs.SnapshotMessage]) -> str | None:
    """Earliest ``created_at`` across the snapshot's message window.

    Returned to filter image events to the same time slice. ``None``
    means the snapshot has no messages — no point fetching images
    that would visually float without context.
    """

    if not snapshot_msgs:
        return None
    return min(m.created_at for m in snapshot_msgs)


async def _build_image_events(
    db: Any, *, session_id: str, message_window_start: str | None
) -> list[ws_msgs.SnapshotImageEvent]:
    """Fetch the recent successful images for ``session_id`` so a
    reconnecting client renders the scene illustrations alongside the
    narration that produced them.

    Only images bound to this session via
    ``generated_images.session_id`` (Phase 6 prep migration) are
    returned. The order matches the snapshot's chronological
    ``messages`` ordering so the client can interleave by
    ``created_at``.

    Scope (matches :class:`~app.realtime.messages.SnapshotImageEvent`):
    only ``ready`` events are surfaced. The worker doesn't persist a
    row for failed generations, and ``pending`` lives only on the
    queue. Reconnecting into a brief in-flight window means the live
    ``image_ready`` event still has to land for that one slot —
    accepted gap for Phase 6.
    """

    if message_window_start is None:
        return []
    stmt = (
        select(models.GeneratedImage)
        .where(models.GeneratedImage.session_id == session_id)
        .where(models.GeneratedImage.created_at >= message_window_start)
        .order_by(models.GeneratedImage.created_at.asc())
        .limit(_SNAPSHOT_IMAGE_LIMIT)
    )
    rows = list((await db.scalars(stmt)).all())
    return [
        ws_msgs.SnapshotImageEvent(
            image_id=row.id,
            url=_image_url(row.id),
            status="ready",
            created_at=row.created_at,
        )
        for row in rows
    ]


@dataclass(frozen=True, slots=True)
class _GateResult:
    """Outcome of an initiative-gate check.

    The hub uses the typed ``reason`` / ``message`` to surface a
    :class:`~app.realtime.messages.DmError` to the offending client
    when a combat action is rejected; ``allowed=True`` short-circuits
    that path.
    """

    allowed: bool
    reason: str = ""
    message: str = ""


async def _check_initiative_gate(*, session_id: str, character_id: str | None) -> _GateResult:
    """Server-side initiative gate for combat-kind pc_action messages.

    Out of combat (no active encounter), every action is allowed.
    During combat, only the current actor's character can submit a
    combat action — anyone else gets ``not_your_turn``. The gate is
    enforced server-side regardless of client behaviour; a buggy or
    malicious UI cannot bypass it by sending the message anyway.

    Edge case: if combat is active but the current initiative slot is
    a non-PC (a monster's turn), nobody can submit a combat action —
    the DM is on the clock and will narrate the monster's turn through
    the orchestrator. The reject reason is the same ``not_your_turn``;
    only the message text differs.
    """

    async with SessionLocal() as db:
        actor = await _resolve_current_actor(db, session_id=session_id)
    if actor is None:
        return _GateResult(allowed=True)
    if not actor.is_player:
        return _GateResult(
            allowed=False,
            reason="not_your_turn",
            message=(
                f"It's {actor.name}'s turn (round {actor.round_number}); the DM"
                " is resolving non-player actions."
            ),
        )
    if character_id is None:
        return _GateResult(
            allowed=False,
            reason="not_your_turn",
            message=(
                f"It's {actor.name}'s turn; submit a combat action with"
                " character_id set to the active PC."
            ),
        )
    if actor.participant_id != character_id:
        return _GateResult(
            allowed=False,
            reason="not_your_turn",
            message=f"It's {actor.name}'s turn (round {actor.round_number}).",
        )
    return _GateResult(allowed=True)


async def _resolve_current_actor(db: Any, *, session_id: str) -> ws_msgs.CurrentActor | None:
    """Read the active encounter (if any) and surface the participant
    whose turn it is. Returns ``None`` out of combat or when no
    encounter is active."""

    enc_stmt = (
        select(models.Encounter)
        .where(models.Encounter.session_id == session_id)
        .where(models.Encounter.status == "active")
        .order_by(models.Encounter.created_at.desc())
        .limit(1)
    )
    encounter = (await db.scalars(enc_stmt)).first()
    if encounter is None:
        return None
    initiative = encounter.initiative or []
    if not initiative:
        return None
    idx = max(0, min(encounter.current_turn, len(initiative) - 1))
    entry = initiative[idx]
    if not isinstance(entry, dict):
        return None
    return ws_msgs.CurrentActor(
        encounter_id=encounter.id,
        participant_id=str(entry.get("participant_id", "")),
        name=str(entry.get("name", "")),
        is_player=bool(entry.get("is_player", False)),
        round_number=int(encounter.round_number),
    )


# ---------------------------------------------------------------------------
# WS endpoint
# ---------------------------------------------------------------------------


async def _resolve_user(ws: WebSocket) -> models.User | None:
    """Resolve the cookie session's ``user_id`` to a ``User`` row.

    Returns ``None`` if the cookie is absent, the session is empty, or
    the user no longer exists.
    """

    user_id = ws.session.get("user_id")
    if not user_id:
        return None
    async with SessionLocal() as db:
        return await db.get(models.User, user_id)


async def _resolve_session_and_membership(
    ws: WebSocket, *, session_id: str, user: models.User
) -> tuple[models.Session, list[models.Character]] | None:
    """Verify session + membership; return (session, user's characters)
    on success, ``None`` on any rejection condition.

    The user's character list is the set of PCs they own in the parent
    campaign — used for whisper filtering and the snapshot's
    ``visible_character_ids``.
    """

    async with SessionLocal() as db:
        session = await db.get(models.Session, session_id)
        if session is None:
            return None
        membership = await db.get(models.CampaignMember, (session.campaign_id, user.id))
        if membership is None:
            return None
        chars = list(
            (
                await db.scalars(
                    select(models.Character)
                    .where(models.Character.campaign_id == session.campaign_id)
                    .where(models.Character.user_id == user.id)
                    .order_by(models.Character.name)
                )
            ).all()
        )
        return session, chars


@router.websocket("/ws/session/{session_id}")
async def session_socket(ws: WebSocket, session_id: str) -> None:
    """The session WebSocket. Spec §9; full lifecycle in module docstring."""

    user = await _resolve_user(ws)
    if user is None:
        # Starlette WebSocket close codes: 1008 for policy violation,
        # which is what an unauthenticated connection is. Return before
        # accepting so the client sees a handshake reject, not a post-
        # accept close.
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    resolved = await _resolve_session_and_membership(ws, session_id=session_id, user=user)
    if resolved is None:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    _session_row, characters = resolved
    visible_character_ids = {c.id for c in characters}
    primary_char = characters[0] if characters else None

    await ws.accept()

    conn_id = str(uuid.uuid4())
    presence = get_presence_registry()
    pubsub = get_pubsub()

    # Record the connection and broadcast presence so other clients see
    # the join. Do this BEFORE the snapshot send so the new client's
    # snapshot includes itself in the roster (one consistent view).
    roster = await presence.connect(
        session_id=session_id,
        user_id=user.id,
        username=user.username,
        character_id=primary_char.id if primary_char else None,
        character_name=primary_char.name if primary_char else None,
        conn_id=conn_id,
    )
    try:
        await pubsub.publish(session_id, ws_msgs.Presence(connected=roster))
    except DmPubsubError:
        # Valkey down → close the WS with an obvious reason. Don't paper
        # over it with retries; spec §13 has Valkey as a hard dependency.
        log.exception("ws: presence publish failed; closing socket")
        await presence.disconnect(
            session_id=session_id,
            user_id=user.id,
            character_id=primary_char.id if primary_char else None,
            conn_id=conn_id,
        )
        await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    # Snapshot the current state for the new client.
    async with SessionLocal() as db:
        snapshot = await _build_snapshot(
            db, session_id=session_id, visible_character_ids=visible_character_ids
        )
    await ws.send_text(snapshot.model_dump_json())

    # Spawn the subscriber that fans Valkey messages out to this socket.
    subscriber = asyncio.create_task(
        _subscriber_task(
            ws=ws,
            session_id=session_id,
            visible_character_ids=visible_character_ids,
        )
    )

    turn_tasks: set[asyncio.Task[None]] = set()

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                break
            await _handle_inbound(
                raw=raw,
                ws=ws,
                session_id=session_id,
                user=user,
                primary_char=primary_char,
                turn_tasks=turn_tasks,
            )
    finally:
        # Cancel the subscriber and any in-flight turn tasks so the
        # connection's resources release promptly. Awaiting the
        # cancellations is best-effort; a misbehaving turn task that
        # hangs would block shutdown otherwise.
        subscriber.cancel()
        for task in list(turn_tasks):
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await subscriber
        for task in list(turn_tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        # Remove from presence; broadcast the new roster so other
        # clients see the leave. Failures here are logged but not
        # propagated — the socket is already closing.
        roster_after = await presence.disconnect(
            session_id=session_id,
            user_id=user.id,
            character_id=primary_char.id if primary_char else None,
            conn_id=conn_id,
        )
        try:
            await pubsub.publish(session_id, ws_msgs.Presence(connected=roster_after))
        except DmPubsubError:
            log.exception("ws: presence-disconnect publish failed")


async def _subscriber_task(
    *,
    ws: WebSocket,
    session_id: str,
    visible_character_ids: set[str],
) -> None:
    """Pull messages from the session's Valkey channel and forward to
    the socket. Whispers are filtered against the receiving user's
    character ids; nothing else is filtered."""

    pubsub = get_pubsub()
    try:
        async for msg in pubsub.subscribe(session_id):
            if isinstance(msg, ws_msgs.Whisper) and not any(
                cid in visible_character_ids for cid in msg.audience
            ):
                continue
            try:
                await ws.send_text(msg.model_dump_json())
            except WebSocketDisconnect:
                return
            except RuntimeError:
                # send_text on a closed socket raises RuntimeError.
                # Surface as a clean exit; the receive-side break has
                # already (or will shortly) fire.
                return
    except asyncio.CancelledError:
        raise
    except DmPubsubError:
        log.exception("ws subscriber: pubsub failure; dropping connection")


async def _handle_inbound(
    *,
    raw: str,
    ws: WebSocket,
    session_id: str,
    user: models.User,
    primary_char: models.Character | None,
    turn_tasks: set[asyncio.Task[None]],
) -> None:
    """Dispatch one inbound client frame.

    Validation errors (malformed JSON, unknown ``type``, missing
    fields) are dropped silently — the contract is that clients send
    well-formed frames. A misbehaving client that floods garbage
    shouldn't be rewarded with a typed error response per frame.
    """

    try:
        client_msg = _CLIENT_MSG_ADAPTER.validate_json(raw)
    except ValidationError:
        return

    pubsub = get_pubsub()

    if isinstance(client_msg, ws_msgs.ClientPing):
        await ws.send_text(ws_msgs.Pong(nonce=client_msg.nonce).model_dump_json())
        return

    if isinstance(client_msg, ws_msgs.ClientPcAction):
        char_id = client_msg.character_id or (primary_char.id if primary_char else None)

        # Initiative gating: combat-kind actions during an active
        # encounter must be the current actor's. Non-combat (talk,
        # look, other) and any action when there's no active encounter
        # are unconditionally accepted.
        if client_msg.kind == "combat":
            gate = await _check_initiative_gate(session_id=session_id, character_id=char_id)
            if not gate.allowed:
                await ws.send_text(
                    ws_msgs.DmError(
                        reason=gate.reason,
                        message=gate.message,
                    ).model_dump_json()
                )
                return

        # Echo the player's intent to the session so other clients see
        # what was declared (the originating client's UI already
        # rendered the action optimistically).
        await pubsub.publish(
            session_id,
            ws_msgs.PcAction(
                character_id=char_id,
                user_id=user.id,
                content=client_msg.content,
            ),
        )
        task = asyncio.create_task(
            _run_turn(
                session_id=session_id,
                sender_user_id=user.id,
                sender_character_id=char_id,
                content=client_msg.content,
            )
        )
        turn_tasks.add(task)
        task.add_done_callback(turn_tasks.discard)
        return

    if isinstance(client_msg, ws_msgs.ClientWhisperToDm):
        # Phase 4 stub: persist as a session_message with audience=['dm']
        # so the DM's prompt history sees it on the next turn. No
        # broadcast — other players never see whisper-to-DM content.
        async with SessionLocal() as db:
            char_id = client_msg.character_id or (primary_char.id if primary_char else None)
            db.add(
                models.SessionMessage(
                    session_id=session_id,
                    sender_kind="player",
                    sender_id=char_id,
                    audience=["dm"],
                    content=client_msg.content,
                )
            )
            await db.commit()
        return

    if isinstance(client_msg, ws_msgs.ClientOutOfBandChat):
        # OOC chat — broadcast verbatim as a PcAction (with no
        # character_id, no orchestrator turn). Phase 6 gets a dedicated
        # ``out_of_band_chat`` server-side message type if we want a
        # different render style.
        await pubsub.publish(
            session_id,
            ws_msgs.PcAction(
                character_id=None,
                user_id=user.id,
                content=client_msg.content,
            ),
        )
        return


async def _run_turn(
    *,
    session_id: str,
    sender_user_id: str,
    sender_character_id: str | None,
    content: str,
) -> None:
    """Run :func:`take_turn` in the background and publish each yielded
    event to the session's Valkey channel.

    Holds the per-session orchestrator lock so two pc_actions on the
    same session serialise (one finishes before the next builds its
    prompt). Without the lock, simultaneous tool calls race the
    SQLAlchemy session state and the LLM sees half-stale prompts.
    """

    pubsub = get_pubsub()
    lock = _session_lock(session_id)
    async with lock, SessionLocal() as db:
        try:
            async for event in take_turn(
                db,
                session_id=session_id,
                sender_user_id=sender_user_id,
                sender_character_id=sender_character_id,
                content=content,
            ):
                ws_msg = orchestrator_event_to_ws(event)
                if ws_msg is None:
                    continue
                try:
                    await pubsub.publish(session_id, ws_msg)
                except DmPubsubError:
                    log.exception("ws: pubsub publish failed mid-turn")
                    # Stop the turn — without the broadcast the
                    # narration only reaches the originator's subscriber
                    # if Valkey came back; cleaner to surface to the
                    # table and return.
                    return
        except Exception:
            log.exception("ws: take_turn raised an unexpected error")
            with contextlib.suppress(DmPubsubError):
                await pubsub.publish(
                    session_id,
                    ws_msgs.DmError(
                        reason="orchestrator_crash",
                        message="DM turn crashed; the table has been notified.",
                    ),
                )


__all__ = ["router"]
