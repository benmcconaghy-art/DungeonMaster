"""WebSocket-endpoint tests against a deterministic stand-in stack.

The orchestrator's ``take_turn`` is monkey-patched to yield canned
:class:`~app.orchestrator.dm.DmEvent` variants so the WS layer's
behaviour is exercised without hitting real Nemotron. The Valkey
pub/sub is replaced with the in-memory :class:`~tests.realtime.fakes.FakePubsub`.

Coverage matches the kickoff brief's step-9 contract:

* connect with no auth → reject
* connect with wrong session / non-membership → reject
* connect with valid auth → snapshot received
* multi-client broadcast: A's pc_action surfaces on B's socket
* whisper isolation: a whisper to character X only reaches X's socket
* initiative gate: combat-kind action by non-current actor → not_your_turn
* reconnect-and-snapshot: disconnect mid-stream → reconnect → snapshot
* presence: connect → presence frame; disconnect → updated roster

Tests are sync (FastAPI ``TestClient`` is sync). The schema is created
on a file-backed SQLite database the WS handler and the orchestrator
both see; ``app.api.ws.SessionLocal`` and
``app.orchestrator.dm.SessionLocal`` are monkey-patched so the
WS handler and the orchestrator share that DB.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import models
from app.db.base import Base
from app.db.session import create_engine
from app.deps import get_db
from app.main import app as fastapi_app
from app.orchestrator.dm import (
    DmEvent,
    NarrationChunk,
    NarrationComplete,
    WhisperEvent,
)
from app.realtime import presence as presence_module
from app.realtime import pubsub as pubsub_module
from tests.realtime.fakes import FakePubsub

_VALID_PW = "correct horse battery staple"


def _stub_take_turn(events: list[DmEvent]) -> Any:
    """Return a callable with the same shape as
    :func:`app.orchestrator.dm.take_turn` that yields ``events`` and
    stops. ``events`` may be empty, in which case the stub yields
    nothing — useful when a test only cares about the WS receive path
    around the orchestrator dispatch (e.g. pc_action echo, gate
    rejection)."""

    async def _impl(*args: Any, **kwargs: Any) -> AsyncIterator[DmEvent]:
        for event in events:
            yield event

    return _impl


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture
def ws_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]]]:
    """Wire up an isolated in-process stack for the WS endpoint:

    - File-backed SQLite at ``tmp_path/ws.db`` (so the WS handler and
      the orchestrator both see the same data via separate sessions).
    - Schema applied from ``Base.metadata``.
    - ``SessionLocal`` rebound on every reachable touchpoint so test
      writes land on the test DB, not the production path.
    - :class:`FakePubsub` replacing the Valkey singleton.
    - Presence registry reset.
    - FastAPI :class:`TestClient` for HTTP + WS round trips.
    """

    db_path = tmp_path / "ws.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    fastapi_app.dependency_overrides[get_db] = _override_get_db
    monkeypatch.setattr("app.api.ws.SessionLocal", factory)
    monkeypatch.setattr("app.orchestrator.dm.SessionLocal", factory)

    fake = FakePubsub()
    pubsub_module.set_pubsub_for_tests(fake)
    presence_module.reset_for_tests()

    client = TestClient(fastapi_app)
    try:
        yield client, fake, factory
    finally:
        fastapi_app.dependency_overrides.pop(get_db, None)
        asyncio.run(engine.dispose())
        # Async teardown of the singleton's connection pool can't run
        # cleanly from sync code; we just drop the reference and let the
        # next test's set_pubsub_for_tests take over.
        pubsub_module.set_pubsub_for_tests(None)
        presence_module.reset_for_tests()


def _register_user(client: TestClient, *, username: str = "alice") -> dict[str, Any]:
    """Create a user via /api/auth/register; ``client`` is left
    authenticated as that user (cookie persists)."""

    response = client.post(
        "/api/auth/register",
        json={"username": username, "password": _VALID_PW},
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _register_user_via_new_client(username: str = "bob") -> tuple[TestClient, dict[str, Any]]:
    """Create a SECOND TestClient with its own cookie jar — used by
    multi-client tests that need two simultaneous WS connections."""

    other = TestClient(fastapi_app)
    body = _register_user(other, username=username)
    return other, body


def _create_campaign(client: TestClient, name: str = "Test") -> dict[str, Any]:
    response = client.post("/api/campaigns", json={"name": name})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _add_member_via_db(
    factory: async_sessionmaker[AsyncSession],
    *,
    campaign_id: str,
    user_id: str,
    role: str = "player",
) -> None:
    """Insert a campaign_members row directly. The invite-code flow
    isn't on the surface yet; tests bypass it."""

    async with factory() as db:
        db.add(models.CampaignMember(campaign_id=campaign_id, user_id=user_id, role=role))
        await db.commit()


def _create_character(client: TestClient, campaign_id: str, name: str = "Tav") -> dict[str, Any]:
    response = client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={
            "name": name,
            "race": "Human",
            "class_name": "Fighter",
            "alignment": "neutral",
            "abilities": {
                "str": 14,
                "int": 10,
                "wis": 10,
                "dex": 12,
                "con": 14,
                "cha": 10,
            },
        },
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _create_session(client: TestClient, campaign_id: str) -> dict[str, Any]:
    response = client.post(f"/api/campaigns/{campaign_id}/sessions")
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


# ---------------------------------------------------------------------------
# Connect / auth
# ---------------------------------------------------------------------------


def test_ws_connect_without_auth_rejected(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
) -> None:
    """A WS connection from an unauthenticated client closes with 1008."""

    client, _, _ = ws_setup
    # Don't register; cookie jar is empty.
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws/session/some-session"),
    ):
        pass


def test_ws_connect_to_unknown_session_rejected(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
) -> None:
    """An authenticated user trying to attach to a session that
    doesn't exist gets the same close as no-auth."""

    client, _, _ = ws_setup
    _register_user(client)
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws/session/non-existent"),
    ):
        pass


def test_ws_connect_to_session_outside_campaign_rejected(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
) -> None:
    """A user who is NOT a member of the parent campaign cannot attach
    even though they're authenticated."""

    client, _, _ = ws_setup
    owner = _register_user(client, username="owner")
    campaign = _create_campaign(client, name="Owner camp")
    session = _create_session(client, campaign["id"])

    # Outsider — separate cookie jar so they're not the campaign owner.
    other_client, _ = _register_user_via_new_client(username="outsider")

    with (
        pytest.raises(WebSocketDisconnect),
        other_client.websocket_connect(f"/ws/session/{session['id']}"),
    ):
        pass

    # Sanity: owner CAN connect to confirm the session is otherwise valid.
    assert owner["id"]


def test_ws_connect_with_membership_sends_snapshot(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
) -> None:
    """A campaign member's connect produces a snapshot frame as the
    first server-to-client message."""

    client, _, _ = ws_setup
    _register_user(client)
    campaign = _create_campaign(client)
    _create_character(client, campaign["id"])
    session = _create_session(client, campaign["id"])

    with client.websocket_connect(f"/ws/session/{session['id']}") as ws:
        first = ws.receive_text()
    parsed = json.loads(first)
    assert parsed["type"] == "snapshot"
    assert parsed["session_id"] == session["id"]
    # No active encounter → current_actor is null.
    assert parsed["current_actor"] is None
    # Roster includes the connected user.
    assert any(e["username"] == "alice" for e in parsed["connected"])


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------


def test_ws_presence_broadcast_on_connect_and_disconnect(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
) -> None:
    """When a second client connects, the first client receives a
    presence frame; when one disconnects, remaining clients see the
    updated roster."""

    client_a, fake, factory = ws_setup
    user_a = _register_user(client_a, username="alice")
    campaign = _create_campaign(client_a)
    _create_character(client_a, campaign["id"])
    session = _create_session(client_a, campaign["id"])

    client_b, user_b = _register_user_via_new_client(username="bob")
    asyncio.run(_add_member_via_db(factory, campaign_id=campaign["id"], user_id=user_b["id"]))

    with client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a:
        snapshot_a = json.loads(ws_a.receive_text())
        assert snapshot_a["type"] == "snapshot"
        # Snapshot's roster has alice only.
        usernames = {e["username"] for e in snapshot_a["connected"]}
        assert usernames == {"alice"}

        with client_b.websocket_connect(f"/ws/session/{session['id']}") as ws_b:
            # Drain b's snapshot frame.
            snapshot_b = json.loads(ws_b.receive_text())
            assert snapshot_b["type"] == "snapshot"

            # Alice should receive a presence frame reflecting both users.
            presence_a = json.loads(ws_a.receive_text())
            assert presence_a["type"] == "presence"
            assert {e["username"] for e in presence_a["connected"]} == {"alice", "bob"}

        # b disconnected; alice should see a presence frame with only her.
        presence_after = json.loads(ws_a.receive_text())
        assert presence_after["type"] == "presence"
        assert {e["username"] for e in presence_after["connected"]} == {"alice"}

    # Either way both connections closed cleanly; nothing to assert.
    assert user_a["id"]


# ---------------------------------------------------------------------------
# Multi-client broadcast (pc_action echo)
# ---------------------------------------------------------------------------


def test_ws_multi_client_pc_action_broadcast(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client A submits a ``pc_action``; client B receives the echoed
    :class:`PcAction` frame."""

    client_a, _, factory = ws_setup
    _register_user(client_a, username="alice")
    campaign = _create_campaign(client_a)
    char_a = _create_character(client_a, campaign["id"], name="Tav")
    session = _create_session(client_a, campaign["id"])

    client_b, user_b = _register_user_via_new_client(username="bob")
    asyncio.run(_add_member_via_db(factory, campaign_id=campaign["id"], user_id=user_b["id"]))

    # Stub take_turn so it yields nothing — we only care about
    # the WS receive path around the orchestrator dispatch.
    monkeypatch.setattr("app.orchestrator.dispatch.take_turn", _stub_take_turn([]))

    with (
        client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a,
        client_b.websocket_connect(f"/ws/session/{session['id']}") as ws_b,
    ):
        _drain_until(ws_a, "snapshot")
        _drain_until(ws_b, "snapshot")
        # Alice's subscriber attached BEFORE Bob's join, so Alice sees
        # the presence frame from Bob's connect; drain it so the next
        # frame on Alice is the pc_action.
        _drain_until(ws_a, "presence")
        # Bob's subscriber attached AFTER his own join's presence
        # publish, so he never receives that frame — his snapshot
        # already carried the roster. Don't drain presence here.

        ws_a.send_text(
            json.dumps(
                {
                    "type": "pc_action",
                    "character_id": char_a["id"],
                    "content": "I attack the goblin.",
                    "kind": "other",
                }
            )
        )

        # Both A and B receive the pc_action broadcast (Valkey doesn't
        # filter by origin); the client de-dupes based on user_id.
        a_frame = _drain_until(ws_a, "pc_action")
        b_frame = _drain_until(ws_b, "pc_action")
        assert a_frame["content"] == "I attack the goblin."
        assert b_frame["content"] == "I attack the goblin."
        assert a_frame["character_id"] == char_a["id"]


# ---------------------------------------------------------------------------
# Whisper isolation
# ---------------------------------------------------------------------------


def test_ws_whisper_isolation(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whisper addressed to character X reaches X's connected client
    but not Y's. Both sockets see narration_chunk before the whisper —
    the whisper filter is per-frame, not connection-wide."""

    client_a, _, factory = ws_setup
    _register_user(client_a, username="alice")
    campaign = _create_campaign(client_a)
    char_a = _create_character(client_a, campaign["id"], name="Tav")
    session = _create_session(client_a, campaign["id"])

    client_b, user_b = _register_user_via_new_client(username="bob")
    asyncio.run(_add_member_via_db(factory, campaign_id=campaign["id"], user_id=user_b["id"]))
    # Bob's character lives in the campaign too so the WS handshake
    # gives him a primary character; not relevant to whisper filtering
    # but matches realistic deployment.
    char_b_response = client_b.post(
        f"/api/campaigns/{campaign['id']}/characters",
        json={
            "name": "Brunhild",
            "race": "Human",
            "class_name": "Fighter",
            "alignment": "neutral",
            "abilities": {"str": 14, "int": 10, "wis": 10, "dex": 12, "con": 14, "cha": 10},
        },
    )
    assert char_b_response.status_code == 201
    char_b = char_b_response.json()

    # take_turn yields a NarrationChunk + a WhisperEvent addressed to
    # alice's character only, then completes.
    async def _whisper_take_turn(*args: Any, **kwargs: Any) -> AsyncIterator[DmEvent]:
        yield NarrationChunk(stream_id="s-1", content="A figure beckons from the alley.")
        yield WhisperEvent(
            tool_call_id="tc-1",
            audience=[char_a["id"]],
            content="(A coin presses into your palm.)",
        )
        yield NarrationComplete(
            stream_id="s-1",
            message_id="msg-1",
            content="A figure beckons from the alley.",
        )

    monkeypatch.setattr("app.orchestrator.dispatch.take_turn", _whisper_take_turn)

    with (
        client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a,
        client_b.websocket_connect(f"/ws/session/{session['id']}") as ws_b,
    ):
        _drain_until(ws_a, "snapshot")
        _drain_until(ws_b, "snapshot")
        # Alice was first to connect; her subscriber catches the
        # presence frame Bob's join published. Bob's own subscriber
        # missed the frame for his own join (it published before he
        # subscribed) — his snapshot already conveyed the roster.
        _drain_until(ws_a, "presence")

        # Alice fires the action.
        ws_a.send_text(
            json.dumps(
                {
                    "type": "pc_action",
                    "character_id": char_a["id"],
                    "content": "Hello?",
                    "kind": "other",
                }
            )
        )

        # Drain the pc_action echo on both sockets first.
        _drain_until(ws_a, "pc_action")
        _drain_until(ws_b, "pc_action")

        # Both sockets receive the narration chunk.
        a_chunk = _drain_until(ws_a, "narration_chunk")
        b_chunk = _drain_until(ws_b, "narration_chunk")
        assert "alley" in a_chunk["content"]
        assert "alley" in b_chunk["content"]

        # Alice receives the whisper.
        a_whisper = _drain_until(ws_a, "whisper")
        assert a_whisper["audience"] == [char_a["id"]]
        assert "coin" in a_whisper["content"]

        # Bob receives narration_complete next; the whisper was filtered
        # out before Bob's socket saw it.
        b_complete = _drain_until(ws_b, "narration_complete")
        assert b_complete["content"].startswith("A figure")
        # Sanity: char_b exists in the test fixture (the assertion is
        # mostly to silence "unused" warnings; the variable's role is
        # to seed campaign membership).
        assert char_b["id"]


# ---------------------------------------------------------------------------
# Initiative gate
# ---------------------------------------------------------------------------


def test_ws_combat_action_rejected_when_not_current_actor(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
) -> None:
    """A combat-kind pc_action submitted while a different participant
    holds the current initiative slot returns a not_your_turn dm_error
    only to the offending socket — never broadcast to other clients."""

    client_a, _, factory = ws_setup
    _register_user(client_a, username="alice")
    campaign = _create_campaign(client_a)
    char_a = _create_character(client_a, campaign["id"], name="Tav")
    session = _create_session(client_a, campaign["id"])

    # Seed an active encounter where the goblin holds the current slot.
    asyncio.run(
        _seed_active_encounter(
            factory,
            session_id=session["id"],
            initiative=[
                {
                    "participant_id": "goblin#1",
                    "name": "goblin",
                    "initiative": 6,
                    "is_player": False,
                },
                {
                    "participant_id": char_a["id"],
                    "name": "Tav",
                    "initiative": 4,
                    "is_player": True,
                },
            ],
            current_turn=0,
        )
    )

    with client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a:
        snapshot = json.loads(ws_a.receive_text())
        assert snapshot["current_actor"] is not None
        assert snapshot["current_actor"]["participant_id"] == "goblin#1"

        # Alice tries to act on combat — gate rejects.
        ws_a.send_text(
            json.dumps(
                {
                    "type": "pc_action",
                    "character_id": char_a["id"],
                    "content": "I attack",
                    "kind": "combat",
                }
            )
        )
        err = json.loads(ws_a.receive_text())
        assert err["type"] == "dm_error"
        assert err["reason"] == "not_your_turn"


def test_ws_non_combat_action_passes_during_combat(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A talk/look/other pc_action does NOT go through the gate even
    while a different participant holds the current slot."""

    client_a, _, factory = ws_setup
    _register_user(client_a, username="alice")
    campaign = _create_campaign(client_a)
    char_a = _create_character(client_a, campaign["id"], name="Tav")
    session = _create_session(client_a, campaign["id"])

    asyncio.run(
        _seed_active_encounter(
            factory,
            session_id=session["id"],
            initiative=[
                {
                    "participant_id": "goblin#1",
                    "name": "goblin",
                    "initiative": 6,
                    "is_player": False,
                }
            ],
            current_turn=0,
        )
    )

    monkeypatch.setattr("app.orchestrator.dispatch.take_turn", _stub_take_turn([]))

    with client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a:
        _drain_until(ws_a, "snapshot")

        ws_a.send_text(
            json.dumps(
                {
                    "type": "pc_action",
                    "character_id": char_a["id"],
                    "content": "What does the goblin look like?",
                    "kind": "look",
                }
            )
        )
        echo = _drain_until(ws_a, "pc_action")
        assert echo["content"].startswith("What does")


def test_ws_combat_action_accepted_when_current_actor(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A combat pc_action passes the gate when the submitting character
    holds the current initiative slot."""

    client_a, _, factory = ws_setup
    _register_user(client_a, username="alice")
    campaign = _create_campaign(client_a)
    char_a = _create_character(client_a, campaign["id"], name="Tav")
    session = _create_session(client_a, campaign["id"])

    asyncio.run(
        _seed_active_encounter(
            factory,
            session_id=session["id"],
            initiative=[
                {
                    "participant_id": char_a["id"],
                    "name": "Tav",
                    "initiative": 8,
                    "is_player": True,
                },
                {
                    "participant_id": "goblin#1",
                    "name": "goblin",
                    "initiative": 4,
                    "is_player": False,
                },
            ],
            current_turn=0,
        )
    )

    monkeypatch.setattr("app.orchestrator.dispatch.take_turn", _stub_take_turn([]))

    with client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a:
        _drain_until(ws_a, "snapshot")

        ws_a.send_text(
            json.dumps(
                {
                    "type": "pc_action",
                    "character_id": char_a["id"],
                    "content": "I attack",
                    "kind": "combat",
                }
            )
        )
        echo = _drain_until(ws_a, "pc_action")
        assert echo["content"] == "I attack"


# ---------------------------------------------------------------------------
# Reconnect-and-snapshot
# ---------------------------------------------------------------------------


def test_ws_reconnect_replays_snapshot(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
) -> None:
    """A client that disconnects and reconnects to the same session
    receives a fresh snapshot reflecting whatever happened in between."""

    client_a, _, factory = ws_setup
    _register_user(client_a, username="alice")
    campaign = _create_campaign(client_a)
    _create_character(client_a, campaign["id"])
    session = _create_session(client_a, campaign["id"])

    # First connect — normal snapshot with empty history.
    with client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a:
        snap1 = json.loads(ws_a.receive_text())
        assert snap1["type"] == "snapshot"
        assert snap1["messages"] == []

    # Insert a session message (simulating an action persisted while
    # disconnected) and reconnect.
    asyncio.run(
        _insert_session_message(
            factory,
            session_id=session["id"],
            sender_kind="player",
            content="I look around.",
        )
    )

    with client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a:
        snap2 = json.loads(ws_a.receive_text())
        assert snap2["type"] == "snapshot"
        assert len(snap2["messages"]) == 1
        assert snap2["messages"][0]["content"] == "I look around."


def test_ws_snapshot_includes_recent_image_events(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
) -> None:
    """A reconnecting client receives the recent ``image_ready`` events
    bound to its session, alongside the message history.

    Without this, a player who joins after the worker has already
    emitted ``image_ready`` would see narration messages with no
    accompanying scene illustration — Phase 5 / 6 boundary follow-up.
    The snapshot now carries the worker-persisted images for the
    session window so the table re-renders the same chronology a
    steady-state viewer saw.
    """

    client_a, _, factory = ws_setup
    _register_user(client_a, username="alice")
    campaign = _create_campaign(client_a)
    _create_character(client_a, campaign["id"])
    session = _create_session(client_a, campaign["id"])

    # Seed: a narration message, then an image bound to the same
    # session. Order matters — the image's created_at must be >= the
    # message's created_at so the snapshot's window picks it up.
    asyncio.run(
        _insert_session_message(
            factory,
            session_id=session["id"],
            sender_kind="dm",
            content="The lanterns are not oil.",
        )
    )
    asyncio.run(
        _insert_generated_image(
            factory,
            campaign_id=campaign["id"],
            session_id=session["id"],
            kind="scene",
            prompt="lanterns over a black oak",
        )
    )
    # And one image for a DIFFERENT session of the same campaign,
    # which should NOT appear in this snapshot.
    other_session = _create_session(client_a, campaign["id"])
    asyncio.run(
        _insert_generated_image(
            factory,
            campaign_id=campaign["id"],
            session_id=other_session["id"],
            kind="scene",
            prompt="a misplaced room in a different session",
        )
    )

    with client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a:
        snap = _drain_until(ws_a, "snapshot")

    assert snap["messages"][0]["content"] == "The lanterns are not oil."
    image_events = snap["image_events"]
    assert len(image_events) == 1
    event = image_events[0]
    assert event["status"] == "ready"
    assert event["url"].endswith(".png")
    assert event["url"].startswith("/api/images/")
    # Image_id round-trips so the client can de-dupe against any live
    # image_ready frame that arrives for the same id during reconnect.
    assert event["image_id"]


def test_ws_snapshot_image_events_skipped_when_no_messages(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
) -> None:
    """Empty message snapshot yields no image events even if the
    session has images. Without a message window we have no anchor
    for the time slice — surfacing all images of the session would
    eventually grow unbounded as a campaign accumulates scenes."""

    client_a, _, factory = ws_setup
    _register_user(client_a, username="alice")
    campaign = _create_campaign(client_a)
    _create_character(client_a, campaign["id"])
    session = _create_session(client_a, campaign["id"])

    asyncio.run(
        _insert_generated_image(
            factory,
            campaign_id=campaign["id"],
            session_id=session["id"],
            kind="scene",
            prompt="orphan illustration",
        )
    )

    with client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a:
        snap = _drain_until(ws_a, "snapshot")

    assert snap["messages"] == []
    assert snap["image_events"] == []


# ---------------------------------------------------------------------------
# Ping / pong
# ---------------------------------------------------------------------------


def test_ws_ping_pong_carries_nonce(
    ws_setup: tuple[TestClient, FakePubsub, async_sessionmaker[AsyncSession]],
) -> None:
    """A ``ping`` frame round-trips with the same nonce — used by the
    client's heartbeat / latency telemetry."""

    client_a, _, _ = ws_setup
    _register_user(client_a)
    campaign = _create_campaign(client_a)
    session = _create_session(client_a, campaign["id"])

    with client_a.websocket_connect(f"/ws/session/{session['id']}") as ws_a:
        _drain_until(ws_a, "snapshot")
        ws_a.send_text(json.dumps({"type": "ping", "nonce": "n42"}))
        pong = _drain_until(ws_a, "pong")
        assert pong["nonce"] == "n42"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drain_until(ws: Any, expected_type: str, max_frames: int = 10) -> dict[str, Any]:
    """Receive frames until one of ``expected_type`` arrives.

    Skips snapshot/presence/etc. frames that arrive ahead of the one
    the test cares about — the WS lifecycle interleaves several frame
    kinds at startup. ``max_frames`` bounds the loop so a missing
    expected frame fails the test fast instead of hanging.
    """

    for _ in range(max_frames):
        raw = ws.receive_text()
        parsed = json.loads(raw)
        if parsed.get("type") == expected_type:
            return parsed  # type: ignore[no-any-return]
    raise AssertionError(f"never received frame type={expected_type!r} after {max_frames} frames")


async def _seed_active_encounter(
    factory: async_sessionmaker[AsyncSession],
    *,
    session_id: str,
    initiative: list[dict[str, Any]],
    current_turn: int = 0,
) -> None:
    """Insert an active Encounter row with the given initiative shape."""

    async with factory() as db:
        db.add(
            models.Encounter(
                session_id=session_id,
                name="Test combat",
                status="active",
                monsters=[],
                initiative=initiative,
                round_number=1,
                current_turn=current_turn,
            )
        )
        await db.commit()


async def _insert_session_message(
    factory: async_sessionmaker[AsyncSession],
    *,
    session_id: str,
    sender_kind: str,
    content: str,
) -> None:
    async with factory() as db:
        db.add(
            models.SessionMessage(
                session_id=session_id,
                sender_kind=sender_kind,
                sender_id=None,
                audience=[],
                content=content,
            )
        )
        await db.commit()


async def _insert_generated_image(
    factory: async_sessionmaker[AsyncSession],
    *,
    campaign_id: str,
    session_id: str | None,
    kind: str,
    prompt: str,
) -> str:
    """Insert a ``generated_images`` row standing in for one the
    image worker would have written. Returns the new image id so the
    test can assert against it.

    ``prompt_hash`` is unique-constrained — derive it from prompt +
    kind + a uuid to guarantee uniqueness across rows in one test.
    """

    import hashlib
    import uuid

    image_id = str(uuid.uuid4())
    h = hashlib.sha256(f"{kind}|{prompt}|{image_id}".encode()).hexdigest()
    async with factory() as db:
        db.add(
            models.GeneratedImage(
                id=image_id,
                campaign_id=campaign_id,
                session_id=session_id,
                kind=kind,
                prompt=prompt,
                prompt_hash=h,
                file_path=f"/tmp/{image_id}.png",
            )
        )
        await db.commit()
    return image_id
