"""Tests for ``app.realtime.presence`` — the per-session connection
registry that drives :class:`~app.realtime.messages.Presence` broadcasts.

What the hub depends on:

* ``connect`` adds an entry; ``disconnect`` removes it; both return the
  full roster so the hub can send a self-contained ``presence`` frame.
* The roster is sorted deterministically — clients redraw every frame,
  so a shuffle on every join would be visually noisy.
* Two attachments by the same (user, character) collapse in the roster
  (one tab vs. two — players see one entry, not two).
* Sessions are isolated — a connect to session A doesn't appear in
  session B.

The registry is a process-local singleton (spec §13 single-worker).
Tests reset it via :func:`reset_for_tests` so each test starts empty.
"""

from __future__ import annotations

import pytest

from app.realtime.presence import (
    PresenceRegistry,
    get_presence_registry,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_for_tests()


@pytest.mark.asyncio
async def test_connect_returns_roster_with_one_entry() -> None:
    reg = PresenceRegistry()
    roster = await reg.connect(
        session_id="s-1",
        user_id="u-1",
        username="alice",
        character_id="ch-1",
        character_name="Tav",
        conn_id="c-A",
    )
    assert len(roster) == 1
    assert roster[0].user_id == "u-1"
    assert roster[0].username == "alice"
    assert roster[0].character_id == "ch-1"
    assert roster[0].character_name == "Tav"


@pytest.mark.asyncio
async def test_disconnect_removes_entry() -> None:
    reg = PresenceRegistry()
    await reg.connect(
        session_id="s-1",
        user_id="u-1",
        username="alice",
        character_id="ch-1",
        character_name="Tav",
        conn_id="c-A",
    )
    roster = await reg.disconnect(
        session_id="s-1",
        user_id="u-1",
        character_id="ch-1",
        conn_id="c-A",
    )
    assert roster == []


@pytest.mark.asyncio
async def test_disconnect_unknown_conn_is_idempotent() -> None:
    """Repeated disconnect from a broken-socket cleanup path shouldn't
    raise."""

    reg = PresenceRegistry()
    roster = await reg.disconnect(
        session_id="s-1", user_id="u-1", character_id="ch-1", conn_id="never-was"
    )
    assert roster == []


@pytest.mark.asyncio
async def test_two_tabs_same_user_character_collapse_in_roster() -> None:
    """Two browser tabs from the same player on the same PC should appear
    once in the roster. Disconnecting one tab keeps the entry visible —
    a player isn't 'gone' until every tab disconnects."""

    reg = PresenceRegistry()
    await reg.connect(
        session_id="s-1",
        user_id="u-1",
        username="alice",
        character_id="ch-1",
        character_name="Tav",
        conn_id="tab-A",
    )
    roster = await reg.connect(
        session_id="s-1",
        user_id="u-1",
        username="alice",
        character_id="ch-1",
        character_name="Tav",
        conn_id="tab-B",
    )
    assert len(roster) == 1, "two tabs should collapse to one roster entry"

    # Close one tab — the roster still shows the player.
    roster_after_one_close = await reg.disconnect(
        session_id="s-1", user_id="u-1", character_id="ch-1", conn_id="tab-A"
    )
    assert len(roster_after_one_close) == 1

    # Close the second tab — gone.
    roster_after_both_close = await reg.disconnect(
        session_id="s-1", user_id="u-1", character_id="ch-1", conn_id="tab-B"
    )
    assert roster_after_both_close == []


@pytest.mark.asyncio
async def test_two_users_appear_sorted_deterministically() -> None:
    """Two different players appear ordered by ``(username, character_name)``
    so clients redrawing on every frame don't see entries shuffle."""

    reg = PresenceRegistry()
    await reg.connect(
        session_id="s-1",
        user_id="u-2",
        username="bob",
        character_id="ch-2",
        character_name="Brunhild",
        conn_id="c-2",
    )
    roster = await reg.connect(
        session_id="s-1",
        user_id="u-1",
        username="alice",
        character_id="ch-1",
        character_name="Tav",
        conn_id="c-1",
    )
    assert [e.username for e in roster] == ["alice", "bob"]


@pytest.mark.asyncio
async def test_sessions_are_isolated() -> None:
    """Connecting to session A does not appear in session B."""

    reg = PresenceRegistry()
    await reg.connect(
        session_id="s-A",
        user_id="u-1",
        username="alice",
        character_id="ch-1",
        character_name="Tav",
        conn_id="c-1",
    )
    other = await reg.roster("s-B")
    assert other == []
    same = await reg.roster("s-A")
    assert len(same) == 1


@pytest.mark.asyncio
async def test_roster_does_not_mutate_state() -> None:
    """Reading the roster doesn't drop entries (it's not a ``pop``)."""

    reg = PresenceRegistry()
    await reg.connect(
        session_id="s-1",
        user_id="u-1",
        username="alice",
        character_id="ch-1",
        character_name="Tav",
        conn_id="c-1",
    )
    a = await reg.roster("s-1")
    b = await reg.roster("s-1")
    assert a == b
    assert len(a) == 1


@pytest.mark.asyncio
async def test_get_presence_registry_returns_singleton() -> None:
    a = get_presence_registry()
    b = get_presence_registry()
    assert a is b


@pytest.mark.asyncio
async def test_reset_for_tests_drops_singleton() -> None:
    a = get_presence_registry()
    await a.connect(
        session_id="s-1",
        user_id="u-1",
        username="alice",
        character_id="ch-1",
        character_name="Tav",
        conn_id="c-1",
    )
    reset_for_tests()
    b = get_presence_registry()
    assert b is not a
    fresh_roster = await b.roster("s-1")
    assert fresh_roster == []
