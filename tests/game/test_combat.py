"""Tests for ``app.game.combat`` (turn-order helpers)."""

from __future__ import annotations

import random

import pytest

from app.game.combat import (
    advance_turn,
    current_turn,
    find,
    is_end_of_round,
    remove_participant,
)
from app.game.rules import InitiativeOrder, Participant, roll_initiative


def _make_order() -> InitiativeOrder:
    rng = random.Random(0)
    participants = [
        Participant(participant_id="p1", name="Alice", dex_modifier=2, is_player=True),
        Participant(participant_id="p2", name="Bart", dex_modifier=0, is_player=True),
        Participant(participant_id="m1", name="Goblin", dex_modifier=-1),
    ]
    return roll_initiative(participants, rng=rng)


def test_current_turn_returns_first_entry() -> None:
    order = _make_order()
    assert current_turn(order).participant_id == order.entries[0].participant_id


def test_advance_turn_increments_index() -> None:
    order = _make_order()
    next_order = advance_turn(order)
    assert next_order.index == 1
    assert next_order.round_number == 1


def test_advance_turn_wraps_to_next_round() -> None:
    order = _make_order()
    o = order
    for _ in range(len(order.entries)):
        o = advance_turn(o)
    assert o.index == 0
    assert o.round_number == 2


def test_is_end_of_round() -> None:
    order = _make_order()
    assert is_end_of_round(order) is False
    last = order
    for _ in range(len(order.entries) - 1):
        last = advance_turn(last)
    assert is_end_of_round(last) is True


def test_remove_participant_before_current_decrements_index() -> None:
    order = _make_order()
    o = advance_turn(order)  # index = 1
    first_id = order.entries[0].participant_id
    new = remove_participant(o, first_id)
    # That participant is gone; the same actor (originally at index 1) is still up.
    assert new.index == 0
    assert new.entries[new.index].participant_id == o.entries[1].participant_id


def test_remove_participant_at_current_keeps_index() -> None:
    order = _make_order()
    current_id = order.entries[order.index].participant_id
    new = remove_participant(order, current_id)
    # Index stays at 0; the actor that was at index 1 is now at index 0.
    assert new.index == 0
    assert new.entries[0].participant_id == order.entries[1].participant_id


def test_remove_participant_after_current_no_index_change() -> None:
    order = _make_order()
    last_id = order.entries[-1].participant_id
    new = remove_participant(order, last_id)
    assert new.index == 0
    assert all(e.participant_id != last_id for e in new.entries)


def test_remove_only_remaining_participant_returns_empty_order() -> None:
    order = _make_order()
    o = order
    for entry in order.entries[:-1]:
        o = remove_participant(o, entry.participant_id)
    assert len(o.entries) == 1
    o = remove_participant(o, o.entries[0].participant_id)
    assert o.entries == []


def test_remove_unknown_participant_raises() -> None:
    order = _make_order()
    with pytest.raises(KeyError):
        remove_participant(order, "nope")


def test_remove_last_participant_clamps_index() -> None:
    order = _make_order()
    # Advance to last entry
    o = order
    for _ in range(len(order.entries) - 1):
        o = advance_turn(o)
    # Remove that last entry while it's the current one
    last_id = o.entries[o.index].participant_id
    new = remove_participant(o, last_id)
    assert new.index == len(new.entries) - 1


def test_find_returns_entry_or_none() -> None:
    order = _make_order()
    assert find(order, "nope") is None
    target_id = order.entries[1].participant_id
    found = find(order, target_id)
    assert found is not None
    assert found.participant_id == target_id


def test_helpers_raise_on_empty_order() -> None:
    empty = InitiativeOrder(entries=[])
    with pytest.raises(IndexError):
        current_turn(empty)
    with pytest.raises(IndexError):
        advance_turn(empty)
    with pytest.raises(IndexError):
        is_end_of_round(empty)
