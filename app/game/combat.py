"""Turn-order helpers for combat encounters.

The data structures live in :mod:`app.game.rules` (``InitiativeOrder``,
``InitiativeEntry``); this module is for the small library of pure
helpers that step through one — advance to the next turn, detect
end-of-round, find a participant, etc.

Everything here is pure: an :class:`InitiativeOrder` is frozen, so
``advance_turn`` returns a new instance rather than mutating in place.
The call site persists the new order back to ``encounters.initiative``.
"""

from __future__ import annotations

from dataclasses import replace

from app.game.rules import InitiativeEntry, InitiativeOrder


def current_turn(order: InitiativeOrder) -> InitiativeEntry:
    """Return the entry whose turn it currently is.

    Raises :class:`IndexError` if ``order.entries`` is empty (a programming
    error — an encounter with no participants shouldn't exist).
    """

    if not order.entries:
        raise IndexError("initiative order is empty")
    return order.entries[order.index]


def advance_turn(order: InitiativeOrder) -> InitiativeOrder:
    """Step to the next turn, wrapping to the next round at the end.

    Returns a new :class:`InitiativeOrder` with the updated ``index``
    and ``round_number``. The original is untouched.
    """

    if not order.entries:
        raise IndexError("initiative order is empty")
    next_index = order.index + 1
    if next_index >= len(order.entries):
        return replace(order, index=0, round_number=order.round_number + 1)
    return replace(order, index=next_index)


def is_end_of_round(order: InitiativeOrder) -> bool:
    """``True`` if the current entry is the last in the round.

    Useful for the call site to fire end-of-round effects (regen,
    durations, morale checks) before advancing into the next round.
    """

    if not order.entries:
        raise IndexError("initiative order is empty")
    return order.index == len(order.entries) - 1


def remove_participant(order: InitiativeOrder, participant_id: str) -> InitiativeOrder:
    """Drop a participant (e.g. killed) and clamp the index sensibly.

    If the removed participant came before the current turn, the index
    shifts down by one so the same actor is still 'up'. If the removed
    participant *is* the current turn, the index stays put — the next
    advance will fall on the entry that was after them. If the removed
    participant came after, no index adjustment is needed.
    """

    new_entries = [e for e in order.entries if e.participant_id != participant_id]
    if len(new_entries) == len(order.entries):
        raise KeyError(f"participant {participant_id!r} not in initiative order")

    removed_at = next(i for i, e in enumerate(order.entries) if e.participant_id == participant_id)
    if not new_entries:
        return InitiativeOrder(entries=[], round_number=order.round_number, index=0)
    if removed_at < order.index:
        new_index = order.index - 1
    elif removed_at == order.index:
        # Index stays where it was — that slot is now occupied by the
        # next participant. Clamp if we removed the very last entry.
        new_index = min(order.index, len(new_entries) - 1)
    else:
        new_index = order.index
    return replace(order, entries=new_entries, index=new_index)


def find(order: InitiativeOrder, participant_id: str) -> InitiativeEntry | None:
    """Return the matching entry, or ``None`` if no such id."""

    for e in order.entries:
        if e.participant_id == participant_id:
            return e
    return None
