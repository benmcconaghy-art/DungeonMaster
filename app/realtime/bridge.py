"""Bridge between orchestrator :class:`~app.orchestrator.dm.DmEvent`
variants and :class:`~app.realtime.messages.ServerMessage` variants.

The DM turn loop yields ``DmEvent`` variants tagged for the table —
narration, dice rolls, state mutations, whispers, errors. The WS
fan-out wants the same payloads as ``ServerMessage`` variants on the
spec §9 wire shape. This module owns the 1:1 mapping so the
orchestrator never has to know about the WS wire and the WS hub never
has to know about ``DmEvent`` internals.

``ToolDispatched`` is intentionally not mapped — it's the orchestrator's
generic audit envelope and the convenience variants
(``DiceRollEvent``, ``StateUpdate``, ``WhisperEvent``) are the ones
the table renders. The SSE bridge already follows the same convention.

Returning ``None`` for unmapped variants keeps callers simple: a single
``if ws_msg is not None: await pubsub.publish(...)`` covers every case
without per-variant branching at the call site.
"""

from __future__ import annotations

from app.orchestrator.dm import (
    DiceRollEvent,
    DmError,
    DmEvent,
    NarrationChunk,
    NarrationComplete,
    StateUpdate,
    ToolDispatched,
    WhisperEvent,
)
from app.realtime import messages as ws


def orchestrator_event_to_ws(event: DmEvent) -> ws.ServerMessage | None:
    """Translate one orchestrator event into its WS server-message
    counterpart, or ``None`` for events that don't surface to clients.

    Mapping:

    * :class:`NarrationChunk` → :class:`ws.NarrationChunk`
    * :class:`NarrationComplete` → :class:`ws.NarrationComplete`
    * :class:`DiceRollEvent` → :class:`ws.DiceRoll`
    * :class:`StateUpdate` → :class:`ws.StateUpdate`
    * :class:`WhisperEvent` → :class:`ws.Whisper`
    * :class:`DmError` → :class:`ws.DmError`
    * :class:`ToolDispatched` → ``None`` (audit envelope; the convenience
      variants carry the player-visible information).
    """

    if isinstance(event, NarrationChunk):
        return ws.NarrationChunk(content=event.content)
    if isinstance(event, NarrationComplete):
        return ws.NarrationComplete(message_id=event.message_id, content=event.content)
    if isinstance(event, DiceRollEvent):
        return ws.DiceRoll(
            tool_call_id=event.tool_call_id,
            expression=event.expression,
            total=event.total,
            individual=list(event.individual),
            purpose=event.purpose,
            target=event.target,
        )
    if isinstance(event, StateUpdate):
        return ws.StateUpdate(
            tool_call_id=event.tool_call_id,
            side_effects=dict(event.side_effects),
        )
    if isinstance(event, WhisperEvent):
        return ws.Whisper(
            tool_call_id=event.tool_call_id,
            audience=list(event.audience),
            content=event.content,
        )
    if isinstance(event, DmError):
        return ws.DmError(reason=event.reason, message=event.message)
    # ToolDispatched is the remaining variant; it doesn't surface to
    # clients (the convenience variants — DiceRollEvent, StateUpdate,
    # WhisperEvent — carry the player-visible payload). The match
    # ladder above is exhaustive over DmEvent today; mypy flagging this
    # as unreachable would mean the union grew without a corresponding
    # branch.
    assert isinstance(event, ToolDispatched)
    return None


__all__ = ["orchestrator_event_to_ws"]
