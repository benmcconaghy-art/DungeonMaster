"""Per-dispatch context shared between the orchestrator and handlers.

The tool-handler signature in :mod:`app.llm.tools` is fixed at
``(db, args) -> ToolResult`` (so handlers stay easy to register and
test). But the handlers genuinely need a tiny bit more context — the
``session_id`` so they can write audit rows scoped to the active
session. Rather than thread that through every Pydantic args model
(which would bleed orchestration concerns into the LLM tool schema),
we use a context variable scoped to one dispatch.

The orchestrator wraps each handler call in
:func:`with_dispatch_context`; the handler reads the context via
:func:`current_context`. Not setting the context before dispatch
raises :class:`LookupError` — that's a programming error, not a
runtime condition, and it surfaces immediately.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DispatchContext:
    """Read-only context object for one tool dispatch.

    Attributes here are stable across the entire turn — the
    orchestrator sets the context once per take_turn and every handler
    invocation in that turn sees the same values.
    """

    session_id: str
    sender_user_id: str
    sender_character_id: str | None


_current: ContextVar[DispatchContext] = ContextVar("dm_dispatch_context")


@contextmanager
def with_dispatch_context(ctx: DispatchContext) -> Iterator[DispatchContext]:
    """Bind a context for the duration of the ``with`` block.

    Restores the previous context on exit (whether the block
    completed normally or raised). Nested binds are fine — they stack
    and unwind in LIFO order.
    """

    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


def current_context() -> DispatchContext:
    """Return the active :class:`DispatchContext`.

    Raises :class:`LookupError` if no context is bound — that means a
    handler was invoked outside the orchestrator's dispatch path,
    which is a bug.
    """

    try:
        return _current.get()
    except LookupError as exc:
        raise LookupError(
            "no DispatchContext bound; did the orchestrator forget to wrap "
            "the handler call in with_dispatch_context(...)?"
        ) from exc


__all__ = ["DispatchContext", "current_context", "with_dispatch_context"]
