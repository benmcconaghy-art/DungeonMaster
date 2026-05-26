"""The DM turn loop.

Receives a player action, builds the layered prompt
(``app/llm/prompts.py``), streams the LLM response, parses tool
calls, dispatches each through the appropriate handler in
``app/orchestrator/handlers/``, persists the narration + tool
outcomes atomically, and yields events as they occur for the SSE /
WebSocket bridge.

Critical invariant (AGENTS.md #2): never hold a write transaction
across the streaming call. Persist input in a tight transaction;
release; stream; reopen for each tool dispatch and for the final
narration.

The function is an async generator yielding :class:`DmEvent`
discriminated-union variants; the SSE bridge serialises each variant
to its WS message type.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app import metrics
from app.db.models import Character, SessionMessage
from app.db.session import SessionLocal
from app.llm.client import ReasoningMode, RunawayTokenError, get_dm_client
from app.llm.memory import (
    extract_and_persist_facts,
    maybe_regenerate_session_summary,
)
from app.llm.prompts import build_dm_prompt
from app.llm.tools import ToolResult, get_handler, parse_tool_args, tool_definitions

# Importing the handlers package triggers @register(...) decorators on
# every per-tool module. Without this import the dispatch table is
# empty. Keep it even if it looks unused.
from app.orchestrator import handlers as _handlers  # noqa: F401
from app.orchestrator.context import DispatchContext, with_dispatch_context

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event variants the orchestrator emits
# ---------------------------------------------------------------------------


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True)


class NarrationChunk(_Event):
    """One streamed narration fragment.

    ``stream_id`` is minted fresh at the top of each
    :data:`_MAX_TOOL_ITERATIONS` body — i.e. once per "the DM continues
    speaking" moment within a single turn. Mid-tool-dispatch
    interruptions therefore start a new ``stream_id``, which the
    client renders as a discrete bubble. Without this, post-tool
    chunks fold back into the original bubble and the player sees
    one merged narration; with it, each iteration is its own beat.
    The Bug 1 (Phase 6.8) fix relies on this being per-iteration,
    not per-DM-turn.
    """

    type: Literal["narration_chunk"] = "narration_chunk"
    stream_id: str
    content: str


class NarrationComplete(_Event):
    """End-of-narration marker carrying the full assistant text and the
    persisted message id.

    Carries the ``stream_id`` of the *final* iteration so the client
    can finalise that bubble (replace its content with the canonical
    full string). Earlier-iteration partials remain as their own
    settled bubbles.
    """

    type: Literal["narration_complete"] = "narration_complete"
    stream_id: str
    message_id: str
    content: str


class ToolDispatched(_Event):
    """Emitted after a tool handler returns. ``side_effects`` is the
    handler's structured record (HP delta, dice values, etc.)."""

    type: Literal["tool_dispatched"] = "tool_dispatched"
    tool_name: str
    tool_call_id: str
    content: str
    side_effects: dict[str, Any]


class DiceRollEvent(_Event):
    """Convenience event for ``request_dice_roll`` outcomes — the SSE
    bridge highlights these so players see the d20 pop up. Emitted in
    addition to ``tool_dispatched``."""

    type: Literal["dice_roll"] = "dice_roll"
    tool_call_id: str
    expression: str
    total: int
    individual: list[int]
    purpose: str
    target: dict[str, Any] | None = None


class StateUpdate(_Event):
    """Emitted when a tool mutates persistent state (HP, location, etc.).
    The bridge uses this to refresh client-side character cards and
    location panels."""

    type: Literal["state_update"] = "state_update"
    tool_call_id: str
    side_effects: dict[str, Any]


class WhisperEvent(_Event):
    """A DM whisper. The bridge routes this only to the addressed
    audience; other clients never see it."""

    type: Literal["whisper"] = "whisper"
    tool_call_id: str
    audience: list[str]
    content: str


class NpcIntroducedEvent(_Event):
    """An NPC was spawned into the campaign. The rail panel uses this to
    add a card immediately; a later ``image_ready`` supplies the portrait."""

    type: Literal["npc_introduced"] = "npc_introduced"
    npc_id: str
    name: str
    brief: str
    portrait_image_id: str | None = None


class DmError(_Event):
    """Something went wrong. ``reason`` is short and machine-readable;
    ``message`` is human-readable."""

    type: Literal["dm_error"] = "dm_error"
    reason: str
    message: str


DmEvent = (
    NarrationChunk
    | NarrationComplete
    | ToolDispatched
    | DiceRollEvent
    | StateUpdate
    | WhisperEvent
    | NpcIntroducedEvent
    | DmError
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bound the tool-call ping-pong so a misbehaving model can't loop forever.
# A single combat round legitimately spends multiple tool calls.
#
# History:
#   - Spec §7 originally proposed 5; that proved too tight before any
#     Nemotron traffic was observed.
#   - Phase 2 raised it to 10 because real combat turns chained 6-10
#     calls when Nemotron drove both sides of a round end-to-end.
#   - Phase 4 prep #1 added pacing language to the system prompt and
#     re-evaluated the cap. Cap=7 tripped iteration_cap (real combat
#     legitimately uses ~6-8 calls); cap=8 tripped empty_completion
#     mid-round in one run (Nemotron's run-to-run variance at the
#     0.85 stream temperature). Holding 10 as the safety net so it
#     fires only on genuine runaways, not on the legitimate upper
#     end of normal-round variance. The pacing prompt is the real
#     prep-#1 deliverable; the cap stays as the canary it was.
#   - Phase 8.2 playthrough: a 3-cultist fight triggered 10 sequential
#     tool calls (3 rolls + 4 apply_damage + start_encounter +
#     end_encounter) with no narration between them, wedging the turn.
#     Root cause was batching (combat-as-computation rather than
#     beat-by-beat theatre); the prompt was tightened. Cap raised to 20
#     as a wider safety net — with correct pacing (1-4 tools per beat)
#     multi-round fights with many combatants need headroom over 10.
_MAX_TOOL_ITERATIONS = 20

# Defensive JSON-block fallback parser: matches a fenced ``json`` block.
# We match the entire body greedily up to the closing fence, then let
# json.loads validate. Greedy is fine here — qwen3_coder produces at
# most one fenced block per missed parse, and if the model happens to
# narrate a json snippet for flavour the fallback validator below
# rejects payloads that don't include ``name`` + ``arguments``.
_JSON_FENCE_RE = re.compile(r"```json\s*(?P<body>.+?)\s*```", re.DOTALL)


# System note appended to ``messages`` when the orchestrator rejects one
# or more tool calls in an iteration. The wording is deliberate:
#
#   * honest about what happened ("rejected as malformed and discarded")
#     so the model doesn't try to narrate a phantom outcome;
#   * instructive about recovery ("describe what you intended in
#     narration or try again") so the next iteration has a clear path;
#   * neutral in tone — no apology framing that would leak into the
#     model's narration as in-character contrition.
#
# Frozen as a module constant so future edits to the recovery message
# are deliberate and so tests can assert against the exact string.
# Critical Invariant: this note replaces the rejected tool call in
# prompt history; the malformed assistant message itself MUST NOT be
# appended (vLLM re-parses ``tool_calls[].function.arguments`` when
# rendering the conversation, and an invalid string there wedges every
# subsequent turn with HTTP 400).
_TOOL_REJECTION_RECOVERY_NOTE = (
    "[engine: an attempted tool call was rejected as malformed and discarded;"
    " describe what you intended in narration or try again]"
)


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


class _AccumulatedToolCall:
    """Mutable accumulator for one streamed tool call.

    OpenAI streams tool calls in *fragments* — each chunk's
    ``delta.tool_calls`` is a list keyed by ``index``; ``id`` and
    ``type`` arrive once on the first fragment, ``function.name`` and
    ``function.arguments`` accumulate incrementally. This class owns
    the accumulation; the orchestrator constructs one per index.
    """

    __slots__ = ("arguments", "id", "index", "name", "type")

    def __init__(self, index: int) -> None:
        self.index = index
        self.id: str | None = None
        self.type: str | None = None
        self.name: str = ""
        self.arguments: str = ""

    def merge(self, fragment: Any) -> None:
        """Fold one streamed fragment into the accumulator."""

        if getattr(fragment, "id", None):
            self.id = fragment.id
        if getattr(fragment, "type", None):
            self.type = fragment.type
        function = getattr(fragment, "function", None)
        if function is not None:
            name_part = getattr(function, "name", None)
            if name_part:
                self.name = (self.name + name_part) if self.name else name_part
            args_part = getattr(function, "arguments", None)
            if args_part:
                self.arguments += args_part

    def is_complete(self) -> bool:
        return bool(self.name) and self.arguments != ""

    def parsed_arguments(self) -> dict[str, Any]:
        if not self.arguments:
            return {}
        parsed: object = json.loads(self.arguments)
        if not isinstance(parsed, dict):
            raise ValueError(f"tool-call arguments did not parse to an object: {self.arguments!r}")
        return parsed


# ---------------------------------------------------------------------------
# JSON-block fallback parser
# ---------------------------------------------------------------------------


def _classify_tool_call(tc: _AccumulatedToolCall) -> tuple[str, str] | None:
    """Pre-flight gate: can this tool call be safely honoured?

    Returns ``None`` if the call parses as JSON, validates against the
    registered Pydantic schema, and has a registered handler. Returns
    ``(reason, message)`` if any of those gates fails.

    Rejected calls must NOT propagate to subsequent prompt history. The
    OpenAI assistant-message shape requires ``tool_calls[].function.
    arguments`` to be a JSON-encoded string; vLLM re-renders the
    conversation through Nemotron's chat template before sending the
    next request, and a malformed ``arguments`` string there trips an
    HTTP 400 ("Expecting property name…") that wedges every subsequent
    turn in the session. This is the Phase 6.9 close-out fix's
    structural gate: classify up-front, then assemble the assistant
    audit from the surviving subset only.
    """

    if not tc.is_complete():
        return (
            "incomplete_tool_call",
            f"tool-call fragment never finished assembling: name={tc.name!r}",
        )
    try:
        raw_args = tc.parsed_arguments()
    except ValueError as exc:
        return ("invalid_tool_args", f"invalid arguments JSON: {exc}")
    try:
        parse_tool_args(tc.name, raw_args)
    except KeyError as exc:
        return ("unknown_tool", f"unknown tool: {exc}")
    except Exception as exc:
        return ("invalid_tool_args", f"tool args failed validation: {exc}")
    if get_handler(tc.name) is None:
        return (
            "not_implemented",
            f"tool {tc.name!r} is declared but has no Phase 2 handler;"
            " try a different approach.",
        )
    return None


def _safe_arguments_string(raw: str) -> str:
    """Coerce a tool-call ``arguments`` string to a JSON-valid object.

    Defence-in-depth: :func:`_classify_tool_call` rejects malformed
    calls before the assistant audit message is built, so in normal
    operation ``raw`` is already a JSON object string. This helper is
    regression insurance — if a future code path ever lets a malformed
    string slip through to :func:`_assistant_message_for_audit`, the
    assistant audit message stays vLLM-valid by substituting ``"{}"``.
    """

    if not raw:
        return "{}"
    try:
        parsed = json.loads(raw)
    except ValueError:
        return "{}"
    if not isinstance(parsed, dict):
        return "{}"
    return raw


def _extract_fallback_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    """Pull a ``{"name": ..., "arguments": ...}`` JSON-block from content.

    Returns ``(name, arguments)`` if a parseable block is found,
    ``None`` otherwise. Defensive — this fires only when the model's
    native ``tool_calls`` came back empty but it stuffed a fenced
    block into ``content``. We log every fallback hit so operators
    can track how often the qwen3_coder parser misses.
    """

    match = _JSON_FENCE_RE.search(content)
    if match is None:
        return None
    body = match.group("body")
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    log.warning("dm.py: JSON-block fallback parser matched a tool call (name=%s)", name)
    return name, arguments


# ---------------------------------------------------------------------------
# The turn loop
# ---------------------------------------------------------------------------


async def take_turn(
    db: AsyncSession,
    *,
    session_id: str,
    sender_user_id: str,
    sender_character_id: str | None,
    content: str,
    opening: bool = False,
) -> AsyncIterator[DmEvent]:
    """Execute one turn for ``content`` from ``sender_user_id``.

    The yielded events are intended for the SSE / WS bridge to relay
    verbatim; ``narration_chunk`` events arrive in stream order, then
    ``tool_dispatched`` (and convenience variants) for any tool calls,
    then ``narration_complete`` when the assistant's final message has
    been persisted.

    ``opening`` switches the turn into auto-greeting mode:

    * The leading message persisted to ``session_messages`` is
      ``sender_kind='system'`` rather than ``'player'``, so future
      prompts surface it as a system note (via
      :func:`_recent_turns_to_messages`) and not as a player utterance
      the DM should "respond to". ``content`` is the bootstrapping
      directive (the caller injects something like
      "[Session begins — set the opening scene given the campaign,
      location, and party]"); the player never sees it as their own
      input.
    * Everything else — prompt build, tool loop, persistence of the
      DM message, post-turn memory work — is unchanged. The opening
      DM message lands in ``session_messages`` exactly like a normal
      assistant turn so a reconnecting client picks it up via the
      snapshot path.

    Discipline:

    1. Persist the leading message in a tight transaction. Commit.
    2. Build the prompt (read-only DB activity, no transaction held).
    3. Loop up to :data:`_MAX_TOOL_ITERATIONS`:
       a. Stream a completion (no transaction held).
       b. If tool calls: dispatch each in its own tight transaction;
          append a ``tool`` message to the running list; loop.
       c. Otherwise: persist the assistant turn in a tight transaction
          and break.
    """

    # ------- 1. Persist the leading message ----------------------------------
    # Player turn → sender_kind='player'; opening turn → sender_kind='system'
    # so the prompt builder surfaces it as a neutral system note rather than
    # tail-position user input the DM is expected to respond to. The DM
    # treats the bootstrapping directive as engine-level instruction.
    leading_kind = "system" if opening else "player"
    leading_sender_id = None if opening else sender_character_id
    async with db.begin():
        player_msg = SessionMessage(
            session_id=session_id,
            sender_kind=leading_kind,
            sender_id=leading_sender_id,
            audience=[],
            content=content,
        )
        db.add(player_msg)

    # ------- 2. Build the prompt (read-only) ---------------------------------
    messages = await build_dm_prompt(db, session_id=session_id)
    # Close any autobegun read transaction before we hand off to the
    # streaming loop. AGENTS.md invariant #2: NO transaction held while
    # streaming. SQLAlchemy autobegins on the first SELECT in
    # build_dm_prompt; without this commit, the read txn would hang
    # open across the network call.
    await db.commit()

    # ------- 3. Tool-call loop -----------------------------------------------
    dispatch_ctx = DispatchContext(
        session_id=session_id,
        sender_user_id=sender_user_id,
        sender_character_id=sender_character_id,
    )
    client = get_dm_client()
    # Hide tools that have no registered handler so the model can't loop
    # on "not_implemented" responses. Exposing only the working subset
    # also keeps the system-prompt token budget tight. As later phases
    # land their handlers, those tools become visible automatically.
    tools = tool_definitions(only_implemented=True)
    final_assistant_text = ""
    final_tool_calls_audit: list[dict[str, Any]] = []
    iteration = 0
    # Track the most recent iteration's stream_id so the trailing
    # NarrationComplete pairs with the correct bubble on the client.
    final_stream_id: str = ""
    # Two-tier empty-completion recovery (Phase 6.11). Counts consecutive
    # empty completions within this turn. First empty → retry at low
    # reasoning + doubled token budget. Second empty → dm_error fallback.
    # Independent of ``iteration``: one turn can have many tool hops before
    # encountering an empty completion; the two counters track different
    # things. Resets per-turn automatically (local var; take_turn is fresh
    # each call).
    empty_completion_count = 0

    while iteration < _MAX_TOOL_ITERATIONS:
        iteration += 1
        # One stream_id per iteration. See ``NarrationChunk`` docstring:
        # post-tool continuations are conceptually a new "the DM
        # continues speaking" beat and the client renders them as a
        # discrete bubble.
        stream_id = uuid.uuid4().hex
        final_stream_id = stream_id
        try:
            # reasoning_mode: "full" on ordinary iterations — tool-call
            # accuracy depends on the full reasoning trace. Switched to
            # "low" on the first empty-completion retry so reduced-effort
            # reasoning is sufficient for a recovery narration (the model
            # already has tool results in context).
            # max_tokens=2048 on both tiers. With Nemotron reasoning_mode=full,
            # max_tokens caps the answer phase only (thinking tokens are
            # separately accounted at the server). 2048 gives real room for
            # complex scenes; length discipline lives in the PACING prompt
            # block, not here. See Critical Invariant #19.
            _reasoning_mode: ReasoningMode = "low" if empty_completion_count > 0 else "full"
            _max_tokens = 2048
            stream = await client.stream_dm(
                messages,
                tools=tools,
                tool_choice="auto",
                reasoning_mode=_reasoning_mode,
                max_tokens=_max_tokens,
            )
        except Exception as exc:
            log.exception("dm.py: stream_dm() failed before first chunk")
            yield DmError(
                reason="stream_failed",
                message=f"LLM stream could not be opened: {exc}",
            )
            return

        accumulated_content = ""
        accumulated_tool_calls: dict[int, _AccumulatedToolCall] = {}

        try:
            async for chunk in stream:
                # The runaway detector raises if it trips; catch is below.
                fragment = _content_of(chunk)
                if fragment:
                    yield NarrationChunk(stream_id=stream_id, content=fragment)
                    accumulated_content += fragment
                _accumulate_tool_calls(chunk, accumulated_tool_calls)
        except RunawayTokenError as exc:
            log.warning("dm.py: runaway-token detector tripped: %s", exc)
            yield DmError(
                reason="runaway_token",
                message=str(exc),
            )
            return
        except Exception as exc:
            log.exception("dm.py: stream consumption failed mid-stream")
            yield DmError(
                reason="stream_error",
                message=f"LLM stream broke: {exc}",
            )
            return

        # If the model emitted no native tool calls, try the JSON-block
        # fallback parser. (Spec §7 watch-item: qwen3_coder occasionally
        # misses native parsing on long inputs.)
        if not accumulated_tool_calls and accumulated_content.strip():
            fallback = _extract_fallback_tool_call(accumulated_content)
            if fallback is not None:
                name, arguments = fallback
                fake = _AccumulatedToolCall(index=0)
                fake.id = f"fallback-{iteration}"
                fake.type = "function"
                fake.name = name
                fake.arguments = json.dumps(arguments)
                accumulated_tool_calls[0] = fake
                # Strip the JSON block from the visible content; the
                # narration the player sees is whatever framing the
                # model put around it.
                accumulated_content = _JSON_FENCE_RE.sub("", accumulated_content).strip()

        if accumulated_tool_calls:
            # Classify each tool call up-front (Phase 6.9 close-out fix).
            # Honourable calls — parse cleanly, validate against the
            # schema, have a registered handler — dispatch normally and
            # land in the assistant audit message. Rejected calls are
            # dropped from prompt history because their malformed
            # ``arguments`` string would re-poison the next vLLM request
            # and wedge the session.
            honourable_calls: dict[int, _AccumulatedToolCall] = {}
            rejection_events: list[DmEvent] = []
            for idx, tc in accumulated_tool_calls.items():
                rejection = _classify_tool_call(tc)
                if rejection is None:
                    honourable_calls[idx] = tc
                else:
                    reason, message = rejection
                    log.warning(
                        "dm.py: rejecting tool call (reason=%s, name=%s, message=%s)",
                        reason,
                        tc.name,
                        message,
                    )
                    rejection_events.append(DmError(reason=reason, message=message))

            if honourable_calls:
                # Build the assistant audit message from honourable
                # calls only — by construction the resulting message is
                # a vLLM-valid OpenAI shape.
                assistant_audit = _assistant_message_for_audit(
                    accumulated_content, honourable_calls
                )
                messages.append(assistant_audit)
                for tc in sorted(honourable_calls.values(), key=lambda t: t.index):
                    async for event, audit_msg in _dispatch_one(db, dispatch_ctx, tc):
                        if audit_msg is not None:
                            messages.append(audit_msg)
                            final_tool_calls_audit.append(_tool_call_audit_record(tc))
                        if event is not None:
                            yield event
            elif accumulated_content.strip():
                # No tool calls survived classification but the model
                # did narrate. Preserve the prose (without any
                # ``tool_calls`` field) so the next iteration's prompt
                # still carries the model's intent.
                messages.append({"role": "assistant", "content": accumulated_content})

            # Surface rejection events to the WS layer (the player sees
            # them as in-log [ERROR] entries) and append a single
            # sanitised system note so the next iteration's model has a
            # clean recovery signal without seeing the malformed call.
            for event in rejection_events:
                yield event
            if rejection_events:
                messages.append({"role": "system", "content": _TOOL_REJECTION_RECOVERY_NOTE})

            # Loop back for another stream; the model now sees the
            # honourable tool results (and any recovery note) in history.
            continue

        # No tool calls — this is the final narration.
        if not accumulated_content.strip():
            empty_completion_count += 1
            if empty_completion_count < 2:
                log.warning(
                    "dm.py: empty completion (iteration %d); retrying with low reasoning",
                    iteration,
                )
                continue
            log.warning(
                "dm.py: consecutive empty completions (iteration %d); aborting turn",
                iteration,
            )
            yield DmError(
                reason="empty_completion",
                message="DM produced an empty response. Try again.",
            )
            return

        empty_completion_count = 0
        final_assistant_text = accumulated_content
        break
    else:
        log.warning("dm.py: exceeded tool-call iteration cap (%d)", _MAX_TOOL_ITERATIONS)
        yield DmError(
            reason="iteration_cap",
            message=(
                f"DM looped tool calls past the safety cap ({_MAX_TOOL_ITERATIONS})."
                " Aborting this turn — the partial state changes are persisted."
            ),
        )
        return

    # ------- 4. Persist the final assistant message --------------------------
    async with db.begin():
        dm_msg = SessionMessage(
            session_id=session_id,
            sender_kind="dm",
            sender_id=None,
            audience=[],
            content=final_assistant_text,
            tool_calls=final_tool_calls_audit or None,
        )
        db.add(dm_msg)

    yield NarrationComplete(
        stream_id=final_stream_id,
        message_id=dm_msg.id,
        content=final_assistant_text,
    )

    # ------- 5. Schedule fire-and-forget post-turn memory work ---------------
    # The fact extractor and the session-summary regeneration both run
    # async after the turn so they never block the next player action.
    # Each task opens its OWN database session — we don't pass ``db`` in
    # because the caller (SSE bridge) closes it as soon as this generator
    # returns.
    #
    # Speaker attribution (AGENTS.md invariant #17): the fact extractor
    # sees only ``player_action`` / ``dm_response`` strings, not the
    # raw SessionMessage row, so we attach the same ``[Name, Class]:``
    # prefix the prompt builder uses. Otherwise facts extracted from a
    # multi-PC turn ("I ask the bartender for rumours") get attributed
    # to the wrong character. Opening turns have no player speaker.
    attributed_action = content
    if not opening and sender_character_id is not None:
        speaker = await db.get(Character, sender_character_id)
        if speaker is not None:
            attributed_action = f"[{speaker.name}, {speaker.class_name}]: {content}"
    _schedule_post_turn_memory(
        session_id=session_id,
        player_action=attributed_action,
        dm_response=final_assistant_text,
    )


# ---------------------------------------------------------------------------
# Post-turn background tasks
# ---------------------------------------------------------------------------


# Module-level set keeps strong refs to in-flight tasks. Without this,
# asyncio's create_task can drop them mid-flight when the only reference
# is on a local variable that goes out of scope as ``take_turn`` returns.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _schedule_post_turn_memory(
    *,
    session_id: str,
    player_action: str,
    dm_response: str,
) -> None:
    """Fire-and-forget the fact extractor and the (possibly-no-op)
    session-summary regeneration."""

    extractor_task = asyncio.create_task(
        _run_fact_extractor(session_id, player_action, dm_response)
    )
    summary_task = asyncio.create_task(_run_session_summary(session_id))
    for task in (extractor_task, summary_task):
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)


async def _run_fact_extractor(session_id: str, player_action: str, dm_response: str) -> None:
    """Open a fresh session and run the fact extractor.

    Errors are swallowed (logged only) — a failure here must not surface
    to the player. ``extract_and_persist_facts`` itself is defensive
    about LLM and JSON failures, so anything escaping is a programming
    bug we want to log loudly without breaking the next turn.
    """

    try:
        async with SessionLocal() as db:
            await extract_and_persist_facts(
                db,
                session_id=session_id,
                player_action=player_action,
                dm_response=dm_response,
            )
    except Exception:
        log.exception("post-turn fact extractor crashed (session=%s)", session_id)


async def _run_session_summary(session_id: str) -> None:
    """Open a fresh session and call the (no-op-most-of-the-time)
    summary regenerator. The function only invokes the LLM when the
    player-message count is a non-zero multiple of every_n_turns, so
    calling it after every turn is cheap by design."""

    try:
        async with SessionLocal() as db:
            await maybe_regenerate_session_summary(db, session_id=session_id)
    except Exception:
        log.exception("post-turn session-summary regen crashed (session=%s)", session_id)


# ---------------------------------------------------------------------------
# Stream-consumption helpers
# ---------------------------------------------------------------------------


def _content_of(chunk: ChatCompletionChunk) -> str:
    """Return the ``delta.content`` string from one chunk, or empty."""

    if not chunk.choices:
        return ""
    content = getattr(chunk.choices[0].delta, "content", None)
    return content or ""


def _accumulate_tool_calls(
    chunk: ChatCompletionChunk,
    accumulated: dict[int, _AccumulatedToolCall],
) -> None:
    """Fold this chunk's ``tool_calls`` deltas into the accumulator."""

    if not chunk.choices:
        return
    fragments = getattr(chunk.choices[0].delta, "tool_calls", None) or []
    for fragment in fragments:
        idx = fragment.index
        slot = accumulated.get(idx)
        if slot is None:
            slot = _AccumulatedToolCall(index=idx)
            accumulated[idx] = slot
        slot.merge(fragment)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def _dispatch_one(
    db: AsyncSession,
    dispatch_ctx: DispatchContext,
    tc: _AccumulatedToolCall,
) -> AsyncIterator[tuple[DmEvent | None, dict[str, Any] | None]]:
    """Dispatch one accumulated tool call.

    Yields ``(event, audit_message)`` tuples. The orchestrator appends
    every audit message to the running prompt list (so the next stream
    sees the tool result) and yields each event to the caller. Any
    failure inside the dispatcher converts to a ``dm_error`` event
    plus a synthetic tool message reporting the failure (so the LLM
    can see and recover instead of looping into the void).
    """

    if not tc.is_complete():
        yield (
            DmError(
                reason="incomplete_tool_call",
                message=f"tool-call fragment never finished assembling: name={tc.name!r}",
            ),
            None,
        )
        return

    # Parse arguments JSON.
    try:
        raw_args = tc.parsed_arguments()
    except ValueError as exc:
        log.warning("dm.py: tool-call args invalid JSON: %s", exc)
        yield (
            DmError(reason="invalid_tool_args", message=str(exc)),
            _tool_audit_message(tc, f"invalid arguments JSON: {exc}"),
        )
        return

    # Validate against the registered Pydantic schema.
    try:
        args = parse_tool_args(tc.name, raw_args)
    except KeyError as exc:
        msg = f"unknown tool: {exc}"
        yield (
            DmError(reason="unknown_tool", message=msg),
            _tool_audit_message(tc, msg),
        )
        return
    except Exception as exc:
        msg = f"tool args failed validation: {exc}"
        yield (
            DmError(reason="invalid_tool_args", message=msg),
            _tool_audit_message(tc, msg),
        )
        return

    # Resolve the handler.
    handler = get_handler(tc.name)
    if handler is None:
        msg = (
            f"tool {tc.name!r} is declared but has no Phase 2 handler;" " try a different approach."
        )
        yield (
            DmError(reason="not_implemented", message=msg),
            _tool_audit_message(tc, msg),
        )
        return

    # Run the handler in its own tight transaction (AGENTS.md #2).
    try:
        async with db.begin():
            with with_dispatch_context(dispatch_ctx):
                result: ToolResult = await handler.fn(db, args)
    except Exception as exc:
        log.exception("dm.py: handler %s raised", tc.name)
        # Phase 7 structured-logging + metrics contract: one record /
        # one counter per tool dispatch with tool_name + outcome. The
        # exception message is already in the log via exc_info; the
        # extras let metrics + alerting key off ``tool_name`` cleanly.
        log.warning(
            "dm tool dispatch failed",
            extra={"tool_name": tc.name, "outcome": "error"},
        )
        metrics.dm_tool_dispatch_total.labels(tool_name=tc.name, outcome="error").inc()
        msg = f"handler raised: {exc}"
        yield (
            DmError(reason="handler_error", message=msg),
            _tool_audit_message(tc, msg),
        )
        return

    log.info(
        "dm tool dispatch ok",
        extra={"tool_name": tc.name, "outcome": "ok"},
    )
    metrics.dm_tool_dispatch_total.labels(tool_name=tc.name, outcome="ok").inc()

    # Convenience events for the bridge (in addition to the generic
    # tool_dispatched).
    yield (
        ToolDispatched(
            tool_name=tc.name,
            tool_call_id=tc.id or f"local-{tc.index}",
            content=result.content,
            side_effects=result.side_effects,
        ),
        None,
    )

    side = result.side_effects or {}
    kind = side.get("kind")
    if kind == "dice_roll":
        yield (
            DiceRollEvent(
                tool_call_id=tc.id or f"local-{tc.index}",
                expression=str(side.get("expression", "")),
                total=int(side.get("total", 0)),
                individual=list(side.get("individual", [])),
                purpose=str(side.get("purpose", "")),
                target=side.get("target"),
            ),
            None,
        )
    elif kind == "state_update":
        yield (
            StateUpdate(
                tool_call_id=tc.id or f"local-{tc.index}",
                side_effects=side,
            ),
            None,
        )
    elif kind == "whisper":
        yield (
            WhisperEvent(
                tool_call_id=tc.id or f"local-{tc.index}",
                audience=list(side.get("audience", [])),
                content=str(side.get("content", "")),
            ),
            None,
        )
    elif kind == "npc_spawned":
        pid = side.get("portrait_image_id")
        yield (
            NpcIntroducedEvent(
                npc_id=str(side.get("npc_id", "")),
                name=str(side.get("name", "")),
                brief=str(side.get("brief", "")),
                portrait_image_id=str(pid) if pid else None,
            ),
            None,
        )

    # Tool message that goes back into the prompt for the next stream.
    yield (None, _tool_audit_message(tc, result.content))


# ---------------------------------------------------------------------------
# Audit-message helpers
# ---------------------------------------------------------------------------


def _tool_audit_message(tc: _AccumulatedToolCall, content: str) -> dict[str, Any]:
    """Build the OpenAI ``tool`` message that goes back into the prompt.

    Each tool message is keyed by ``tool_call_id`` so the model can pair
    its outgoing request with the response.
    """

    return {
        "role": "tool",
        "tool_call_id": tc.id or f"local-{tc.index}",
        "name": tc.name,
        "content": content,
    }


def _assistant_message_for_audit(
    content: str,
    accumulated: dict[int, _AccumulatedToolCall],
) -> dict[str, Any]:
    """Reconstruct the assistant message that requested the tool calls.

    OpenAI requires an assistant message with non-null ``tool_calls``
    immediately preceding the ``tool`` messages — without it, sending
    the next stream raises ``messages.X.role: ...``. Build the
    canonical shape here.

    The orchestrator's :func:`_classify_tool_call` gate filters out any
    call whose ``arguments`` aren't valid JSON before this builder is
    invoked, so in normal operation the embedded strings are already
    safe. :func:`_safe_arguments_string` is regression insurance: if a
    future code path lets a malformed string slip through, we substitute
    ``"{}"`` rather than embed a vLLM-poisoning value.
    """

    tool_calls = []
    for tc in sorted(accumulated.values(), key=lambda t: t.index):
        tool_calls.append(
            {
                "id": tc.id or f"local-{tc.index}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": _safe_arguments_string(tc.arguments),
                },
            }
        )
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": tool_calls,
    }


def _tool_call_audit_record(tc: _AccumulatedToolCall) -> dict[str, Any]:
    """Compact JSON-friendly record of a dispatched tool call.

    Persisted into ``session_messages.tool_calls`` so the audit trail
    survives independently of the OpenAI message format.
    """

    return {
        "id": tc.id or f"local-{tc.index}",
        "name": tc.name,
        "arguments": tc.arguments,
    }


__all__ = [
    "DiceRollEvent",
    "DmError",
    "DmEvent",
    "NarrationChunk",
    "NarrationComplete",
    "NpcIntroducedEvent",
    "StateUpdate",
    "ToolDispatched",
    "WhisperEvent",
    "take_turn",
]
