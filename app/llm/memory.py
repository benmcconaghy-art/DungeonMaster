"""Long-term memory: summarisation, world-fact extraction, NumPy retrieval.

Four memory tiers per spec §7:

1. **Verbatim** — :func:`recent_turns` returns the last N
   ``SessionMessage`` rows. Default N=40 (Phase 3, after Phase 2's KV-
   cache headroom measurement on Pro 6000 confirmed we have room).

2. **Session summary** — :func:`maybe_regenerate_session_summary`. Every
   ``every_n_turns`` player actions, an LLM call compresses the recent
   play log into a 400-700 token third-person summary that lives on
   ``sessions.summary``. Per-session ``asyncio.Lock`` serialises
   concurrent calls; the second one returns the just-written summary
   without re-invoking the LLM.

3. **Campaign summary** — :func:`regenerate_campaign_summary`. Triggered
   explicitly at session-end (the API endpoint calls this after marking
   the session ended). Joins every session summary chronologically with
   the existing ``campaigns.long_summary`` and produces a longer-arc
   narrative that lives on ``campaigns.long_summary``.

4. **World facts (vector)** — :class:`WorldFactRetriever` does
   per-campaign brute-force cosine retrieval over the L2-normalised
   embeddings stored as BLOBs on ``world_facts``. The fact extractor
   :func:`extract_and_persist_facts` runs fire-and-forget after each DM
   turn and writes new rows.

Concurrency / transaction discipline (AGENTS.md invariant #2): every LLM
call in this module runs OUTSIDE any open write transaction. The
summarisers read inputs in a quick read transaction, release, call the
LLM, then reopen a tight transaction to persist the result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Campaign, SessionMessage, WorldFact
from app.db.models import Session as DmSession
from app.llm.client import DmClient, DmClientError, get_dm_client
from app.llm.embeddings import Embedder, get_embedder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# (a) Verbatim retrieval
# ---------------------------------------------------------------------------


async def recent_turns(
    db: AsyncSession,
    session_id: str,
    n: int = 40,
) -> list[SessionMessage]:
    """Return the last ``n`` ``SessionMessage`` rows in chronological order.

    Used by :func:`app.llm.prompts.build_dm_prompt` for the
    ``[RECENT TURNS]`` block. The ordering is "ascending by created_at"
    so the LLM sees the conversation in natural reading order; the
    ``LIMIT`` is applied to the descending tail and then reversed in
    Python.

    Default ``n=40`` (was 20 in Phase 2). The Phase 2 integration test
    confirmed Pro 6000 has 256k context with comfortable KV-cache
    headroom; spec §7's footnote already anticipated pushing N up.

    Phase 2 returns *every* message regardless of audience — Phase 5+
    multi-player support will filter whispers per-character. For the
    single-player Phase 2/3 loop the player sees everything the DM has
    said anyway.
    """

    if n <= 0:
        return []

    stmt = (
        select(SessionMessage)
        .where(SessionMessage.session_id == session_id)
        .order_by(SessionMessage.created_at.desc())
        .limit(n)
    )
    result = await db.scalars(stmt)
    rows = list(result.all())
    rows.reverse()
    return rows


# ---------------------------------------------------------------------------
# (b, c) Summarisation — session + campaign
# ---------------------------------------------------------------------------


# In-process locks to serialise concurrent summary regeneration per
# entity. AGENTS.md invariant #2 mandates we don't hold a DB transaction
# across the LLM call, but two background tasks racing to regenerate the
# same summary would still waste an LLM round trip and last-writer-wins
# the persisted row. The lock collapses concurrent attempts to one.
_session_summary_locks: dict[str, asyncio.Lock] = {}
_campaign_summary_locks: dict[str, asyncio.Lock] = {}


def _session_lock(session_id: str) -> asyncio.Lock:
    lock = _session_summary_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_summary_locks[session_id] = lock
    return lock


def _campaign_lock(campaign_id: str) -> asyncio.Lock:
    lock = _campaign_summary_locks.get(campaign_id)
    if lock is None:
        lock = asyncio.Lock()
        _campaign_summary_locks[campaign_id] = lock
    return lock


_SESSION_SUMMARY_SYSTEM = (
    "You are a session-archive editor for a tabletop RPG. Given the\n"
    "chronological play log below, produce a tight 400-700 token\n"
    "summary covering: who the party is, where they've been, what\n"
    "they've done, who they've met, what they're carrying, and any\n"
    "cliffhanger / loose threads. Past tense. Third person. No\n"
    "in-character voice. No repetition."
)


_CAMPAIGN_SUMMARY_SYSTEM = (
    "You are a campaign-archive editor for a tabletop RPG. Given the\n"
    "session-by-session log below (and any prior long-term summary),\n"
    "produce a coherent 600-1000 token long-arc summary describing\n"
    "where the campaign currently stands: the party's status, the\n"
    "places and factions they've engaged with, the threads still open,\n"
    "and the next obvious goal. Past tense. Third person. No\n"
    "in-character voice. No repetition. Prefer specifics (names,\n"
    "places) over generalities."
)


async def maybe_regenerate_session_summary(
    db: AsyncSession,
    *,
    session_id: str,
    every_n_turns: int = 20,
) -> str | None:
    """Regenerate ``sessions.summary`` if the player-action count is a
    non-zero multiple of ``every_n_turns``.

    Returns the new summary string on regeneration, ``None`` on no-op.

    Concurrency: per-session ``asyncio.Lock`` serialises racing calls.
    The second caller, finding the summary already up-to-date, returns
    that summary without firing another LLM round-trip.
    """

    # Quick read to check the trigger condition. Open + release a read
    # transaction; we do NOT hold one across the LLM call (invariant #2).
    count_stmt = select(func.count(SessionMessage.id)).where(
        SessionMessage.session_id == session_id,
        SessionMessage.sender_kind == "player",
    )
    player_count = int((await db.execute(count_stmt)).scalar_one())

    if player_count == 0 or player_count % every_n_turns != 0:
        return None

    lock = _session_lock(session_id)
    async with lock:
        # If another coroutine already regenerated while we were waiting
        # on the lock, observe the player count again — if it hasn't
        # advanced past the trigger boundary the summary on disk is the
        # one our trigger would have produced. Return it without another
        # LLM call.
        await db.commit()  # drop any stale read snapshot before re-reading
        recheck_count = int((await db.execute(count_stmt)).scalar_one())
        existing = await db.get(DmSession, session_id)
        if existing is None:
            raise ValueError(f"unknown session_id: {session_id!r}")
        if recheck_count > player_count and existing.summary:
            # Player advanced past us during the wait; trigger no longer
            # applies cleanly. Return the existing summary so the caller
            # has something useful.
            return existing.summary
        if recheck_count == player_count and existing.summary is not None:
            # The "first" coroutine in the lock queue may have already
            # regenerated and persisted; detect it via summary presence
            # at the same count. Avoids double-spending the LLM.
            #
            # Note: this is best-effort. If summary was set BEFORE this
            # function ran (e.g. by a previous trigger), we may skip a
            # regeneration we should have done. The 20-turn cadence is
            # forgiving — one missed trigger means one extra turn of
            # stale summary, not a correctness issue.
            return existing.summary

        # Build the play log inputs — last ~80 messages, two N=40
        # windows. Enough context for a coherent narrative without
        # blowing the summariser's context window.
        msgs_stmt = (
            select(SessionMessage)
            .where(SessionMessage.session_id == session_id)
            .order_by(SessionMessage.created_at.desc())
            .limit(80)
        )
        msgs = list((await db.scalars(msgs_stmt)).all())
        msgs.reverse()

        # Release any read snapshot before the LLM call. SQLAlchemy
        # opens an implicit transaction on first read; commit() ends it.
        await db.commit()

        play_log = _format_play_log(msgs)
        client = get_dm_client()
        summary_text = await _summarise(
            client,
            system_text=_SESSION_SUMMARY_SYSTEM,
            user_text=f"PLAY LOG (chronological):\n\n{play_log}",
        )

        # Tight write transaction to persist.
        target = await db.get(DmSession, session_id)
        if target is None:
            raise ValueError(f"session {session_id!r} disappeared during summarisation")
        target.summary = summary_text
        await db.commit()
        return summary_text


async def regenerate_campaign_summary(
    db: AsyncSession,
    *,
    campaign_id: str,
) -> str:
    """Regenerate ``campaigns.long_summary`` from every session's summary.

    Triggered explicitly at session-end (the API caller does this after
    marking the session ended). Joins each ``sessions.summary``
    chronologically and folds the existing ``campaigns.long_summary``
    in as prior context. Returns the new long_summary string.

    Concurrency: per-campaign ``asyncio.Lock``. Two concurrent triggers
    serialise; the second sees the just-written summary and returns it
    without re-invoking the LLM.
    """

    lock = _campaign_lock(campaign_id)
    async with lock:
        # Refresh the read snapshot under the lock so the second caller
        # sees the first's commit.
        await db.commit()

        campaign = await db.get(Campaign, campaign_id)
        if campaign is None:
            raise ValueError(f"unknown campaign_id: {campaign_id!r}")

        sessions_stmt = (
            select(DmSession)
            .where(DmSession.campaign_id == campaign_id)
            .order_by(DmSession.started_at)
        )
        sessions = list((await db.scalars(sessions_stmt)).all())

        # If there's nothing new to summarise (no session has produced a
        # summary yet), keep whatever long_summary the campaign has.
        per_session_summaries = [s.summary for s in sessions if s.summary]
        prior_long = campaign.long_summary or ""
        if not per_session_summaries and not prior_long:
            return ""

        # Release the read snapshot before the LLM call (invariant #2).
        await db.commit()

        sections: list[str] = []
        if prior_long:
            sections.append(f"PRIOR LONG-TERM SUMMARY:\n{prior_long}")
        if per_session_summaries:
            joined = "\n\n---\n\n".join(
                f"SESSION {i + 1}:\n{s}" for i, s in enumerate(per_session_summaries)
            )
            sections.append(f"SESSION SUMMARIES (chronological):\n\n{joined}")
        user_text = "\n\n".join(sections)

        client = get_dm_client()
        new_long = await _summarise(
            client,
            system_text=_CAMPAIGN_SUMMARY_SYSTEM,
            user_text=user_text,
        )

        target = await db.get(Campaign, campaign_id)
        if target is None:
            raise ValueError(f"campaign {campaign_id!r} disappeared during summarisation")
        target.long_summary = new_long
        await db.commit()
        return new_long


def _format_play_log(messages: list[SessionMessage]) -> str:
    """Render a chronological play log for the summariser prompt.

    Each line is ``[kind] content``. Tool-call audit notes are omitted —
    the summariser is producing prose, not a mechanics audit.
    """

    parts: list[str] = []
    for msg in messages:
        kind = msg.sender_kind.upper()
        # Strip whitespace; some DM messages have huge trailing newlines
        # from streaming.
        body = (msg.content or "").strip()
        if not body:
            continue
        parts.append(f"[{kind}] {body}")
    return "\n\n".join(parts)


async def _summarise(client: DmClient, *, system_text: str, user_text: str) -> str:
    """Single non-streaming LLM call returning the response content.

    Runs at ``reasoning_mode="low"`` — compression is the canonical
    use case for Nemotron's ``low_effort`` template kwarg. Saves a few
    seconds per call and a meaningful chunk of completion tokens; the
    summary's *content* doesn't change perceptibly because the work is
    archival, not creative or structurally-strict. Both session and
    campaign summarisers reuse this helper, so they inherit the choice.
    """

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return await client.complete(
        messages,
        max_tokens=2048,
        temperature=0.3,
        reasoning_mode="low",
    )


# ---------------------------------------------------------------------------
# (d) WorldFactRetriever — per-campaign cosine retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldFactHit:
    """One result from :meth:`WorldFactRetriever.topk`."""

    id: str
    fact: str
    score: float
    importance: int
    tags: list[str]


@dataclass
class _CachedMatrix:
    """Per-campaign in-memory representation of ``world_facts``.

    The matrix is shape ``(N, dim)`` ``float32`` and L2-normalised
    (invariant #5). ``ids[i]`` / ``facts[i]`` / ``importances[i]`` /
    ``tags[i]`` align with row ``i`` of the matrix.
    """

    ids: list[str]
    facts: list[str]
    importances: list[int]
    tags: list[list[str]]
    matrix: NDArray[np.float32]
    dim: int


class WorldFactRetriever:
    """Per-campaign in-memory cache of (id, fact, embedding) tuples.

    Loaded on first ``topk`` for a campaign, invalidated on insert.
    NumPy brute-force cosine similarity (matrix @ query) — at the spec's
    expected scale (≤2k facts per campaign) this is sub-5 ms with zero
    operational complexity. Memory ceiling per active campaign is
    roughly 10 MB at 1024 dims x 2500 facts x 4 bytes.

    Embeddings are pre-normalised on insert (AGENTS.md invariant #5) so
    the cosine reduces to a dot product. The retriever assumes this and
    will silently produce wrong rankings if the invariant is violated —
    extraction is the only path that writes ``world_facts`` and it goes
    through :func:`extract_and_persist_facts`, which uses the embedder
    factory (which always L2-normalises).
    """

    def __init__(self) -> None:
        self._cache: dict[str, _CachedMatrix] = {}

    async def topk(
        self,
        db: AsyncSession,
        campaign_id: str,
        query: str,
        k: int = 5,
    ) -> list[WorldFactHit]:
        """Return up to ``k`` top-scoring facts for ``query``.

        Empty list if the campaign has no facts, ``k <= 0``, or the
        embedder produced a zero-vector for the query (e.g. empty input).
        If ``k`` exceeds the number of facts cached, every fact is
        returned ranked by score.
        """

        if k <= 0:
            return []

        cached = self._cache.get(campaign_id)
        if cached is None:
            cached = await self._load(db, campaign_id)
            self._cache[campaign_id] = cached

        if cached.matrix.shape[0] == 0:
            return []

        embedder = get_embedder()
        query_mat = await embedder.embed([query])
        if query_mat.shape[0] == 0:
            return []
        query_vec = query_mat[0]
        if not np.any(query_vec):
            # Zero-norm query: dot products would all be zero, ranking
            # would be arbitrary. Return empty rather than mislead the
            # caller.
            return []

        scores = cached.matrix @ query_vec  # shape (N,)
        n = scores.shape[0]
        eff_k = min(k, n)

        if eff_k == n:
            # ``argpartition`` requires a strictly partitioned region;
            # asking it to partition all-but-the-end is fine, but if k
            # equals N we just sort directly.
            top = np.argsort(-scores)
        else:
            # Partition the top eff_k (cheaper than a full sort), then
            # sort just those for stable ordering.
            top = np.argpartition(-scores, eff_k - 1)[:eff_k]
            top = top[np.argsort(-scores[top])]

        return [
            WorldFactHit(
                id=cached.ids[int(i)],
                fact=cached.facts[int(i)],
                score=float(scores[int(i)]),
                importance=cached.importances[int(i)],
                tags=cached.tags[int(i)],
            )
            for i in top
        ]

    def invalidate(self, campaign_id: str) -> None:
        """Drop the cache entry for ``campaign_id``. The next ``topk``
        rebuilds from the database."""

        self._cache.pop(campaign_id, None)

    async def _load(self, db: AsyncSession, campaign_id: str) -> _CachedMatrix:
        """Read every ``WorldFact`` for the campaign and decode the BLOB
        embeddings into a single ``(N, dim)`` matrix."""

        stmt = (
            select(WorldFact)
            .where(WorldFact.campaign_id == campaign_id)
            .order_by(WorldFact.created_at)
        )
        rows = list((await db.scalars(stmt)).all())

        if not rows:
            return _CachedMatrix(
                ids=[],
                facts=[],
                importances=[],
                tags=[],
                matrix=np.zeros((0, 0), dtype=np.float32),
                dim=0,
            )

        # Every row in a campaign must share an embedding_dim — switching
        # the embedding model invalidates the campaign's vector space and
        # is treated as a hard configuration error here. Raising rather
        # than silently misrouting is the contract from the orchestrator.
        dims = {r.embedding_dim for r in rows}
        if len(dims) != 1:
            raise EmbeddingDimensionMismatchError(
                f"campaign {campaign_id!r} has world_facts with mixed embedding_dim"
                f" values {sorted(dims)}; the embedding model must not be changed"
                f" mid-campaign without re-embedding existing rows."
            )
        dim = dims.pop()

        ids = [r.id for r in rows]
        facts = [r.fact for r in rows]
        importances = [r.importance for r in rows]
        tags = [list(r.tags) for r in rows]

        # Decode the raw float32 bytes into a (N, dim) matrix. Each row's
        # length must be dim*4 bytes; mismatches indicate a corrupt
        # write and we surface them as the same dimension error.
        matrix = np.empty((len(rows), dim), dtype=np.float32)
        expected_bytes = dim * 4
        for i, row in enumerate(rows):
            blob = row.embedding
            if len(blob) != expected_bytes:
                raise EmbeddingDimensionMismatchError(
                    f"world_fact {row.id!r}: embedding blob is {len(blob)} bytes,"
                    f" expected {expected_bytes} (dim={dim} * 4 bytes/float32)"
                )
            matrix[i] = np.frombuffer(blob, dtype=np.float32)

        return _CachedMatrix(
            ids=ids,
            facts=facts,
            importances=importances,
            tags=tags,
            matrix=matrix,
            dim=dim,
        )


class EmbeddingDimensionMismatchError(RuntimeError):
    """Raised when a campaign's world_facts contain mixed embedding
    dimensions (typically because the embedding model was changed
    without a re-embed migration)."""


_world_fact_retriever_singleton: WorldFactRetriever | None = None


def get_world_fact_retriever() -> WorldFactRetriever:
    """Process-wide :class:`WorldFactRetriever` singleton."""

    global _world_fact_retriever_singleton
    if _world_fact_retriever_singleton is None:
        _world_fact_retriever_singleton = WorldFactRetriever()
    return _world_fact_retriever_singleton


def reset_world_fact_retriever_for_tests() -> None:
    """Drop the singleton so tests start with a fresh cache."""

    global _world_fact_retriever_singleton
    _world_fact_retriever_singleton = None


# ---------------------------------------------------------------------------
# (e) Fact extractor
# ---------------------------------------------------------------------------


class _ExtractedFact(BaseModel):
    """Validated shape of one entry in the extractor's JSON response."""

    fact: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=5, ge=1, le=10)


_FACT_EXTRACTOR_SYSTEM = (
    "You are a memory archivist. Given the most recent player action\n"
    "and DM response, identify any facts that should be remembered\n"
    "long-term: NPC names and traits, locations and their features,\n"
    "decisions made, deals struck, secrets revealed, items acquired\n"
    "or lost, character relationships established.\n"
    "\n"
    'OUTPUT FORMAT (strict): a JSON object {"facts": [...]} whose\n'
    '"facts" value is an array of OBJECTS. Each entry MUST be an\n'
    'object with three keys: "fact" (string, required, the full\n'
    'sentence to remember), "tags" (array of short string labels,\n'
    'optional, may be empty), "importance" (integer 1-10).\n'
    "\n"
    "Bare strings, NPC names alone, or one-word entries are NOT\n"
    "valid — every entry must be the full {fact, tags, importance}\n"
    "object. Importance: 10 = central plot, 1 = trivial colour.\n"
    "\n"
    "EXAMPLE (good):\n"
    '{"facts": [\n'
    '  {"fact": "Castellan Thorvald hired the party to investigate'
    ' goblin raids.", "tags": ["npc", "hook"], "importance": 9},\n'
    '  {"fact": "The keep gates close at dusk.", "tags": ["location"],'
    ' "importance": 4}\n'
    "]}\n"
    "\n"
    "EXAMPLE (bad — do not do this):\n"
    '{"facts": [{"fact": "...", ...}, "Castellan Thorvald", "keep"]}\n'
    "\n"
    'Return {"facts": []} if nothing notable happened. Output ONLY\n'
    "the JSON object — no fences, no commentary."
)


# Match a fenced ```json ... ``` (or ``` ... ```) code block, capturing
# the body. Nemotron sometimes wraps structured responses in fences even
# when ``response_format=json_object`` is set.
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_json_envelope(raw: str) -> str:
    """Pull a JSON object/array out of a possibly-fenced or trailing-prose
    response.

    Handles four Nemotron-isms:

    1. Fenced code block: ```json\\n{...}\\n``` or ```json\\n[...]\\n```.
    2. Pure JSON with leading/trailing whitespace.
    3. JSON object followed by trailing commentary ("Here's the data:
       {...} — hope that helps!").
    4. JSON array followed by trailing commentary (legacy bare-array
       responses, kept for backwards compat with the older prompt).

    Falls back to the original string if none of these match; the JSON
    parser then surfaces the failure with a clear error.
    """

    text = raw.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    # Pick the envelope by which delimiter appears OUTERMOST in the text,
    # not by which is "preferred". A bare-array response with prose
    # ("Here's: [{...}]") would otherwise have its inner object braces
    # mistakenly chosen and the array dropped.
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    has_object = first_brace != -1 and last_brace > first_brace
    has_array = first_bracket != -1 and last_bracket > first_bracket
    # Effective "first index" sentinel for comparison: missing delimiter
    # is treated as +infinity so the present one wins.
    brace_at = first_brace if has_object else float("inf")
    bracket_at = first_bracket if has_array else float("inf")
    if has_object and brace_at <= bracket_at:
        return text[first_brace : last_brace + 1]
    if has_array:
        return text[first_bracket : last_bracket + 1]
    return text


async def extract_and_persist_facts(
    db: AsyncSession,
    *,
    session_id: str,
    player_action: str,
    dm_response: str,
) -> list[str]:
    """Ask the LLM what's worth remembering from a turn and persist any
    returned facts to ``world_facts``.

    Designed to be scheduled as ``asyncio.create_task(...)`` — it must
    NEVER block the next player turn. Failures (LLM error, JSON parse
    error, dimension mismatch) are logged and swallowed.

    Returns the list of fact strings persisted (for logging / tests).
    """

    # Look up the campaign id; the caller passes the session id.
    session = await db.get(DmSession, session_id)
    if session is None:
        log.warning("extract_and_persist_facts: unknown session_id %s", session_id)
        return []
    campaign_id = session.campaign_id

    # Release any read snapshot — the LLM call is about to run.
    await db.commit()

    user_text = f"PLAYER ACTION:\n{player_action}\n\nDM RESPONSE:\n{dm_response}"
    messages = [
        {"role": "system", "content": _FACT_EXTRACTOR_SYSTEM},
        {"role": "user", "content": user_text},
    ]

    client = get_dm_client()
    try:
        # max_tokens=2048 (was 1024) — Phase 3 integration logs showed
        # Nemotron's {facts: [...]} responses for a single rich turn
        # commonly run 1500-1800 tokens, with the previous 1024 cap
        # truncating mid-array and breaking the parse. 2048 leaves
        # headroom; the summarisers run at the same ceiling.
        #
        # reasoning_mode="low" — Phase 5 prep tuned this empirically.
        # The structural prompt below is explicit enough about JSON
        # shape that low-effort reasoning still produces parseable
        # {facts: [...]} payloads; ``_strip_json_envelope`` is the
        # tripwire — if Nemotron's structure breaks under low_effort,
        # parse failures spike in the logs and we escalate this back
        # to "full". Future Phase 8 module extractor stays at "full"
        # because its output is much richer and structurally fragile.
        raw = await client.complete(
            messages,
            response_format={"type": "json_object"},
            max_tokens=2048,
            temperature=0.2,
            reasoning_mode="low",
        )
    except DmClientError:
        log.exception("fact extractor: LLM call failed; skipping turn")
        return []

    stripped = _strip_json_envelope(raw)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        log.warning(
            "fact extractor: JSON parse failed; raw response (first 500 chars): %s",
            raw[:500],
        )
        return []

    # The prompt asks for {"facts": [...]} (which matches response_format=
    # json_object). Earlier prompts asked for a bare array — accept both
    # shapes so tests written against the array-only contract keep passing
    # and so a momentary Nemotron deviation doesn't kill the whole turn.
    entries: list[Any]
    if isinstance(parsed, dict):
        facts_value = parsed.get("facts")
        if not isinstance(facts_value, list):
            log.warning(
                'fact extractor: response object missing "facts" array;'
                " keys=%s; raw (first 500): %s",
                sorted(parsed.keys()),
                raw[:500],
            )
            return []
        entries = facts_value
    elif isinstance(parsed, list):
        entries = parsed
    else:
        log.warning(
            "fact extractor: response was neither object nor array; got %s; raw (first 500): %s",
            type(parsed).__name__,
            raw[:500],
        )
        return []

    facts: list[_ExtractedFact] = []
    for entry in entries:
        # Defence in depth: Nemotron sometimes mixes bare strings into
        # the array even with the schema spelled out in the prompt
        # (Phase 3 finding: ~36% of candidate facts dropped this way).
        # A non-empty string is a recoverable signal — coerce to the
        # canonical {fact, tags, importance} shape with sensible
        # defaults rather than dropping. Anything else (numbers, null,
        # lists, dicts missing 'fact', etc.) still drops with a warning.
        if isinstance(entry, str):
            stripped = entry.strip()
            if not stripped:
                log.warning("fact extractor: dropping empty bare-string entry")
                continue
            entry = {"fact": stripped, "tags": [], "importance": 5}
        try:
            facts.append(_ExtractedFact.model_validate(entry))
        except ValidationError:
            log.warning("fact extractor: dropping malformed fact entry: %s", entry)
            continue

    if not facts:
        return []

    embedder: Embedder = get_embedder()
    try:
        vectors = await embedder.embed([f.fact for f in facts])
    except Exception:  # EmbeddingError or transport failure
        log.exception("fact extractor: embedding call failed; skipping persist")
        return []

    if vectors.shape[0] != len(facts):
        log.warning(
            "fact extractor: embedding row count %d != fact count %d; skipping",
            vectors.shape[0],
            len(facts),
        )
        return []

    persisted: list[str] = []
    for fact, vec in zip(facts, vectors, strict=True):
        # ``vec`` is already L2-normalised by the embedder (invariant #5).
        # Avoid any further transform that could de-normalise it.
        row = WorldFact(
            campaign_id=campaign_id,
            fact=fact.fact,
            embedding=vec.astype(np.float32, copy=False).tobytes(),
            embedding_dim=int(vec.shape[0]),
            tags=fact.tags,
            importance=fact.importance,
            source_session_id=session_id,
        )
        db.add(row)
        persisted.append(fact.fact)

    await db.commit()

    # Cache invalidation must run AFTER the commit so the next topk
    # call sees the new rows on reload.
    get_world_fact_retriever().invalidate(campaign_id)

    return persisted


__all__ = [
    "EmbeddingDimensionMismatchError",
    "WorldFactHit",
    "WorldFactRetriever",
    "extract_and_persist_facts",
    "get_world_fact_retriever",
    "maybe_regenerate_session_summary",
    "recent_turns",
    "regenerate_campaign_summary",
    "reset_world_fact_retriever_for_tests",
]
