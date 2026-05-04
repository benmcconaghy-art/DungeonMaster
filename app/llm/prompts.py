"""DM prompt builders.

Composes the layered system prompt described in spec §7: role, condensed
BFRPG rules, house rules, campaign / location / PCs / NPCs, recent turns,
session summary, retrieved world facts, and active encounter state.

Phase 2 surface is :func:`build_dm_prompt`. It returns the full
OpenAI-style chat-completions message list — a single ``[system]``
message followed by alternating ``[user]`` / ``[assistant]`` messages
for the verbatim recent-turns tier. Empty sections are rendered as
``(none)`` rather than skipped so the prompt structure stays visible to
humans debugging.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Campaign,
    Character,
    Encounter,
    Location,
    SessionMessage,
)
from app.db.models import (
    Session as DmSession,
)
from app.llm.memory import WorldFactHit, get_world_fact_retriever, recent_turns
from app.llm.rules_text import render_rules_text

# Spec §4 default house rules. We render the *defaults* explicitly when
# the campaign hasn't overridden anything, so the DM sees them in every
# prompt regardless of the JSON column being empty.
_DEFAULT_HOUSE_RULES: list[tuple[str, str]] = [
    ("Death and Dismemberment", "On (BFRPG-compatible OSR table)"),
    ("Death's Door (negative HP buffer)", "Off"),
    ("Fast healing (1d3/day)", "On"),
    ("Variable weapon damage", "On"),
    ("Ascending AC only", "On"),
    ("XP for treasure", "On (1 XP per 1 gp recovered)"),
]


_ROLE_TEXT = (
    "You are the Dungeon Master for a Basic Fantasy RPG game. You narrate vividly,\n"
    "adjudicate fairly, and maintain a gritty tone. Player characters can die — do\n"
    "not pull punches, but do telegraph danger. Speak in the second person to the\n"
    'players ("you see", "you hear"); describe NPCs and creatures in third person.\n'
    "\n"
    "DISCIPLINE — non-negotiable:\n"
    "  - Never roll your own dice. Always call request_dice_roll and use the result.\n"
    "  - Never declare HP changes in prose. Always call apply_damage or heal.\n"
    "  - Never move the party in prose. Always call transition_location.\n"
    "  - Never narrate initiative order or monster HP. The engine owns those facts.\n"
    "  - Send private DM-to-one-player content via whisper, not by addressing them\n"
    "    publicly. Other players cannot read whispers.\n"
    "  - When you call a tool, the engine returns the authoritative outcome. Narrate\n"
    "    that outcome faithfully on the next turn — do not contradict it.\n"
    "  - Never ask the player for ids, identifiers, location_id, character_id, or\n"
    "    any other schema parameter — those are engine concerns, not fiction.\n"
    "    Reference locations, characters, NPCs, and items by name in narration\n"
    "    and pass them as ``name`` arguments to tools; the engine resolves names\n"
    "    against the campaign's existing entities or creates new ones as needed.\n"
    "    Surfacing a database id to the player is a fourth-wall break.\n"
    "\n"
    "PLAYER ATTRIBUTION — multi-character parties:\n"
    "  - Player messages arrive prefixed with [Character Name, Class]:\n"
    '    e.g. "[Slowhand, Fighter]: I draw my axe and charge the goblin."\n'
    "    The bracketed prefix is engine metadata identifying the speaking\n"
    "    character — it is NOT part of what the player typed. Do not echo\n"
    "    it back, do not treat it as fiction.\n"
    "  - Address the named character. Resolve the action that character\n"
    "    declared, and never mistake one PC's action for another's. Never\n"
    "    ask the player to clarify which character is speaking — the\n"
    "    prefix tells you.\n"
    "  - When more than one PC is in the scene, name the speaker once at\n"
    '    the start of the response so "you" is unambiguous to everyone\n'
    '    at the table — e.g. "Lila, you scan the common room…".\n'
    "    Otherwise the other players cannot tell who acted.\n"
    "  - A message without a prefix is engine context (a system note),\n"
    "    not a player utterance.\n"
    "\n"
    "PACING — give the player a turn back (critical for agency):\n"
    "  - Always narrate. Always describe what happens. Pacing is about\n"
    "    when to wrap up, not whether to write — close the message with a\n"
    "    natural beat ending and an implicit invitation to act, then wait\n"
    "    for the player's next message.\n"
    "  - One player decision per turn. Resolve the action they declared\n"
    "    and the round it triggers; wrap up the message after that beat.\n"
    "    The next decision belongs to the player.\n"
    "  - Chain tool calls freely WITHIN one beat (to-hit + damage +\n"
    "    apply_damage for a single attack is one beat — chain those).\n"
    "  - In combat: resolve at most ONE complete initiative round per\n"
    "    message. Once every participant has acted once this round, narrate\n"
    "    the result and end your reply. The next round comes after the\n"
    "    player declares their next action.\n"
    "  - After a scene transition or an important discovery, narrate the\n"
    "    new context vividly, then end your reply so the player can react.\n"
    "  - Don't roll player saves on their behalf. If an effect targets a\n"
    "    player (spell, trap, breath weapon), narrate the threat and end\n"
    "    your reply. The player declares their response on their next\n"
    "    turn and you resolve it then.\n"
    "  - If you've already made five or six tool calls this turn and the\n"
    "    player hasn't gotten a chance to react, you've over-narrated —\n"
    "    wrap the current beat and hand control back."
)


async def build_dm_prompt(
    db: AsyncSession,
    *,
    session_id: str,
    recent_turns_n: int = 40,
) -> list[dict[str, Any]]:
    """Build the full chat-completions message list for one DM turn.

    Returns a list of OpenAI-style messages: a single system message
    composed of the layered sources documented in spec §7, followed by
    alternating user/assistant messages for the verbatim recent-turns
    tier (so the model sees them as a real conversation rather than a
    text dump in the system block).

    The session must exist; the function is read-only and does not
    open a write transaction. Callers wrap the broader turn loop in
    their own transaction discipline (AGENTS.md invariant #2).
    """

    session = await db.get(DmSession, session_id)
    if session is None:
        raise ValueError(f"unknown session_id: {session_id!r}")

    campaign = await db.get(Campaign, session.campaign_id)
    if campaign is None:
        raise ValueError(f"session {session_id!r} references missing campaign")

    location = (
        await db.get(Location, session.current_location_id)
        if session.current_location_id is not None
        else None
    )

    characters_stmt = (
        select(Character)
        .where(Character.campaign_id == campaign.id)
        .where(Character.status == "alive")
        .order_by(Character.name)
    )
    characters = list((await db.scalars(characters_stmt)).all())

    # Speaker-attribution index: every character in the campaign, regardless
    # of status, so a recently-deceased PC's earlier turns still get attributed
    # in the prompt history. AGENTS.md invariant #17.
    attribution_stmt = select(Character).where(Character.campaign_id == campaign.id)
    character_index: dict[str, tuple[str, str]] = {
        ch.id: (ch.name, ch.class_name) for ch in (await db.scalars(attribution_stmt)).all()
    }

    encounters_stmt = (
        select(Encounter)
        .where(Encounter.session_id == session.id)
        .where(Encounter.status == "active")
        .order_by(Encounter.created_at)
    )
    active_encounters = list((await db.scalars(encounters_stmt)).all())

    turns = await recent_turns(db, session_id=session.id, n=recent_turns_n)

    # Retrieve relevant world facts using the most recent player message
    # as the query. Skip retrieval entirely if no player has spoken yet
    # (the section renders ``(none yet)``). The retriever is read-only
    # and has no transaction discipline implications.
    last_player_query = _last_player_content(turns)
    world_fact_hits: list[WorldFactHit] | None
    if last_player_query is None:
        world_fact_hits = None
    else:
        world_fact_hits = await get_world_fact_retriever().topk(
            db, campaign.id, last_player_query, k=5
        )

    system_text = _render_system(
        campaign=campaign,
        location=location,
        characters=characters,
        encounters=active_encounters,
        session=session,
        world_fact_hits=world_fact_hits,
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
    messages.extend(_recent_turns_to_messages(turns, character_index=character_index))
    return messages


def _last_player_content(turns: list[SessionMessage]) -> str | None:
    """Return the most recent player message's content, or ``None``."""

    for msg in reversed(turns):
        if msg.sender_kind == "player" and msg.content:
            return msg.content
    return None


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_system(
    *,
    campaign: Campaign,
    location: Location | None,
    characters: list[Character],
    encounters: list[Encounter],
    session: DmSession,
    world_fact_hits: list[WorldFactHit] | None,
) -> str:
    """Compose every layered section into the single system message body."""

    blocks: list[str] = []

    blocks.append(_block("ROLE", _ROLE_TEXT))
    blocks.append(_block("RULES SUMMARY", render_rules_text()))
    blocks.append(_block("HOUSE RULES", _render_house_rules(campaign.house_rules)))
    blocks.append(_block("CAMPAIGN", _render_campaign(campaign)))
    blocks.append(_block("CURRENT LOCATION", _render_location(location)))
    blocks.append(_block("ACTIVE PCs", _render_characters(characters)))
    # Phase 2 leaves NPCs in the scene empty — Phase 3 wires it in.
    blocks.append(_block("ACTIVE NPCs IN SCENE", "(none)"))
    # Recent turns are emitted as alternating messages outside the system
    # block, but we leave the marker so the prompt's structure is visible
    # to humans reading the system body.
    blocks.append(
        _block(
            "RECENT TURNS",
            "(rendered as alternating user/assistant messages below this system block)",
        )
    )
    blocks.append(_block("SESSION SO FAR", session.summary or "(none)"))
    blocks.append(_block("RELEVANT WORLD FACTS", _render_world_fact_hits(world_fact_hits)))
    blocks.append(_block("ACTIVE ENCOUNTER", _render_encounters(encounters)))
    # Phase 8 wires modules in.
    blocks.append(_block("MODULE", "(none — Phase 8 will fill)"))

    return "\n\n".join(blocks)


def _render_world_fact_hits(hits: list[WorldFactHit] | None) -> str:
    """Bullet list of retrieved world facts, sorted by score descending.

    ``hits is None`` means we never queried (no player message yet) and
    we render ``(none yet)``. ``hits == []`` means we queried and got
    nothing — render ``(none retrieved)`` so the human reading the
    prompt can tell the two states apart.
    """

    if hits is None:
        return "(none yet)"
    if not hits:
        return "(none retrieved)"
    # Hits arrive sorted by score descending from the retriever, but
    # sort defensively in case a future caller pre-filters / reorders.
    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    lines: list[str] = []
    for hit in ordered:
        tag_str = ",".join(hit.tags) if hit.tags else "-"
        lines.append(f"  - [{hit.importance}/10, {tag_str}] {hit.fact}")
    return "\n".join(lines)


def _block(title: str, body: str) -> str:
    """Render one ``[SECTION]\\n<body>`` block."""

    return f"[{title}]\n{body}"


def _render_house_rules(overrides: dict[str, Any]) -> str:
    """Bullet list of the spec §4 defaults, with campaign overrides folded in.

    Phase 2 campaigns won't have non-default values, but if a key is
    present we emit the override value rather than the default.
    """

    lines: list[str] = []
    for label, default in _DEFAULT_HOUSE_RULES:
        # Allow either the human label or a snake_case key for overrides.
        snake = label.lower().replace(" ", "_").replace("-", "_")
        value = overrides.get(snake)
        if isinstance(value, bool):
            rendered = "On" if value else "Off"
        elif value is None:
            rendered = default
        else:
            rendered = str(value)
        lines.append(f"  - {label}: {rendered}")
    # Surface any extra overrides verbatim so a human DM can see them in
    # the prompt (the LLM may not understand them, but at least nothing
    # is lost).
    known_keys = {
        label.lower().replace(" ", "_").replace("-", "_") for label, _ in _DEFAULT_HOUSE_RULES
    }
    for key, val in overrides.items():
        if key in known_keys:
            continue
        lines.append(f"  - {key}: {val}")
    return "\n".join(lines)


def _render_campaign(campaign: Campaign) -> str:
    summary = campaign.long_summary or "(no long-term summary yet)"
    return f"Name: {campaign.name}\nLong-term context: {summary}"


def _render_location(location: Location | None) -> str:
    if location is None:
        return "(none)"
    description = location.description or "(no description)"
    return f"{location.name} — {description}"


def _render_characters(characters: list[Character]) -> str:
    if not characters:
        return "(none)"
    lines: list[str] = []
    for ch in characters:
        lines.append(
            f"  - {ch.name} ({ch.race} {ch.class_name} L{ch.level}) — "
            f"HP {ch.hp_current}/{ch.hp_max}, AC {ch.ac}, "
            f"STR {ch.str_score} DEX {ch.dex_score} CON {ch.con_score} "
            f"INT {ch.int_score} WIS {ch.wis_score} CHA {ch.cha_score}, "
            f"status: {ch.status} [id={ch.id}]"
        )
    return "\n".join(lines)


def _render_encounters(encounters: list[Encounter]) -> str:
    if not encounters:
        return "(none)"
    parts: list[str] = []
    for enc in encounters:
        initiative_lines: list[str] = []
        for entry in enc.initiative or []:
            if isinstance(entry, dict):
                name = entry.get("name", "?")
                init = entry.get("initiative", "?")
                initiative_lines.append(f"      {init}: {name}")
        monster_lines: list[str] = []
        for monster in enc.monsters or []:
            if isinstance(monster, dict):
                m_name = monster.get("name", "?")
                m_hp = monster.get("hp", "?")
                m_count = monster.get("count", 1)
                monster_lines.append(f"      {m_name} x{m_count} (hp {m_hp})")
        parts.append(
            f"  - {enc.name} (round {enc.round_number}, turn index {enc.current_turn})"
            + ("\n    initiative:\n" + "\n".join(initiative_lines) if initiative_lines else "")
            + ("\n    monsters (DM-only):\n" + "\n".join(monster_lines) if monster_lines else "")
        )
    return "\n".join(parts)


def _recent_turns_to_messages(
    turns: list[SessionMessage],
    *,
    character_index: dict[str, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Convert verbatim ``SessionMessage`` rows into chat-completions messages.

    Mapping:
      sender_kind == 'player'  -> {"role": "user", "content": ...}
      sender_kind == 'dm'      -> {"role": "assistant", "content": ...}
      sender_kind == 'system'  -> {"role": "system", "content": ...}
                                  (e.g. location-change banners)

    Whispers (audience non-empty) are still surfaced — the DM needs to
    see what it whispered so it stays consistent. Phase 5+ multi-player
    will filter for the *receiving* player but the DM-side prompt
    always sees everything.

    Player messages are prefixed with ``[Name, Class]:`` when the
    speaker resolves through ``character_index``. The OpenAI chat format
    has no per-message speaker field that vLLM/Nemotron's chat template
    is guaranteed to honour, so attribution lives in the message body.
    See AGENTS.md invariant #17.
    """

    index = character_index or {}

    out: list[dict[str, Any]] = []
    for msg in turns:
        if msg.sender_kind == "player":
            content = msg.content
            speaker = index.get(msg.sender_id) if msg.sender_id else None
            if speaker is not None:
                name, class_name = speaker
                content = f"[{name}, {class_name}]: {content}"
            out.append({"role": "user", "content": content})
        elif msg.sender_kind == "dm":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                # Surface the tool-call audit so the model has a faithful
                # record of what it asked the engine for last time. We
                # don't re-emit them as live tool_calls (the engine has
                # already executed them) — embedding them in the content
                # is the simplest way to keep the audit visible without
                # the API treating them as new pending calls.
                entry["content"] = (
                    msg.content + "\n\n[engine: previously executed tool calls — informational]"
                )
            out.append(entry)
        else:
            # 'system', or any other future kind — render as a neutral system note.
            out.append({"role": "system", "content": msg.content})
    return out


__all__ = ["build_dm_prompt"]
