"""Factory helpers for orchestrator tests.

Wired against the in-memory ``db_session`` fixture in
``tests/conftest.py``. Each factory persists the ORM row and returns
it so tests can reach into the result for IDs.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Campaign,
    Character,
    Encounter,
    Location,
    SessionMessage,
    User,
)
from app.db.models import (
    Session as DmSession,
)


async def make_user(db: AsyncSession, **overrides: Any) -> User:
    defaults: dict[str, Any] = {
        "username": "tester",
        "email": "tester@example.com",
        "pwd_hash": "x",
    }
    user = User(**(defaults | overrides))
    db.add(user)
    await db.flush()
    return user


async def make_campaign(db: AsyncSession, *, owner_id: str, **overrides: Any) -> Campaign:
    defaults: dict[str, Any] = {
        "name": "Test Campaign",
        "owner_id": owner_id,
        "long_summary": "A small test setting.",
    }
    campaign = Campaign(**(defaults | overrides))
    db.add(campaign)
    await db.flush()
    return campaign


async def make_session(db: AsyncSession, *, campaign_id: str, **overrides: Any) -> DmSession:
    defaults: dict[str, Any] = {"campaign_id": campaign_id}
    session = DmSession(**(defaults | overrides))
    db.add(session)
    await db.flush()
    return session


async def make_character(
    db: AsyncSession, *, user_id: str, campaign_id: str, **overrides: Any
) -> Character:
    defaults: dict[str, Any] = {
        "user_id": user_id,
        "campaign_id": campaign_id,
        "name": "Brunhild",
        "race": "Human",
        "class_name": "Fighter",
        "level": 1,
        "hp_current": 8,
        "hp_max": 8,
        "ac": 14,
        "str_score": 14,
        "int_score": 10,
        "wis_score": 10,
        "dex_score": 12,
        "con_score": 12,
        "cha_score": 10,
        "alignment": "Neutral",
    }
    char = Character(**(defaults | overrides))
    db.add(char)
    await db.flush()
    return char


async def make_location(db: AsyncSession, *, campaign_id: str, **overrides: Any) -> Location:
    defaults: dict[str, Any] = {
        "campaign_id": campaign_id,
        "name": "The Inn",
        "description": "Smoky common room, low rafters.",
    }
    loc = Location(**(defaults | overrides))
    db.add(loc)
    await db.flush()
    return loc


async def make_encounter(db: AsyncSession, *, session_id: str, **overrides: Any) -> Encounter:
    defaults: dict[str, Any] = {
        "session_id": session_id,
        "name": "Goblins",
        "status": "active",
        "monsters": [{"name": "goblin", "count": 3, "hp": 5}],
        "initiative": [
            {"participant_id": "goblin#1", "name": "goblin", "initiative": 4, "is_player": False},
        ],
        "round_number": 1,
        "current_turn": 0,
    }
    enc = Encounter(**(defaults | overrides))
    db.add(enc)
    await db.flush()
    return enc


async def make_message(
    db: AsyncSession,
    *,
    session_id: str,
    sender_kind: str,
    content: str,
    **overrides: Any,
) -> SessionMessage:
    defaults: dict[str, Any] = {
        "session_id": session_id,
        "sender_kind": sender_kind,
        "content": content,
        "audience": [],
    }
    msg = SessionMessage(**(defaults | overrides))
    db.add(msg)
    await db.flush()
    return msg


__all__ = [
    "make_campaign",
    "make_character",
    "make_encounter",
    "make_location",
    "make_message",
    "make_session",
    "make_user",
]
