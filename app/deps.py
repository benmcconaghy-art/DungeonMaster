"""FastAPI dependency providers.

Holds the shared dependencies injected into HTTP handlers and WebSocket
endpoints — the database session and the current authenticated user.

The session-cookie story: on login, the handler sets
``request.session["user_id"]``. ``get_current_user`` reads that value back
and resolves it to a ``User`` row, returning ``None`` if the session is
empty or the user no longer exists. ``require_user`` is the strict variant
that raises 401 instead.

``Annotated[T, Depends(...)]`` aliases are exported so handlers can write
``db: DbSession`` instead of repeating ``Depends(get_db)`` (which trips
ruff's B008 lint).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.db.session import SessionLocal


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yield a session, close it after the request."""

    async with SessionLocal() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user(
    request: Request,
    db: DbSession,
) -> models.User | None:
    """Resolve the session cookie's ``user_id`` to a ``User``, or ``None``."""

    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return await db.get(models.User, user_id)


CurrentUserOrNone = Annotated[models.User | None, Depends(get_current_user)]


async def require_user(user: CurrentUserOrNone) -> models.User:
    """Strict variant: 401 if no current user."""

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="not authenticated",
        )
    return user


CurrentUser = Annotated[models.User, Depends(require_user)]


__all__ = [
    "CurrentUser",
    "CurrentUserOrNone",
    "DbSession",
    "by_username",
    "get_current_user",
    "get_db",
    "require_user",
]


def by_username(username: str) -> Select[tuple[models.User]]:
    """Helper: case-insensitive lookup of a user by username.

    The application-layer normalisation (username stored lowercase) means
    a plain equality match is correct — the column-level COLLATE NOCASE
    in spec §5 is handled here in code rather than at the schema level.
    """

    return select(models.User).where(models.User.username == username.lower())
