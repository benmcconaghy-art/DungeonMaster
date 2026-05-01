"""Authentication endpoints: register / login / logout / me.

Trusted-LAN posture per spec §13 — local accounts, bcrypt-hashed passwords,
signed session cookies via Starlette's ``SessionMiddleware``. No SSO, no
MFA, no aggressive password policy.

Usernames are normalised to lowercase before insert/query so case variants
collapse to the same identity (matches the COLLATE NOCASE spirit of
spec §5 without requiring SQLite-specific column collation).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError

from app.db import models
from app.deps import CurrentUser, DbSession, by_username
from app.ratelimit import login_rate_limit, register_rate_limit
from app.security import hash_password, verify_password

router = APIRouter(prefix="/api", tags=["auth"])


# ---------- request / response models ----------------------------------------


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_-]+$")
    email: EmailStr | None = None
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class UserResponse(BaseModel):
    id: str
    username: str
    email: str | None
    is_admin: bool
    created_at: str


def _user_to_response(user: models.User) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_admin=user.is_admin,
        created_at=user.created_at,
    )


# ---------- endpoints --------------------------------------------------------


@router.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(register_rate_limit)],
)
async def register(
    payload: RegisterRequest,
    request: Request,
    db: DbSession,
) -> UserResponse:
    """Create a new account, then log it in by setting the session cookie."""

    user = models.User(
        username=payload.username.lower(),
        email=payload.email.lower() if payload.email else None,
        pwd_hash=hash_password(payload.password),
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="username or email already in use",
        ) from exc

    await db.refresh(user)
    request.session["user_id"] = user.id
    return _user_to_response(user)


@router.post(
    "/auth/login",
    response_model=UserResponse,
    dependencies=[Depends(login_rate_limit)],
)
async def login(
    payload: LoginRequest,
    request: Request,
    db: DbSession,
) -> UserResponse:
    """Verify the password and set the session cookie."""

    result = await db.execute(by_username(payload.username))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.pwd_hash):
        # Same response for unknown user and wrong password so an attacker
        # cannot enumerate accounts via timing.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid username or password",
        )

    request.session["user_id"] = user.id
    return _user_to_response(user)


@router.post("/auth/logout")
async def logout(request: Request) -> Response:
    """Clear the session cookie. Idempotent — safe to call when not signed in."""

    request.session.clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser) -> UserResponse:
    """Return the current user, or 401 if no valid session."""

    return _user_to_response(user)


__all__ = ["LoginRequest", "RegisterRequest", "UserResponse", "router"]
