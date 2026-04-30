"""Password hashing and verification (bcrypt).

bcrypt is the default per spec §13 ("simple username + bcrypt-hashed-password
store"). We use the ``bcrypt`` package directly rather than wrapping it in
passlib — passlib 1.7.x is unmaintained and has open compatibility issues
with bcrypt 4.x (the ``__about__`` import path was removed). One less layer,
no version-pinning fight.

Cost factor 12 — modern default; ~250ms per hash on the deploy host, the
right amount of slow for an auth endpoint that runs once per login.

bcrypt's hard 72-byte input limit is hidden behind a SHA-256 pre-hash so
arbitrarily long passwords work without surprising the user. The pre-hash
is keyed by nothing (no separate secret) — its only purpose is to fold a
long input into a 32-byte digest before bcrypt sees it.
"""

from __future__ import annotations

import hashlib

import bcrypt

_BCRYPT_ROUNDS = 12


def _prehash(plaintext: str) -> bytes:
    """SHA-256 the password to a fixed-length input bcrypt always accepts."""

    return hashlib.sha256(plaintext.encode("utf-8")).digest()


def hash_password(plaintext: str) -> str:
    """Return a bcrypt hash of ``plaintext`` suitable for storing in
    ``users.pwd_hash``."""

    salt = bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    return bcrypt.hashpw(_prehash(plaintext), salt).decode("ascii")


def verify_password(plaintext: str, hashed: str) -> bool:
    """Return ``True`` iff ``plaintext`` matches the stored bcrypt
    ``hashed``. Constant-time comparison; safe to call on every login.

    Returns ``False`` (not raises) on malformed hashes so a corrupt row
    in the database fails the login rather than 500-ing the request.
    """

    try:
        return bcrypt.checkpw(_prehash(plaintext), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False
