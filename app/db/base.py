"""SQLAlchemy declarative base.

Every ORM model in ``app/db/models.py`` inherits from ``Base``. Keeping the
Base in its own module breaks the import cycle Alembic would otherwise
trip on (``env.py`` needs ``Base.metadata`` without pulling in the live
session machinery).
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
