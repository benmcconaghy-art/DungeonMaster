"""Bootstrap script: register a system module on first server startup.

Usage:
    uv run python -m app.scripts.load_module morgansfort

Loads the named module from data/bfrpg/modules/<name>.json, validates it
against ModuleContent, and inserts (or skips if already present) a Module
row owned by the system admin user.

A "system admin user" is the first user in the database with is_admin=True.
If no admin user exists, the script exits with an error rather than creating
a synthetic user — the server setup should ensure an admin exists first.

Idempotent: if a Module row with the same name already exists (case-insensitive),
the script exits 0 with a "skipped" message. This makes it safe to run on
every startup without fear of duplication.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from sqlalchemy import func, select

log = logging.getLogger(__name__)


def _resolve_module_path(module_name: str) -> Path:
    """Find the JSON file for a module by name."""
    repo_root = Path(__file__).parent.parent.parent
    path = repo_root / "data" / "bfrpg" / "modules" / f"{module_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"module file not found: {path}")
    return path


async def _load(module_name: str) -> None:
    from app.db.models import Module, User
    from app.db.session import SessionLocal
    from app.llm.modules import ModuleContent
    from pydantic import ValidationError

    module_path = _resolve_module_path(module_name)
    log.info("loading module from %s", module_path)

    with module_path.open() as f:
        raw = json.load(f)

    try:
        content = ModuleContent.model_validate(raw)
    except ValidationError as exc:
        print(f"ERROR: {module_path.name} failed ModuleContent validation:\n{exc}", file=sys.stderr)
        sys.exit(1)

    async with SessionLocal() as db:
        # Find the system admin user.
        admin = (
            await db.scalars(select(User).where(User.is_admin.is_(True)).limit(1))
        ).first()
        if admin is None:
            print(
                "ERROR: no admin user found in database. "
                "Create an admin user before running this script.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Idempotence: skip if a module with this name already exists.
        existing_name = content.synopsis[:40] + "..."  # use synopsis prefix for logging
        module_title = f"{module_name.replace('_', ' ').title()}"
        existing = (
            await db.scalars(
                select(Module).where(func.lower(Module.name) == module_title.lower()).limit(1)
            )
        ).first()
        if existing is not None:
            print(f"SKIP: module '{module_title}' already registered (id={existing.id})")
            return

        module_row = Module(
            author_id=admin.id,
            name=module_title,
            description=content.synopsis[:200] if content.synopsis else None,
            min_level=content.level_range[0] if content.level_range else None,
            max_level=content.level_range[-1] if len(content.level_range) > 1 else None,
            tone=content.tone,
            estimated_sessions=content.estimated_sessions,
            content=raw,
            public=True,
        )
        db.add(module_row)
        await db.commit()
        await db.refresh(module_row)

        print(
            f"OK: registered module '{module_title}' "
            f"(id={module_row.id}, "
            f"{len(content.locations)} locations, "
            f"{len(content.npcs)} NPCs, "
            f"{len(content.plot_beats)} beats)"
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    if len(sys.argv) < 2:
        print("Usage: uv run python -m app.scripts.load_module <module_name>", file=sys.stderr)
        print("Example: uv run python -m app.scripts.load_module morgansfort", file=sys.stderr)
        sys.exit(1)

    module_name = sys.argv[1]
    asyncio.run(_load(module_name))


if __name__ == "__main__":
    main()
