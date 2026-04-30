"""FastAPI app factory and lifespan.

Builds the app, mounts static assets, registers ``SessionMiddleware``
(spec §13: server-signed session cookies for auth), registers the auth
router and the bootstrap routes, and disposes of the SQLAlchemy engine
on shutdown so the gunicorn worker shuts down cleanly under systemd.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from app.api.auth import router as auth_router
from app.api.campaigns import router as campaigns_router
from app.api.characters import router as characters_router
from app.api.sessions import router as sessions_router
from app.api.sse import router as sse_router
from app.config import get_settings
from app.db import models
from app.db.session import engine
from app.deps import CurrentUser, CurrentUserOrNone, DbSession
from app.llm.client import DmClientError, get_dm_client

log = logging.getLogger(__name__)

_APP_DIR = Path(__file__).resolve().parent
_TEMPLATES = Jinja2Templates(directory=str(_APP_DIR / "templates"))


def _project_version() -> str:
    """Resolve the project version from installed metadata.

    Falls back to ``"0+unknown"`` only if the package is somehow not
    installed (which would mean ``uv sync`` hasn't run — the deployment
    paths all install the package, so the fallback is dev-only insurance).
    """

    try:
        return version("dungeon-master")
    except PackageNotFoundError:
        return "0+unknown"


VERSION: str = _project_version()


class HealthResponse(BaseModel):
    """Shape returned by ``GET /health``."""

    status: Literal["ok", "error"]
    db: Literal["ok", "error"]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Process-level setup/teardown.

    Boot: probe the vLLM endpoint and resolve the served model id (so a
    misconfigured endpoint fails the worker startup loudly rather than
    on first DM turn). Failure is non-fatal — the rest of the app
    starts; ``/health`` will reflect the broken DM client and the
    orchestrator will surface ``dm_error`` events when a turn is
    attempted.

    Shutdown: dispose the engine and the openai client so the gunicorn
    worker shuts down cleanly under systemd.
    """

    client = get_dm_client()
    try:
        await client.health()
        log.info("DM client booted; resolved model: %s", client.model)
    except DmClientError as exc:
        log.warning("DM client health check failed at boot: %s", exc)

    try:
        yield
    finally:
        await client.aclose()
        await engine.dispose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""

    settings = get_settings()
    app = FastAPI(
        title="Dungeon Master",
        version=VERSION,
        lifespan=lifespan,
    )

    # Signed-cookie sessions per spec §13. ``https_only`` would set the
    # Secure flag, but our self-signed dev cert means a dev browser
    # connecting via plain http:// would refuse to send the cookie back —
    # leave it off so dev works locally; production runs behind nginx TLS
    # so cookies arrive over HTTPS regardless.
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        session_cookie="dm_session",
        max_age=60 * 60 * 24 * 30,  # 30 days, per spec §13
        same_site="lax",
    )

    app.include_router(auth_router)
    app.include_router(campaigns_router)
    app.include_router(characters_router)
    app.include_router(sessions_router)
    app.include_router(sse_router)

    app.mount(
        "/static",
        StaticFiles(directory=str(_APP_DIR / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, user: CurrentUserOrNone) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "base.html",
            {"version": VERSION, "user": user},
        )

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "login.html", {})

    @app.get("/play/{session_id}", response_class=HTMLResponse)
    async def play_screen(
        request: Request,
        session_id: str,
        user: CurrentUser,
        db: DbSession,
    ) -> HTMLResponse:
        from sqlalchemy import select

        session = await db.get(models.Session, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        membership = await db.get(models.CampaignMember, (session.campaign_id, user.id))
        if membership is None:
            raise HTTPException(status_code=403, detail="not a member of this campaign")

        campaign = await db.get(models.Campaign, session.campaign_id)
        characters = list(
            (
                await db.scalars(
                    select(models.Character)
                    .where(models.Character.campaign_id == session.campaign_id)
                    .where(models.Character.user_id == user.id)
                    .order_by(models.Character.name)
                )
            ).all()
        )
        recent_messages = list(
            (
                await db.scalars(
                    select(models.SessionMessage)
                    .where(models.SessionMessage.session_id == session_id)
                    .order_by(models.SessionMessage.created_at.desc())
                    .limit(50)
                )
            ).all()
        )
        recent_messages.reverse()
        # Filter whispers the user shouldn't see.
        visible_character_ids = {c.id for c in characters}
        recent_messages = [
            m
            for m in recent_messages
            if not m.audience or any(cid in visible_character_ids for cid in m.audience)
        ]
        characters_by_id = {c.id: c for c in characters}

        recent_rolls = list(
            (
                await db.scalars(
                    select(models.DiceRoll)
                    .where(models.DiceRoll.session_id == session_id)
                    .order_by(models.DiceRoll.created_at.desc())
                    .limit(10)
                )
            ).all()
        )

        return _TEMPLATES.TemplateResponse(
            request,
            "table.html",
            {
                "session_id": session_id,
                "campaign": campaign,
                "characters": characters,
                "characters_by_id": characters_by_id,
                "messages": recent_messages,
                "recent_rolls": recent_rolls,
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health(db: DbSession) -> JSONResponse:
        db_status: Literal["ok", "error"] = "ok"
        try:
            await db.execute(text("SELECT 1"))
        except Exception:
            db_status = "error"

        body = HealthResponse(
            status="ok" if db_status == "ok" else "error",
            db=db_status,
        )
        return JSONResponse(
            status_code=200 if db_status == "ok" else 503,
            content=body.model_dump(),
        )

    return app


app = create_app()
