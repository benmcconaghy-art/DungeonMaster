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

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from app.api.auth import router as auth_router
from app.api.campaigns import router as campaigns_router
from app.api.characters import (
    campaign_scoped_router as characters_campaign_router,
)
from app.api.characters import (
    router as characters_router,
)
from app.api.chargen import router as chargen_router
from app.api.images import router as images_router
from app.api.portraits import router as portraits_router
from app.api.sessions import router as sessions_router
from app.api.ws import router as ws_router
from app.config import get_settings
from app.db import models
from app.db.session import engine
from app.deps import CurrentUser, CurrentUserOrNone, DbSession
from app.images.portrait import get_queue_client
from app.images.portrait import reset_for_tests as reset_queue_client
from app.llm.client import DmClientError, get_dm_client
from app.logging_config import configure_logging
from app.middleware import AccessLogMiddleware, RequestIdMiddleware
from app.realtime.pubsub import DmPubsubError, get_pubsub
from app.views import chargen as chargen_view
from app.views import dashboard as dashboard_view

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

    Boot: install the JSON logging formatter (Phase 7), probe vLLM
    (resolve the served model id) and Valkey (confirm the KV store is
    reachable). Both upstream failures are logged but non-fatal so the
    rest of the app starts and the WS hub / orchestrator surface typed
    errors when first hit. The boot log is the canonical place
    operators see whether the dependencies came up.

    Shutdown: dispose the engine, the openai client, and the Valkey
    connection pool so the gunicorn worker shuts down cleanly under
    systemd.
    """

    configure_logging()

    client = get_dm_client()
    try:
        await client.health()
        log.info("DM client booted; resolved model: %s", client.model)
    except DmClientError as exc:
        log.warning("DM client health check failed at boot: %s", exc)

    pubsub = get_pubsub()
    try:
        await pubsub.health()
        log.info("Valkey pubsub healthy at %s", pubsub.url)
    except DmPubsubError as exc:
        log.warning("Valkey health check failed at boot: %s", exc)

    # Pre-build the image queue client so portrait endpoints don't
    # pay the connection-establishment cost on first request. The
    # Valkey URL is the same as pubsub's, but we keep a separate
    # connection pool — image-queue traffic and session pub/sub
    # have different timeout / retry profiles.
    get_queue_client()

    try:
        yield
    finally:
        await client.aclose()
        await pubsub.aclose()
        await reset_queue_client()
        await engine.dispose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""

    settings = get_settings()
    app = FastAPI(
        title="Dungeon Master",
        version=VERSION,
        lifespan=lifespan,
    )

    # Middleware order (registration is OUTERMOST → INNERMOST as you
    # read down). Starlette runs them in reverse-registration order on
    # the request side and forward order on the response side, so:
    #   1. RequestIdMiddleware runs first on the request, sets the
    #      contextvar, runs LAST on the response (writing the header).
    #   2. AccessLogMiddleware runs after the request_id is bound and
    #      before SessionMiddleware so its access log carries the
    #      request_id but not yet the user_id (the user-resolution
    #      dependency sets that lazily once auth runs).
    #   3. SessionMiddleware is the innermost — closest to the handler.
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIdMiddleware)

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

    # Rate limiting (Phase 7 hardening) is wired per-route via FastAPI
    # ``Depends(login_rate_limit)`` etc. on the auth/join endpoints.
    # No app-level middleware required: the dependency raises
    # HTTPException(429) with a Retry-After header and a human message
    # when a limit trips.

    app.include_router(auth_router)
    app.include_router(campaigns_router)
    app.include_router(chargen_router)
    app.include_router(characters_router)
    app.include_router(characters_campaign_router)
    app.include_router(images_router)
    app.include_router(portraits_router)
    app.include_router(sessions_router)
    app.include_router(ws_router)

    app.mount(
        "/static",
        StaticFiles(directory=str(_APP_DIR / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, user: CurrentUserOrNone) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {"version": VERSION, "user": user},
        )

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "login.html", {})

    @app.get("/register", response_class=HTMLResponse)
    async def register_form(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "register.html", {})

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(
        request: Request,
        user: CurrentUser,
        db: DbSession,
    ) -> HTMLResponse:
        ctx = await dashboard_view.build_context(db, user=user)
        return _TEMPLATES.TemplateResponse(request, "campaign_dashboard.html", ctx)

    @app.get("/campaigns/{campaign_id}/party", response_class=HTMLResponse)
    async def party_setup(
        request: Request,
        campaign_id: str,
        user: CurrentUser,
        db: DbSession,
    ) -> HTMLResponse:
        campaign = await db.get(models.Campaign, campaign_id)
        if campaign is None:
            raise HTTPException(status_code=404, detail="campaign not found")
        membership = await db.get(models.CampaignMember, (campaign_id, user.id))
        if membership is None:
            raise HTTPException(status_code=403, detail="not a member of this campaign")
        return _TEMPLATES.TemplateResponse(
            request,
            "party_setup.html",
            {"user": user, "campaign": campaign},
        )

    @app.get("/campaigns/{campaign_id}/chargen", response_class=HTMLResponse)
    async def chargen_page(
        request: Request,
        campaign_id: str,
        user: CurrentUser,
        db: DbSession,
    ) -> HTMLResponse:
        ctx = await chargen_view.build_context(db, campaign_id=campaign_id, user=user)
        return _TEMPLATES.TemplateResponse(request, "chargen.html", ctx)

    @app.get("/characters/{character_id}", response_class=HTMLResponse)
    async def character_sheet(
        request: Request,
        character_id: str,
        user: CurrentUser,
        db: DbSession,
    ) -> HTMLResponse:
        from sqlalchemy import select as _sa_select

        from app.api.characters import _detail_response, _require_character_visibility

        character = await _require_character_visibility(db, character_id=character_id, user=user)
        inventory = list(
            (
                await db.execute(
                    _sa_select(models.InventoryItem)
                    .where(models.InventoryItem.character_id == character_id)
                    .order_by(
                        models.InventoryItem.equipped.desc(),
                        models.InventoryItem.name,
                    )
                )
            ).scalars()
        )
        spells = list(
            (
                await db.execute(
                    _sa_select(models.SpellKnown)
                    .where(models.SpellKnown.character_id == character_id)
                    .order_by(models.SpellKnown.spell_level, models.SpellKnown.spell_name)
                )
            ).scalars()
        )
        detail = _detail_response(character, viewer_id=user.id, inventory=inventory, spells=spells)
        campaign = await db.get(models.Campaign, character.campaign_id)
        return _TEMPLATES.TemplateResponse(
            request,
            "character_sheet.html",
            {"user": user, "character": detail, "campaign": campaign},
        )

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
                "user": user,
                "characters": characters,
                "characters_by_id": characters_by_id,
                "messages": recent_messages,
                "recent_rolls": recent_rolls,
            },
        )

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        """Prometheus exposition format. Internal-only: nginx restricts
        the location to localhost; reaching this requires shell access
        to the box. The handler itself is intentionally unauthenticated
        because Prometheus scrapers don't carry session cookies — the
        deployment-layer restriction is the gate.
        """

        from app.metrics import render_exposition

        body, content_type = render_exposition()
        return Response(content=body, media_type=content_type)

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
