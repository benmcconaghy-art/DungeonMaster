"""FastAPI app factory and lifespan.

Builds the app, mounts static assets, registers Phase 0's two routes
(``/`` and ``/health``), and disposes of the SQLAlchemy engine on
shutdown so the gunicorn worker shuts down cleanly under systemd.

Phase 1 onward will register routers from ``app/api/`` here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import text

from app.db.session import SessionLocal, engine

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

    Phase 0: nothing to start at boot. On shutdown, dispose the engine so
    the gunicorn worker doesn't leave dangling SQLite handles.
    """

    try:
        yield
    finally:
        await engine.dispose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""

    app = FastAPI(
        title="Dungeon Master",
        version=VERSION,
        lifespan=lifespan,
    )

    app.mount(
        "/static",
        StaticFiles(directory=str(_APP_DIR / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "base.html",
            {"version": VERSION},
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> JSONResponse:
        db_status: Literal["ok", "error"] = "ok"
        try:
            async with SessionLocal() as session:
                await session.execute(text("SELECT 1"))
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
