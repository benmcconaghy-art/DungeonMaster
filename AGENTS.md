# Dungeon Master — Agent Working Document

You are working on **Dungeon Master**, an LLM-driven Basic Fantasy RPG web app.
This file is loaded at the start of every Claude Code session. For deep details,
read `dungeon-master-spec.md` — it is the source of truth. This document is the
TL;DR.

## What we are building

A self-hosted web app where 2–4 humans play Basic Fantasy RPG with an LLM
Dungeon Master narrating, adjudicating, and running encounters. AI-generated
scene images. Persistent campaigns with long-term memory. Reusable adventure
modules. Single-server deployment on AlmaLinux 10.1, trusted internal LAN.

- **LLM:** Nemotron 3 Super on internal vLLM at `http://svrai01.mcconaghygroup.internal:8000` (OpenAI-compatible)
- **Image gen:** FLUX.1 [dev] + FLUX.1 Kontext [dev] at `http://svrai01.mcconaghygroup.internal:11437`
- **Database:** SQLite 3 with WAL mode, single file at `/var/lib/dungeon-master/dm.db`

## Critical invariants — never violate these

1. **The LLM narrates; the backend adjudicates.** Every dice roll, HP change,
   XP award, inventory mutation goes through the rules engine via tool calls.
   The LLM never declares mechanical outcomes — it requests them.

2. **Never hold a write transaction across an LLM streaming call.** SQLite
   serialises writes; a 30-second open transaction blocks every other writer.
   Pattern: persist input → release transaction → stream tokens → reopen
   transaction → persist completion atomically.

3. **WAL pragmas are mandatory on every connection.** `journal_mode=WAL`,
   `foreign_keys=ON`, `busy_timeout=5000`, `synchronous=NORMAL`. Without WAL,
   multi-player concurrency breaks.

4. **Tool calls are the only way state mutates from LLM output.** The full
   list of mutation tools (`apply_damage`, `award_xp`, `mark_beat`, etc.) lives
   in `app/llm/tools.py`. `session_messages.tool_calls` is the authoritative
   event log.

5. **L2-normalise embeddings before storing.** The retrieval routine assumes
   normalised vectors so cosine reduces to a dot product. Skip normalisation
   and retrieval breaks silently.

## Tech stack

- Python 3.12
- FastAPI (async)
- SQLAlchemy 2.x async + Alembic
- SQLite 3.45+ (WAL)
- Redis 7 (pub/sub + queue for image worker)
- httpx for outbound HTTP
- openai client (pointed at vLLM)
- numpy for vector retrieval
- passlib + bcrypt for auth
- Jinja2 + HTMX 2 + Alpine.js for frontend
- pytest + pytest-asyncio for tests
- uv for dependency / env management

Pin major versions in `pyproject.toml` but allow patch updates. Dependency
upgrades go through their own PR, never bundled with feature work.

## Architecture at a glance

```
Browser ─HTTPS/WSS─> nginx ─> FastAPI (single gunicorn worker)
                                │
                ┌───────────────┼─────────────────┐
                │               │                 │
              SQLite          Redis        Image worker (systemd)
              (WAL)          (pub/sub)            │
                                                  └──> FLUX :11437
                │
                └─ vLLM :8000 (Nemotron 3 Super)
```

Single gunicorn worker on purpose — SQLite serialises writes anyway, multiple
workers add overhead and lock contention. Concurrency comes from FastAPI's
async model within one worker. See spec §13 for rationale.

## Code conventions

### Python
- Type hints on every function and method. `from __future__ import annotations` at module top.
- Pydantic v2 for request/response models, config (`pydantic-settings`), and external API contracts.
- `async def` throughout the request path. No sync I/O in handlers.
- f-strings for formatting. Never `%` or `.format()`.
- Imports grouped: stdlib → third-party → local. Alphabetical within each group.
- Format with `ruff format` (line length 100). Lint with `ruff check`.
- Type-check with `mypy --strict`.

### Database
- SQLAlchemy 2.x style: `select(...)`, `session.scalars(...)`. No legacy `Query`.
- IDs are UUIDv7 strings (use the `uuid7` library), generated in Python, not by SQLite.
- Timestamps are ISO-8601 strings, never bare Python `datetime` in DB columns.
- JSON columns: `TEXT` with `CHECK(json_valid(col))`. Always (de)serialise at the model boundary; business logic deals with Python objects, not JSON strings.
- All FKs declared. `ON DELETE CASCADE` where the child cannot exist without the parent (e.g. `inventory_items` → `characters`).
- Index every column used in a `WHERE` filter. `campaign_id` is almost always indexed.
- Migrations via Alembic autogenerate, but inspect the output before committing — autogenerate misses CHECK constraints and some default expressions.

### FastAPI
- Routers per resource in `app/api/`.
- Shared dependencies in `app/deps.py`.
- Validate every input with Pydantic. Never accept raw `dict` in handlers.
- Return Pydantic models from API endpoints; let FastAPI handle serialisation.
- WebSocket endpoint at `/ws/session/{session_id}`; messages are JSON, types defined in `app/realtime/messages.py`.

### Tests
- Each new module gets a parallel `tests/test_<module>.py` from day one.
- pytest + pytest-asyncio. `@pytest.mark.asyncio` on async tests.
- Mock `vLLM` and `FLUX` clients at the boundary. Never call real services in unit tests.
- Use in-memory SQLite (`sqlite+aiosqlite:///:memory:`) for fast suites; file-backed only for migration tests.
- Inject `random.Random(seed)` into rules-engine functions for reproducibility. No module-level `random.*` calls in production code.

## File organisation

```
app/
├── main.py             # FastAPI factory + lifespan
├── config.py           # pydantic-settings (env-driven config)
├── deps.py             # FastAPI dependencies (db session, current user, etc.)
├── db/
│   ├── base.py         # SQLAlchemy declarative base
│   ├── session.py      # async session factory + WAL pragma event
│   └── models.py       # all ORM models
├── api/                # HTTP routers (one file per resource)
├── llm/
│   ├── client.py       # vLLM/openai client wrapper
│   ├── prompts.py      # prompt builders, layered system prompt
│   ├── tools.py        # tool schemas (Pydantic) + dispatcher
│   ├── memory.py       # session/campaign summarisation, world-fact extractor, NumPy retrieval
│   └── rules_text.py   # condensed BFRPG rules text injected into system prompt
├── game/               # rules engine: dice, combat, chargen, classes, monsters, items
├── images/
│   ├── client.py       # FLUX HTTP client (synchronous /generate, /edit)
│   └── worker.py       # async queue consumer
├── realtime/
│   ├── hub.py          # WebSocket session hub
│   ├── messages.py     # WS message types
│   └── pubsub.py       # Redis pub/sub
├── orchestrator/
│   ├── dm.py           # the DM turn loop
│   └── handlers/       # one file per tool: apply_damage.py, award_xp.py, etc.
├── templates/          # Jinja2
└── static/             # CSS, JS, images

data/bfrpg/             # YAML: classes, spells, monsters, equipment
data/bfrpg/modules/     # bundled modules (Morgansfort)
deploy/                 # nginx.conf, systemd units, SELinux policy
tests/                  # mirrors app/ structure
alembic/                # migration scripts
```

## Common commands

```bash
uv sync                                         # install / update deps
uv run alembic upgrade head                     # run migrations
uv run alembic revision --autogenerate -m "msg" # generate migration
uv run uvicorn app.main:app --reload            # dev server
uv run pytest                                   # full test suite
uv run pytest -k name                           # focused
uv run pytest --cov=app                         # with coverage
uv run ruff check .                             # lint
uv run ruff format .                            # format
uv run mypy app                                 # type check
```

## Where things live in the spec

- **§2** — Architecture diagram, hardware topology, data flow
- **§4** — Why BFRPG, house rules baked into v1
- **§5** — Full SQLite DDL, pragmas, concurrency rules
- **§6** — Rules engine surface
- **§7** — LLM integration: prompt structure, tool calls, memory tiers, NumPy retrieval
- **§8** — FLUX.1 image generation: API contract, performance characteristics, Kontext-based consistency
- **§9** — Multiplayer / WebSocket protocol
- **§10** — Adventure modules (phase 8)
- **§11** — REST + WS endpoints
- **§13** — AlmaLinux deployment, security posture (trusted LAN)
- **§14** — Phased implementation plan
- **§15** — Decisions log

## Working with subagents

Specialist agents are defined in `.claude/agents/`:

- **db-migrations** — schema work, Alembic, SQLite quirks
- **rules-engine** — BFRPG mechanics, dice, combat, chargen
- **bfrpg-data** — YAML content (classes, spells, monsters, equipment)
- **llm-orchestrator** — vLLM client, prompts, tools, memory
- **frontend** — Jinja templates, HTMX, Alpine.js, WS client
- **test-writer** — pytest tests in parallel with implementation

Several have `isolation: worktree` so they automatically run in their own
worktree. For top-level parallel sessions, use `claude --worktree <name>`.

Add `.claude/worktrees/` to `.gitignore`.

## Current build phase

**Phase 2 complete; Phase 3 (memory) ready to start.**

Update this line as phases complete. The phased plan is in spec §14.
