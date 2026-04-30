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

6. **Only declare implemented tools to the LLM.** `tool_definitions(only_implemented=True)`
   is the orchestrator's default. Surfacing not-yet-implemented stubs causes Nemotron
   to call them, get a "not_implemented" tool result, retry — and chain to the
   iteration cap. Tools become visible automatically as their handlers register.

7. **Tool-call iteration cap is 10, not the spec's 5.** Real BFRPG combat rounds
   chain 6-8 tool calls (to-hit, damage, apply_damage, save, monster's counter,
   …). 5 false-positives on legitimate play. 10 is the calibrated value; if a
   turn legitimately needs more we're modelling combat wrong (the LLM should
   pause for the player rather than driving the round end-to-end). Constant in
   `app/orchestrator/dm.py`. Phase 4 prep #1 added a PACING block to the role
   text that strongly improves player handoffs; the cap stays at 10 because
   real-traffic measurement showed 7 trips iteration_cap and 8 trips
   empty_completion under run-to-run Nemotron variance — both noisier than
   leaving the cap headroom.

8. **Initiative gating is server-side, never client-trusted.** During an active
   encounter, combat-kind `pc_action` messages must come from the player whose
   character holds the current initiative slot; non-current actors get a
   typed `not_your_turn` `dm_error`. The check lives in `app/api/ws.py`'s
   `_check_initiative_gate`; clients may render a visual "your turn" hint
   from `current_actor` in the snapshot, but bypassing the hint and sending
   a combat action anyway still gets rejected. Non-combat kinds (talk, look,
   other) are unconditionally accepted regardless of initiative.

9. **Whisper audience filtering happens at the WS broadcast layer, not at
   storage.** `session_messages` always stores the full whisper content with
   `audience=[character_id]` so the DM's prompt history stays consistent
   across turns. The filter is in the WS subscriber task and the snapshot
   builder — both check the receiving user's character ids against the
   audience before letting a frame land on a socket. Never redact at write
   time; you'll lose context the DM needs for consistency on later turns.

## Tech stack

- Python 3.12
- FastAPI (async)
- SQLAlchemy 2.x async + Alembic
- SQLite 3.45+ (WAL)
- Valkey 7+ (Redis-compatible fork; pub/sub + queue for image worker)
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
- Handlers that depend on `require_user` (or any dependency that reads from the DB)
  use `db.add(...)` + `await db.commit()`, **not** `async with db.begin():`. The
  user-resolution dependency autobegins a read transaction during DI; an explicit
  `begin()` in the handler body collides with "transaction already begun". Use
  the explicit-begin pattern only in handlers that don't read from the DB before
  writing — currently rare; see `app/orchestrator/dm.py` for the canonical
  exception (it manages its own session lifecycle outside the DI chain).

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

## Current build phase

**Phase 4 complete; Phase 5 (image generation) ready to start.**

Update this line as phases complete. The phased plan is in spec §14.

## Follow-ups

Parking lot for items deliberately deferred. Each entry is a short,
actionable note; the trigger condition tells future-you when to pick it
up. Promote to a real issue / phase task when its trigger fires.

- **Reasoning mode tuning** (added 2026-04-30, target Phase 5).
  Apply Nemotron's `low_effort` reasoning mode to memory subsystem
  calls (session summariser, campaign summariser, fact extractor)
  while keeping the DM turn loop at full reasoning. Mechanical change
  to `app/llm/client.py` adding a `reasoning_mode` parameter.
  **Verify:** `low_effort` doesn't degrade JSON output on the fact
  extractor — if it does, keep the extractor at full and only the
  summarisers low. Phase 5 is the right home because it's where the
  project first cares seriously about latency budgets across async
  workloads. **Trigger:** start of Phase 5 image-generation work
  (latency on memory tasks competes with image-worker priority).
  **Context:** discussion of `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
  reasoning modes in the Phase 3 conversation.

- **Per-class spell levels** (added 2026-04-30, target: when it bites).
  Spells like Hold Person and Continual Light have different levels per
  caster class (MU3 vs Cleric2). Current schema requires duplicate
  records — `data/bfrpg/spells.yaml` ships ~6 affected spells as two
  entries each, with a disambiguating `(Magic-User)` suffix on the
  secondary entry. Real fix: `castings: list[{caster_class, level}]`
  on the spell schema. **Trigger:** a player actually casts one of the
  affected spells in play and the rules engine returns a wrong outcome,
  OR a Phase 6 spell-prep UI surfaces the duplication. Premature
  schema work otherwise. **Context:** Phase 2 finding #6.

- **Production embedding endpoint** (added 2026-04-30, target Phase 7
  hardening / production deploy). Default backend is local
  `sentence-transformers` with `BAAI/bge-large-en-v1.5` (1024-dim).
  Pulls torch as a transitive dep (~2GB), loads ~1.5GB of weights into
  RAM. Production should set `EMBEDDING_BASE_URL` pointing at Ollama
  once an operator runs `ollama pull <embedding-model>` on
  `svrai01:11436` (currently no embedding model is loaded there;
  `nomic-embed-text` is 768-dim, `bge-large` is 1024-dim — match
  `embedding_dim`). After the swap, drop `sentence-transformers` from
  required deps to optional. **Trigger:** start of Phase 7 hardening,
  OR an operator schedules a deploy.
  **Context:** Phase 3 prep probes; commit `64d690e`.

- **Cross-worker shared state** (added 2026-04-30, expanded 2026-04-30
  for Phase 4, target Phase 7+ scale-out). Three pieces of process-local
  state would race under multi-worker scale-out: the `WorldFactRetriever`
  cache, the per-session / per-campaign summariser locks
  (`_session_summary_locks`, `_campaign_summary_locks`), and the new
  Phase 4 surfaces — `_session_turn_locks` in `app/api/ws.py` (orchestrator
  serialisation per session) and the `PresenceRegistry` singleton in
  `app/realtime/presence.py`. Spec §13's single-gunicorn-worker deployment
  makes all four correct today. Multi-worker would need: (a) Valkey-backed
  distributed lock keyed by `session_id` for the orchestrator gate;
  (b) Valkey pub/sub fan-out for presence so a worker advertising "alice
  joined" reaches every other worker; (c) cache invalidation broadcast
  for the world-fact retriever. **Trigger:** spec gets a multi-worker
  deployment story (no such plans currently), OR Phase 7+ load testing
  shows the single worker is the bottleneck.
  **Context:** Phase 3 commit `64d690e`, Phase 4 commits
  `327a9c7`/`bd9ba9c`; spec §13 single-worker rationale.

- **SSE bridge removal** (added 2026-04-30, target Phase 5 prep or
  end-of-Phase-5 cleanup). Phase 4 made `app/api/sse.py` and
  `tests/test_sse_bridge.py` orphaned — WS hub is now the primary
  transport. Remove only after confirming Phase 5's image worker
  doesn't reuse the bridge for any streaming concern (the worker
  publishes via Valkey pub/sub which the WS hub consumes directly,
  so SSE shouldn't matter). Verify with grep + integration test
  rerun before deletion. **Trigger:** at the start of Phase 5, the
  agent picks this up as a 10-minute prep task.
  **Context:** Phase 2 commit `40e96af` (introduced); Phase 4 commit
  `13d9b60` (orphaned).

- **Multi-tab cross-visibility for the same user** (added 2026-04-30,
  target Phase 6). When a user opens two browser tabs of the same
  session and submits an action from tab A, tab B doesn't see it.
  Cause: `pc_action` frames echo back via Valkey to every connection,
  and the JS de-dupes by `selfUserId === msg.user_id`. The fix needs
  a per-connection identifier — easiest is to have the server filter
  the originating socket out of the broadcast (currently the publisher
  doesn't know which sockets it came from). Acceptable for Phase 4
  (real multi-tab is rare and not blocking gameplay). **Trigger:**
  a player reports it, OR Phase 6 polish surfaces it as a confused
  user moment. **Context:** `app/templates/table.html` `pc_action`
  dispatcher; Phase 4 commit `13d9b60`.

- **Multi-client integration test** (added 2026-04-30, target Phase 7
  hardening). `tests/integration/test_multiplayer.py` is single-client
  because two `TestClient` instances each hold their own anyio
  `BlockingPortal` with its own event loop, and the Pubsub singleton's
  redis-py async client binds to whichever loop first built it.
  Sharing the singleton across portals fires "Future attached to a
  different loop" and breaks the test. The unit suite covers the
  multi-client semantics deterministically with `FakePubsub`; for
  end-to-end multi-client coverage on the real stack, spin up a real
  uvicorn server in a thread and connect via the `websockets` library
  from the test's main asyncio loop. **Trigger:** Phase 7 hardening,
  OR a regression that the unit suite misses but a multi-client real
  run would catch. **Context:** Phase 4 commit `e89bcbc`; module
  docstring of `tests/integration/test_multiplayer.py`.

- **`:memory-sentinel:` SQLite artefacts in repo root** (added 2026-04-30,
  target whenever annoying). `tests/conftest.py` sets
  `DB_PATH=:memory-sentinel:` so production code that reads the env
  before fixtures override the engine writes a literal file named
  `:memory-sentinel:` (plus `-wal` / `-shm`) into the CWD. Pre-existing
  in Phase 2; got noticed during Phase 4. The fixtures clean up their
  own engines but the production-default-path leak does happen. Three
  acceptable fixes: (a) set the env override to a `tmp_path`-derived
  per-test path; (b) gate the production engine creation behind a
  lazy factory so tests never trigger it; (c) `.gitignore` the
  sentinel filenames so they're invisible. Don't deserve their own
  PR. **Trigger:** anyone cares.
  **Context:** Phase 4 step-4 close-out.

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
