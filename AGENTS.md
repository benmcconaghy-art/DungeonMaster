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

10. **Image worker uses /generate as its watchdog probe, not /health.** The
    FLUX service at svrai01:11437 has been observed returning 200 OK from
    /health while /generate returns 500 (a non-FLUX process held VRAM
    above the threshold). Watchdog liveness is a 256x256/1-step
    /generate every 30s — a real inference is the only definitive signal.
    Sustained failure (any 5xx, transport, or malformed response) past
    DEGRADED_THRESHOLD_S flips `image:status` to "degraded"; first success
    flips back. The status key is the rendezvous between the imageworker
    process and the FastAPI process.

11. **Image-job side effects commit before the WS publish.** The worker's
    `_persist_and_link` writes the `generated_images` row + any
    `canonical_image_id` FK update inside a single transaction; only after
    commit does it publish `image_ready`. A subscriber that reads the DB on
    receipt always finds the row. This is the same transaction discipline as
    invariant #2 but at the worker boundary — never publish before commit.

12. **Portrait dispatch is `kind=npc` regardless of subject.** PCs and NPCs
    both use the npc parameter set (768x1024/32-step) per spec §8 — they
    are both single-figure portraits. The distinction is only in the FK
    target: `subject_character_id` vs `subject_npc_id`. The `enqueue_portrait`
    helper rejects setting both. A regression that sent `kind=scene` for
    portraits would silently produce wide landscape renders.

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
- **Don't use slowapi's `@limiter.limit` decorator** for rate limiting in this
  codebase. It silently returns 422 on every request when combined with
  `from __future__ import annotations` (which we use throughout). slowapi's
  wrapper rebinds `__globals__` to its own module, so FastAPI's
  `evaluate_forwardref(annotation, globalns, globalns)` can't resolve string
  annotations like `RegisterRequest`/`DbSession` — body and dep parameters
  silently degrade to query params. Drive the `limits` library directly via
  FastAPI `Depends` instead. See `app/ratelimit.py` for the pattern.

### Operational scripts (systemd-driven, watchdogs, cron)
- **Any systemd-driven script in this codebase MUST exit 0
  unconditionally**, even on internal failure. Failures route through
  the alerts hook (`deploy/alerts/notify.sh`), not through exit codes.
  Reason: a non-zero exit causes systemd to mark the `.service` Failed
  and stop further invocation by the timer — the watchdog goes silent
  exactly when it's needed most. The cron-driven backup script is the
  one principled exception (cron-monitoring tools want a non-zero exit
  when the snapshot itself failed); even there, *retention* failures
  log + alert + return 0 so a slightly-too-old backup beats a missed
  next-night run. Phase 7 watchdogs in `deploy/watchdogs/` follow this
  invariant; any Phase 8+ timer-driven work should too.
- State-bearing watchdogs persist their last-tick state under
  `/var/lib/dungeon-master/watchdog-state/<name>` so transitions can
  be detected without spawning a long-running daemon. Alert on
  *transition*, not on every tick — an always-degraded condition
  would otherwise spam the alert log.
- Watchdog scripts read configuration from environment variables with
  sensible defaults (`STATE_FILE`, `ALERT_HOOK`, threshold values) so
  unit-testing them locally with stubbed paths and a mock alert hook
  doesn't require root or a real systemd.

### Tests
- Each new module gets a parallel `tests/test_<module>.py` from day one.
- pytest + pytest-asyncio. `@pytest.mark.asyncio` on async tests.
- Mock `vLLM` and `FLUX` clients at the boundary. Never call real services in unit tests.
- Use in-memory SQLite (`sqlite+aiosqlite:///:memory:`) for fast suites; file-backed only for migration tests.
- Inject `random.Random(seed)` into rules-engine functions for reproducibility. No module-level `random.*` calls in production code.
- Rate-limit tests must exercise the real storage backend the
  application uses, not mocked storage. Phase 7 playthrough surfaced
  two bugs (coredis prerequisite missing, then coredis pool not
  initialised in async context) that the original test suite missed
  because tests overrode the storage URI to `memory://` while
  production used `redis://`. Pin the production URI in tests; only
  monkeypatch the *limit thresholds* to keep cases fast. See
  `tests/test_ratelimit.py::test_production_storage_backend_handles_full_lifecycle`
  for the canonical pattern.

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
│   ├── client.py       # FLUX HTTP client (/health, /probe, /generate, /edit)
│   ├── health.py       # image:status Valkey rendezvous (worker writes, orchestrator reads)
│   ├── portrait.py     # prompt composer + enqueue helpers + queue-client singleton
│   ├── queue.py        # ImageJob model + push/pop helpers
│   └── worker.py       # async queue consumer + watchdog
├── realtime/
│   ├── hub.py          # WebSocket session hub
│   ├── messages.py     # WS message types
│   └── pubsub.py       # Redis pub/sub
├── orchestrator/
│   ├── dm.py           # the DM turn loop
│   └── handlers/       # one file per tool: apply_damage.py, award_xp.py, etc.
├── views/              # server-side template-context builders
│                       # (one composer per non-trivial view)
├── templates/          # Jinja2 (base + per-view: index/login/register/
│                       # campaign_dashboard/character_sheet/table)
└── static/             # CSS (per-view: tokens/base/auth/dashboard/sheet/
                        # table), JS, images

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

**Phase 7 complete; Phase 6.5 (chargen UI) landed 2026-05-01 to
unblock real play; Phase 8 (Adventure modules) ready to start.**

Update this line as phases complete. The phased plan is in spec §14.

Phase 7 deploy-readiness — the pieces that only meaningfully verify
against a real production deploy (SELinux enforcing, watchdog timer
drills, backup integrity on a real DB, restore-procedure timing,
cron firing on schedule, /metrics nginx restriction) — are
captured in `deploy/PHASE_7_VERIFICATION.md` as a runbook for
the first production stand-up. None block Phase 8.

## Follow-ups

Parking lot for items deliberately deferred. Each entry is a short,
actionable note; the trigger condition tells future-you when to pick it
up. Promote to a real issue / phase task when its trigger fires.

- **Reasoning mode tuning** *(resolved 2026-05-01, Phase 5 prep #2)*.
  `app/llm/client.py` gained a `reasoning_mode` parameter that maps
  to Nemotron's `chat_template_kwargs`. Verdict per call site:
  **summarisers → "low"** (compression IS structuring; JSON shape
  stays clean), **fact extractor → "full"** (kept after a 4-run
  integration study: 3× "low" → 21/38/51 facts vs 1× "full" → 128
  facts on the same 25-turn fixture; "what merits long-term memory"
  is salience judgement, not structuring). Latency cost at "full":
  ~+8s/turn extractor delta (mean turn 8.6s → 18.6s on the same
  fixture). DM turn loop stays "full" for tool-call accuracy. Phase
  8 module extractor will start at "full" for the same salience
  reason. **Future watch:** if a multi-session campaign shows
  campaign-summary drift (the "low" summariser missing arcs), revisit
  promoting the campaign summariser to "full" too.

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

- **SSE bridge removal** *(resolved before Phase 6)*. Verified during
  Phase 6 Step 3 prep that `app/api/sse.py` and
  `tests/test_sse_bridge.py` were both already removed (presumably
  during Phase 5 step prep). No action needed at Phase 6 close.

- **Multi-tab cross-visibility for the same user** (added 2026-04-30,
  re-evaluated 2026-05-01 at Phase 6 close, target Phase 7+).
  When a user opens two browser tabs of the same session and submits
  an action from tab A, tab B doesn't see it. Cause: `pc_action`
  frames echo back via Valkey to every connection, and the JS
  de-dupes by `selfUserId === msg.user_id`. The fix needs a
  per-connection identifier — easiest is to have the server filter
  the originating socket out of the broadcast (currently the
  publisher doesn't know which sockets it came from). **Phase 6
  re-evaluation**: deferred again. The Phase 6 design's WHISPERS
  sidebar gives a natural future surface for "your other tab acted"
  notifications, but the underlying server-side filtering is real
  architectural work, not polish. Whoever picks this up should
  consider piggy-backing on the same per-conn identifier needed for
  the multi-client integration test below. **Trigger:** a player
  reports it, OR Phase 7+ multi-client load testing motivates the
  conn-id surface. **Context:** `app/templates/table.html`
  `pc_action` dispatcher; Phase 4 commit `13d9b60`; Phase 6
  WHISPERS panel as the natural notification surface.

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

- **Kontext /edit output dimensions vs scene-kind params** (added 2026-05-01,
  Phase 5 step 8 close-out, re-evaluated 2026-05-01 at Phase 6 close).
  Kontext `/edit` preserves the source image's aspect ratio — a
  scene-edit job that references an NPC's canonical portrait
  (768x1024) produces a 768x1024 image, not the scene-kind
  1280x768. The worker today persists `width=null, height=null` for
  /edit jobs because we don't know the output size until decoded.
  **Phase 6 polish-pass observation**: the design's image-card frame
  (parchment plate with `width: 280px; height: auto;` for the img,
  scaled to 220px at ≤1280px) renders portrait-shaped images at
  ~280×373px without breaking the float-and-letterbox layout. The
  inconsistency between landscape `/generate` scenes (~280×168) and
  portrait `/edit` scenes (~280×373) in the same column is real but
  not visually broken — the frame composition is the same, the
  plate just gets taller. Real fix when it matters: (a) decode the
  PNG in the worker to fill width/height post-hoc; (b) add an
  explicit `target_width` / `target_height` to the edit request so
  Kontext outputs scene-shaped; (c) crop/pad to scene aspect on the
  FastAPI side before serving. **Trigger:** a player or playtest
  comments on the inconsistent aspect (the visual layer no longer
  hides this — readers see the size delta).
  **Context:** spec §8 line 766; Phase 6 step 4 image-card
  composition decision in `app/static/css/table.css`.

- **Image events not in Snapshot.messages** *(resolved 2026-05-01,
  Phase 6 step 3)*. Migration `78fa9cf6ec1a` added a nullable
  `session_id` column to `generated_images`. The worker writes it
  from `ImageJob.session_id`. `_build_snapshot` queries the session's
  recent images and returns them on a parallel `image_events` list
  on `Snapshot`. The frontend snapshot handler interleaves messages
  and image events by `created_at` so the rebuilt log matches the
  chronology a steady-state viewer saw.
  **Remaining narrow gap**: only `ready` events have rows. Pending
  lives on the queue, failed doesn't get a row, so a reconnect during
  the brief in-flight window still relies on the live `image_ready` /
  `image_failed` to land for that one slot. The `status` field stays
  on the wire shape for forward compatibility — a future
  `session_image_events` table could expand the set if it bites.

- **FLUX cold-load measurement vs spec estimate** (added 2026-05-01,
  Phase 5 step 6 close-out). Spec §8 estimated "cold pipeline load
  ~15-30s plus 28-step inference ~8-18s, ~25-45s end-to-end". Live
  measurement on `svrai01:11437` against FLUX.1 [dev] on a 5090:
  256x256/1-step = 4.85s, 1280x768/28-steps cold = 16.95s, warm =
  17.02s (no cold-load tax detected — model stayed resident or there
  is no real cold-load on warm hardware). The service-reported
  `generation_time_seconds` (12.0s for the 28-step run) understates
  wall-clock by ~5s, so it's not load+inference split — it's
  inference-only with the rest in encoding/transport. **Implication:**
  the 180s read timeout in `app/images/client.py` has plenty of headroom;
  the 60s probe timeout is generous; the watchdog's 30s tick interval
  is fine. **Watch:** a different GPU or a degraded VRAM state could
  swing this 3-5x. If portrait/scene latency complaints surface,
  capture timing again before tuning. **Context:** commits
  `51d09b4`/`c520651` and the post-Step 6 measurement run.

- **GPU squatter ops watch** (added 2026-05-01, ongoing). Phase 5 Step
  6 close-out: a 7GB unrelated process squatting on the 5090 pushed
  FLUX over VRAM. /health stayed 200 OK with `flux_txt2img_loaded:false`
  while /generate returned 500. The watchdog now uses a deep
  /generate probe (256x256/1-step) so this state surfaces as
  "degraded" within ~120s. **Operator runbook:** if the worker logs
  show `image:status -> degraded` and the FLUX service is reachable
  on /health, run `nvidia-smi` on `svrai01` and look for non-FLUX
  PIDs holding VRAM. **Trigger:** add this to deploy docs the first
  time anyone writes them. **Context:** Step 6 commits, the
  `_watchdog` docstring in `app/images/worker.py`.

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

- **Chargen UI does not exist** *(resolved 2026-05-01, Phase 6.5)*.
  Built as a deferred Phase 6 piece that landed before Phase 8
  because real play blocked on it (the curl-only path was
  unsustainable for a fresh campaign). Single page at
  `/campaigns/{id}/chargen` with progressive reveal: abilities
  (3d6 / 4d6kh3 toggle, re-roll with settle animation) → heritage
  (eligibility filtered by race ability requirements, ineligible
  cards dim with their failed minimum) → calling (filtered to
  `race.allowed_classes`) → alignment → name → commit. New endpoint
  `POST /api/chargen/roll-abilities` is the only API addition; the
  commit reuses the existing `POST /api/campaigns/{id}/characters`.
  Eligibility is a JS data-table lookup, not a BFRPG mechanic, so
  no rules surface in the frontend; server-side `generate_character`
  remains authoritative on commit. Files:
  `app/templates/chargen.html`, `app/static/css/chargen.css`,
  `app/views/chargen.py`, `app/api/chargen.py`,
  `tests/api/test_chargen.py`. Dashboard `.char-roll` hrefs now
  point at `/campaigns/{id}/chargen`.

- **Dashboard list-campaigns N+1 queries** (added 2026-05-01, Phase 6
  close-out, target if it ever bites). `app/api/campaigns.list_campaigns`
  and the parallel `app/views/dashboard._list_campaigns_for` issue a
  per-campaign `_campaign_last_played` query inside a loop. Acceptable
  at the spec's scale (2-4 players, low-tens of campaigns per user)
  but unbounded as the campaign count grows. Fix: a single grouped
  query with `MAX(coalesce(ended_at, started_at)) GROUP BY
  campaign_id` joined back to the campaigns rows. **Trigger:** the
  dashboard feels slow on a campaign-rich account, OR Phase 7
  load-testing flags it.
  **Context:** `app/api/campaigns.py` and `app/views/dashboard.py`
  share the same loop pattern.

- **Spell-slot tracking separate from prepared/known**
  (added 2026-05-01, Phase 6 close-out, target when spell prep UI
  lands). The character sheet renders prepared spells as filled
  ember pips and unprepared/known as hollow pips, derived from the
  `spells_known.prepared` boolean. There's no separate notion of
  spent vs ready slots within a level — the design's "slots" line
  in the spell-level header is currently empty in the implementation.
  When spell-prep UI lands, add a `spent_slots` counter (probably
  on `characters.sheet` JSON or a new `prepared_slots` table) so the
  sheet can render `slots ●●○` for "two ready, one spent at lvl 1".
  **Trigger:** spell-prep UI, OR a player asks why slots don't
  decrement after casting.
  **Context:** `app/templates/character_sheet.html` spell-level
  header; `app/db/models.py` `SpellKnown`.

- **Inventory `item_type` is a string** (added 2026-05-01, Phase 6
  close-out, target when item editing lands). The character sheet's
  WIELDED / OFF-HAND / WORN loadout strip filters by
  `item.item_type == "weapon" / "shield" / "armor"`. Those are
  strings on the model with no enum or CHECK constraint. New
  inventory features (drag-drop equip, item editing, item creation
  from chargen) should consolidate the type vocabulary or accept
  the strings as is and document them.
  **Trigger:** an inventory editing surface lands, OR a chargen
  flow auto-populates inventory and produces a typo.
  **Context:** `app/db/models.py` `InventoryItem.item_type`;
  `app/templates/character_sheet.html` loadout filter.

- **Invite-code audit / revocation** *(resolved 2026-05-01,
  Phase 7 step 3B)*. Promoted to a row-backed surface
  (`campaign_invites` table, single-use semantics, owner-only
  GET-list and DELETE-revoke endpoints, 7-day legacy grace for
  Phase 6 stateless tokens until 2026-05-08). Migration
  `a96d3a6e501d`. Tests in `tests/test_invite_revocation.py`.

- **SELinux confined domain not implemented** (added 2026-05-01,
  Phase 7 step 3C, target if posture changes). The Phase 7
  policy module (`deploy/selinux/dungeon-master.te`) lets the
  service run as `unconfined_service_t` — the targeted-policy
  default for systemd-managed services without an explicit
  transition. The .fc file labels the data / log / config paths
  correctly; the install script handles port 8001 + the
  `httpd_can_network_connect` boolean + restorecon. What's
  shipped is SELinux-aware deployment, not SELinux-confined:
  the service has wide access via `unconfined_service_t`
  rather than narrow allow rules in a custom domain. Real
  confinement would mean a `dungeon_master_t` domain with an
  `init_t` → `dungeon_master_t` transition and explicit allow
  rules narrowed to what the service actually uses. ~2-3 hours
  of policy-engineering work when triggered. **Trigger:** if
  the deployment posture changes from trusted-LAN to anything
  with public-internet exposure, multi-tenant deployment, or a
  regulatory pressure (e.g. compliance audit). **Context:**
  `deploy/selinux/dungeon-master.te` design notes; the README
  in that directory has the "when to introduce a confined
  domain" section.

- **Phase 7 deploy-readiness checklist** (added 2026-05-01,
  Phase 7 close-out, target first production stand-up). Several
  Phase 7 pieces only meaningfully verify against a real deploy
  box and were not run from the dev sandbox: SELinux policy
  under enforcing mode against the running service, watchdog
  timer drills (stop FLUX / fill the data dir / runaway log),
  backup integrity on a production-shaped DB (the unit tests
  stub `sqlite3`), `RESTORE.md` cold-operator drill timing,
  cron actually firing at 02:00/03:00, and the `/metrics`
  nginx-restriction returning 403 from a non-localhost peer.
  All six are spelled out as a runbook in
  `deploy/PHASE_7_VERIFICATION.md`. None block Phase 8 work,
  but the checklist needs to pass before the deploy is
  considered Phase 7-verified rather than just Phase 7-complete.
  **Trigger:** first production stand-up, OR any deploy-box
  smoke after a hardening change. **Context:**
  `deploy/PHASE_7_VERIFICATION.md`.

- **Dev-default `session_secret` not rejected at boot** (added
  2026-05-01, Phase 7 step 3D, target if posture changes from
  trusted-LAN). `app/config.py` declares
  `session_secret: str = Field(default="dev-only-not-secret-replace-in-production",
  min_length=16)`. The 38-char placeholder satisfies the length
  constraint, so a deploy that somehow skipped bootstrap.sh
  (which writes a real openssl-generated secret to
  `/etc/dungeon-master/env`) would silently use the placeholder
  in production. On the trusted-LAN deployment this is benign:
  bootstrap.sh always writes the env file and the systemd unit
  always reads it, and even if the dev secret leaks the only
  thing it forges is a session cookie on a network nobody else
  can reach. Real fix when triggered: add a startup check that
  rejects the dev-default placeholder unless an explicit
  `DM_ALLOW_DEV_SECRET=1` opt-out is set. **Trigger:** if the
  deployment posture changes from trusted-LAN to anything with
  public-internet exposure. **Context:** `app/config.py:57`;
  `deploy/bootstrap.sh:157` (where the real secret is generated).

## Working with Claude Design handoffs

Phase 6 (UX polish) consumed a Claude Design handoff bundle —
HTML/CSS/JS prototypes, design tokens, and a chat transcript
showing the iteration history. The pattern worked well; record it
here for future UI-heavy phases:

1. **Read the chat transcript first.** The transcripts live in the
   bundle's `chats/` directory and tell you what the user actually
   asked for, where they landed after iterations, and which file
   was the last version. The HTML files are the output; the chat
   is where the intent lives. Skip this and you'll re-litigate
   decisions the user already settled.

2. **Inventory before translating.** For each design HTML file in
   the bundle: which view, what viewport widths it covers, what
   variants (combat vs exploration, alive vs dying, empty state).
   Cross-reference against existing templates so you know which
   are translations vs creations vs extrapolations.

3. **Tokens.css first.** The design's tokens.css is the single
   source of palette, typography, spacing, and shared idioms
   (chips, plate frames, cap-tabs, grain texture). Land it as
   `app/static/css/tokens.css` verbatim; every per-view
   stylesheet then references its custom properties without
   re-declaring them.

4. **Per-view stylesheets, not one giant CSS file.** Mirror the
   handoff's structure: tokens.css + base.css for shared chrome +
   one CSS per view. Keeps maintainership tight; matches the
   convention in `.claude/agents/frontend.md`.

5. **Preserve every WS/HTMX/Alpine binding during translation.**
   The design rearranges DOM; the bindings need to follow.
   Integration tests are the contract — if they pass after the
   visual translation, behaviour was preserved.

6. **Don't port the prototype's JS.** The design files include
   demo interactivity (key bindings to toggle states, etc.) that
   exists only to let the user evaluate the design. The
   production layer uses the existing WS dispatcher / Alpine
   bindings / HTMX swaps. The chat transcript usually flags this
   explicitly.

7. **Extrapolation is on you.** Auth views, base layout shells,
   empty states, error states — the design rarely covers every
   surface. Match the established voice (see Phase 6
   `register.html` blurb for the "world-aware, three-rolls-and-
   a-name" register) and document any non-obvious decisions
   inline (Phase 6 image-card narrow-width comment in
   `app/static/css/table.css` is the canonical example).

8. **Acknowledge what didn't make it.** The design might imply
   functionality that doesn't exist (Phase 6 dashboard's "Roll a
   new character" link points at a chargen UI that's still API-
   only). Add a Follow-up entry rather than building the missing
   piece in the polish phase — keeps phase scope honest.

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
