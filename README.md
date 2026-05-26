# Dungeon Master

A self-hosted web app where 2–4 humans play Basic Fantasy RPG with an LLM
Dungeon Master narrating, adjudicating, and running encounters. AI-generated
scene images. Persistent campaigns with long-term memory. Reusable adventure
modules.

Deployed on a single AlmaLinux 10.1 host on a trusted internal LAN.

- **LLM:** Nemotron 3 Super via vLLM
- **Image generation:** FLUX.1 [dev] + FLUX.1 Kontext [dev]
- **Database:** SQLite 3 (WAL)
- **Frontend:** Jinja2 + HTMX 2 + Alpine.js

The full design lives in [`dungeon-master-spec.md`](dungeon-master-spec.md).
A working-document TL;DR for agents lives in [`AGENTS.md`](AGENTS.md).

## Status

Phase 8 (Adventure modules) **complete**. See `dungeon-master-spec.md` §14 for the phased plan.

## Repository layout

See `AGENTS.md` "File organisation".

## Adventure modules

Dungeon Master ships with the **Morgansfort** module — a BFRPG adventure
for levels 1–3 (estimated 6 sessions). Three additional workflows let you
manage modules:

### Load a bundled module

Register a module JSON from `data/bfrpg/modules/` as a playable Module row.
Requires an admin user to exist first (see §13 of the spec for bootstrap).
The command is idempotent — safe to run on every server start.

```bash
uv run python -m app.scripts.load_module morgansfort
# OK: registered module 'Morgansfort' (id=..., 19 locations, 26 NPCs, 5 beats)
# (second run) SKIP: module 'Morgansfort' already registered (id=...)
```

### Start a campaign from a module

Once a Module row exists, load it into a new campaign via the API. This
mints UUIDs for every location, NPC, encounter, beat, and secret;
populates `module_state` with all beats pending; and enqueues NPC
portrait images.

```bash
curl -s -X POST http://localhost:8000/api/campaigns/from-module \
  -H "Content-Type: application/json" \
  -d '{"module_id": "<uuid>", "name": "My Morgansfort Run"}' | jq
# {"campaign_id": "...", "locations_created": 19, "npcs_created": 26, ...}
```

Optional `image_style_override` replaces the module's default art style for
this campaign.

### Extract a module from a played session

After a session ends, the DM can crystallise the improvised content into a
reusable module JSON. The endpoint replays the session transcript through
the LLM and validates the output against `ModuleContent`.

```bash
curl -s -X POST http://localhost:8000/api/sessions/<session_id>/extract-module | jq
# {"module_id": "...", "synopsis": "...", "locations": 4, "npcs": 7, ...}
```

The resulting Module row has `public=false`. Use `POST
/api/campaigns/from-module` to load it into a fresh campaign.

### Author a module

Module files are plain JSON validated against the `ModuleContent` schema
(`app/llm/modules.py`). Key rules:

- Every entity needs a typed `symbol` (prefixes: `loc_`, `npc_`, `enc_`,
  `beat_`, `sec_`, `end_`). Symbols are the only cross-reference mechanism;
  UUIDs are assigned at load time.
- `trigger_hint` on plot beats and `reveal_when` on secrets are
  natural-language DM guidance, not mechanical conditions — they appear
  verbatim in the system prompt.
- Validate before loading:

```bash
uv run python -c "
import json, sys
from app.llm.modules import ModuleContent
ModuleContent.model_validate(json.load(open(sys.argv[1])))
print('OK')
" data/bfrpg/modules/morgansfort.json
```

## Quickstart (dev)

Requirements: Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies and materialise the venv.
uv sync

# 2. Tell the app where to put the database. The default
#    (/var/lib/dungeon-master/dm.db) matches the production layout in
#    spec §13 but is unwritable in dev — point it at a local file.
export DB_PATH="$PWD/dev.db"

# 3. Apply migrations. Phase 0 ships only an empty base migration; Phase 1
#    autogenerate will chain off it.
uv run alembic upgrade head

# 4. Start the server.
uv run uvicorn app.main:app --reload
```

Then:

```bash
curl http://127.0.0.1:8000/health
# → {"status":"ok","db":"ok"}

open http://127.0.0.1:8000/
# Renders the "Dungeon Master — bootstrapping" page with the version.
```

Quality gates:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy app
uv run pytest
```

For production deployment on AlmaLinux 10.1, see `deploy/bootstrap.sh`
and the systemd / nginx artefacts alongside it.

## License

MIT — see [`LICENSE`](LICENSE).
