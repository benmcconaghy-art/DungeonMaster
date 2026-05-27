# Dungeon Master

A self-hosted web app for playing **Basic Fantasy RPG** with an AI dungeon master. 2–4 players connect from their browsers; the LLM narrates, adjudicates rules, runs encounters, and generates scene images in real time. Campaigns persist between sessions with long-term memory. Reusable adventure modules let you drop into pre-built worlds or build your own.

Built for a single trusted household LAN — no cloud dependencies, no subscriptions.

---

## Features

- **AI dungeon master** — Nemotron 3 Super via vLLM; narrates in a consistent world-voice, tracks plot beats, reveals secrets at the right moment
- **Live play table** — WebSocket-driven; all players see narration, dice rolls, and images as they happen
- **Character system** — full BFRPG chargen (3d6 in order or heroic 4d6kh3), saving throws, inventory, spells known; characters persist in a roster between campaigns
- **Adventure modules** — five bundled CC BY-SA modules (Morgansfort, Chaotic Caves, Fortress Tomb Tower, Palace of the Vampire Queen, Rock Hollow); campaign creation seeds locations, NPCs, world facts, and plot state in one transaction
- **AI-generated portraits and scene images** — FLUX.1 [dev] / FLUX.1 Kontext; per-campaign art style; image dedup by prompt hash
- **Long-term memory** — world facts embedded with `bge-large-en-v1.5`; top-k semantic retrieval injected into the DM prompt each turn
- **Campaign dashboard** — character roster, recent sessions, party setup flow, module picker
- **Single-host deployment** — SQLite WAL, Redis pub/sub, systemd + nginx on AlmaLinux 10.1

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | FastAPI + SQLAlchemy (async) + Alembic |
| Database | SQLite 3 (WAL mode) |
| Frontend | Jinja2 server-rendered + vanilla JS (no build step) |
| LLM | Nemotron 3 Super via vLLM (OpenAI-compatible) |
| Image gen | FLUX.1 [dev] + FLUX.1 Kontext via internal endpoint |
| Embeddings | `bge-large-en-v1.5` via sentence-transformers or any `/v1/embeddings` endpoint |
| Queue | Redis pub/sub (image jobs + WebSocket fan-out) |
| Auth | bcrypt + signed session cookies (itsdangerous) |

---

## Quickstart (dev)

Requirements: Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Point the app at a local database (default is /var/lib/dungeon-master/dm.db)
export DB_PATH="$PWD/dev.db"

# 3. Run migrations
uv run alembic upgrade head

# 4. Start the server
uv run uvicorn app.main:app --reload
```

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","db":"ok"}
```

Register a user at `http://127.0.0.1:8000/register`, then promote them to admin so you can load modules:

```bash
sqlite3 dev.db "UPDATE users SET is_admin=1 WHERE username='yourname';"
```

Load the bundled adventure modules:

```bash
for m in morgansfort chaotic_caves fortress_tomb_tower palace_vampire_queen rock_hollow; do
    uv run python -m app.scripts.load_module $m
done
```

---

## Configuration

All settings are read from environment variables (or a `.env` file at the repo root). The defaults match the production layout.

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `/var/lib/dungeon-master/dm.db` | SQLite database path |
| `IMAGE_STORAGE_PATH` | `/var/lib/dungeon-master/images` | Directory for generated PNGs |
| `VLLM_BASE_URL` | `http://svrai01…:8000` | vLLM endpoint (OpenAI-compatible) |
| `FLUX_BASE_URL` | `http://svrai01…:11437` | Image generation endpoint |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis connection |
| `SESSION_SECRET` | *(dev placeholder)* | Cookie signing secret — **change in production** |
| `EMBEDDING_BASE_URL` | *(unset)* | Optional `/v1/embeddings` endpoint; falls back to local sentence-transformers |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model (1024-dim) |

---

## Adventure modules

Five modules ship in `data/bfrpg/modules/`, all released under CC BY-SA:

| Module | Levels | Sessions |
|---|---|---|
| Morgansfort | 1–3 | ~6 |
| Chaotic Caves | 1–3 | ~4 |
| Fortress Tomb Tower | 2–4 | ~5 |
| Palace of the Vampire Queen | 5–8 | ~5 |
| Rock Hollow | 1–2 | ~3 |

### Loading a module

```bash
uv run python -m app.scripts.load_module morgansfort
# OK: registered module 'Morgansfort' (id=..., 19 locations, 26 NPCs, 5 beats)
```

The command is idempotent — safe to run on every server start.

### Authoring a module

Modules are plain JSON validated against the `ModuleContent` schema (`app/llm/modules.py`). Every entity needs a typed symbol (`loc_`, `npc_`, `enc_`, `beat_`, `sec_`, `end_`) — UUIDs are assigned at load time. Validate before loading:

```bash
uv run python -c "
import json, sys
from app.llm.modules import ModuleContent
ModuleContent.model_validate(json.load(open(sys.argv[1])))
print('OK')
" data/bfrpg/modules/morgansfort.json
```

### Extracting a module from a played session

After a session ends the DM can crystallise improvised content into a reusable module:

```bash
curl -s -X POST http://localhost:8000/api/sessions/<session_id>/extract-module | jq
```

The resulting Module row has `public=false`. Use the dashboard or `POST /api/campaigns/from-module` to load it into a fresh campaign.

---

## Character system

Characters are independent of campaigns — they live in a **roster** and can be enrolled into campaigns, then returned to the roster when a campaign ends or mid-run.

**Chargen:** race and class eligibility is computed from `data/bfrpg/races.yaml` and `data/bfrpg/classes.yaml`. Abilities roll 3d6 in order (classic) or 4d6 drop lowest (heroic). Starting HP, AC, gold, and saving throws are derived server-side.

**Party setup:** after creating a campaign the party setup screen lets you pick characters from your roster or roll fresh ones. Characters rolled inside a campaign land directly in it.

**Roster:** the "Save to roster" button on any character sheet returns that character to your roster (sets `campaign_id = NULL`). The dashboard's **Your Characters** section shows all your characters across every campaign, with a campaign tag or *roster* badge.

---

## Quality gates

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy app
uv run pytest
```

---

## Production deployment

See `deploy/bootstrap.sh` and the systemd/nginx artefacts alongside it. The full design and operational runbook live in `dungeon-master-spec.md`.

Short version: single AlmaLinux 10.1 host, gunicorn + uvicorn workers behind nginx, Redis on localhost, vLLM and FLUX served from a separate GPU box on the internal LAN.

---

## License

MIT — see [`LICENSE`](LICENSE).

Module content (`data/bfrpg/modules/`) is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), derived from the Basic Fantasy RPG core rules.
