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

Phase 0 (Bootstrap) **complete**. Phase 1 (BFRPG engine) ready to start.
See `dungeon-master-spec.md` §14 for the phased plan.

## Repository layout

See `AGENTS.md` "File organisation".

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
