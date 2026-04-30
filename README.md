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

Phase 0 (Bootstrap). See `dungeon-master-spec.md` §14 for the phased plan.

## Repository layout

See `AGENTS.md` "File organisation".

## Development

Quickstart instructions land in step 12 of the Phase 0 deliverable.

## License

MIT — see [`LICENSE`](LICENSE).
