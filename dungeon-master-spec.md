# Dungeon Master — Technical Specification

**Version:** 0.7 (draft)
**Target host:** AlmaLinux 10.1
**LLM backend:** Nemotron 3 Super via vLLM at `http://svrai01.mcconaghygroup.internal:8000`
**Image backend:** FLUX.1 [dev] (txt2img) + FLUX.1 Kontext [dev] (edit) at `http://svrai01.mcconaghygroup.internal:11437`
**Ruleset:** Basic Fantasy RPG (BFRPG) 4th edition — CC BY-SA
**Audience:** 2–4 humans per table, mixed/new TTRPG experience
**Tone:** Gritty & deadly (PCs can and do die)

---

## 1. Goals & non-goals

### Goals
- A self-hosted web app where 2–4 players sit at a virtual table while an LLM-driven Dungeon Master narrates, adjudicates, and runs encounters under BFRPG rules.
- AI-generated scene images for locations, NPCs, and key beats.
- Authoritative game state held server-side. The LLM narrates; the backend rules.
- Persistent campaigns with long-term memory across sessions.
- Runs entirely on internal infrastructure — no external API dependencies beyond what's already on the network.

### Non-goals (v1)
- Voice (TTS / STT). Defer to v2.
- Battlemaps with token movement. Theatre-of-the-mind only.
- Mobile-first UI. Desktop-first; mobile usable but not optimised.
- Public deployment. Assume internal network behind VPN or trusted LAN.

---

## 2. High-level architecture

```
                      ┌──────────────────────────┐
                      │   Browser (HTMX + WS)    │
                      └────────────┬─────────────┘
                                   │ HTTPS / WSS
                      ┌────────────▼─────────────┐
                      │   nginx  (TLS, static)   │
                      └────────────┬─────────────┘
                                   │
                ┌──────────────────▼──────────────────┐
                │  FastAPI app  (uvicorn, systemd)    │
                │  - REST API                         │
                │  - WebSocket session hub            │
                │  - DM orchestrator                  │
                │  - Rules engine (BFRPG)             │
                └─┬─────────────┬───────────────────┬─┘
                  │             │                   │
        ┌─────────▼──────┐ ┌────▼─────┐  ┌──────────▼──────────┐
        │   SQLite 3     │ │  Redis   │  │  Image worker       │
        │   (WAL mode,   │ │  (pub/   │  │  (asyncio task)     │
        │   single file) │ │  sub,    │  │                     │
        │   + numpy mem  │ │  cache)  │  │                     │
        └────────────────┘ └──────────┘  └─────┬───────────────┘
                  ▲                            │
                  │                            ▼
                  │                ┌──────────────────────────────┐
                  │                │  FLUX.1 [dev] + Kontext [dev]│
                  │                │  @ svrai01:11437             │
                  │                │  GPU 1: RTX 5090 (32GB)      │
                  │                │  shares w/ Ollama Gemma 3    │
                  │                └──────────────────────────────┘
                  │
        ┌─────────┴────────────────────────────────────────┐
        │  vLLM @ svrai01.mcconaghygroup.internal:8000     │
        │  Nemotron 3 Super, OpenAI-compatible API         │
        │  GPU 0: RTX Pro 6000 Blackwell (96GB)            │
        └──────────────────────────────────────────────────┘
```

### Hardware topology

Both AI services live on `svrai01` but on physically distinct GPUs:

| GPU | Card | VRAM | Resident workloads |
|---|---|---|---|
| 0 | RTX Pro 6000 Blackwell | 96 GB | Nemotron 3 Super (vLLM, port 8000) |
| 1 | RTX 5090 Blackwell | 32 GB | FLUX.1 [dev] / Kontext [dev] (port 11437), Ollama Gemma 3 |

This separation is load-bearing for the design. Image generation runs in the literal background — while FLUX is loading its fp8 pipeline and grinding 28 inference steps on GPU 1, Nemotron on GPU 0 is free to keep narrating, parsing player input, running async summarisation, and answering follow-up turns. The placeholder-then-replace WebSocket UX is a real concurrency model, not a polite stall.

The 96 GB on the Pro 6000 is meaningful for sustained sessions: Nemotron's KV cache grows with conversation length, and headroom there directly translates to longer in-session context windows before we have to fall back on summarisation.

### Data flow for a player action
1. Player types an action in the browser.
2. Browser sends it over the session WebSocket.
3. FastAPI session hub appends it to the session log, persists, and re-broadcasts to the other players.
4. DM orchestrator builds a prompt from system rules + scene + recent turns + retrieved long-term facts + active PC stats.
5. Orchestrator streams from vLLM. Token chunks are pushed to all WebSocket clients in the session as they arrive.
6. If the LLM emits a tool intent block (dice roll, HP change, image request), it's parsed, the backend executes the rule deterministically, and the result is fed back into the next prompt.
7. Image requests are handed to the image worker async queue; placeholders show in the UI, real images replace them when ready.

---

## 3. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.12 | Async-friendly, available in AL10.1 AppStream |
| Web framework | FastAPI 0.115+ | Async-native, Pydantic, clean WS support |
| ASGI server | uvicorn (with gunicorn workers in prod) | Standard FastAPI deployment |
| ORM | SQLAlchemy 2.x (async) | Mature, async support |
| Migrations | Alembic | Pairs with SQLAlchemy |
| Database | SQLite 3.45+ (WAL mode) | Single-file, daemon-free, sufficient for 2–4 concurrent players. Ships with Python; nothing to install. |
| Vector retrieval | NumPy brute-force cosine over BLOB-stored embeddings | A few hundred to low-thousand world facts per campaign retrieves in <5 ms. No extension needed. Swap to `sqlite-vec` if scale demands. |
| Cache / pubsub | Redis 7 | WebSocket fan-out, rate limit counters |
| Templating | Jinja2 | Server-rendered HTML |
| Frontend interactivity | HTMX 2 + Alpine.js + SSE/WS | No SPA build pipeline |
| LLM client | `openai` Python SDK | vLLM is OpenAI-compatible |
| Reverse proxy | nginx | TLS termination, static assets |
| Process supervision | systemd | Native to AL10.1 |
| Env management | `uv` | Fast, reproducible Python envs |
| Auth | passlib + bcrypt + session cookies | Local accounts; OAuth out of scope v1 |

---

## 4. Why Basic Fantasy RPG

Concrete reasons, given the constraints of this project:

- **CC BY-SA license** — the rules can be embedded in prompts, redistributed in the app, and bundled with adventures.
- **Small rules surface** — the core combat loop is `d20 + bonus ≥ AC`, saves are `d20 ≥ target`, ability checks are `d20 ≤ ability score`. Easy for the LLM to apply consistently and easy for the backend to adjudicate.
- **Ascending AC** (no THAC0). Modern feel without 5e's complexity.
- **Lethal at low levels** — 1d4–1d8 starting HP fits the "gritty & deadly" tone naturally; no homebrew needed.
- **Race and class are separate** — friendlier to new players than B/X's race-as-class.
- **Treasure = XP** — most XP comes from recovering treasure rather than killing things, which encourages roleplay and clever play. Important for new players: combat isn't the only path forward.
- **Henchmen rules** — small parties can hire NPC retainers, which is both lore-appropriate and a hedge against a 2-player session where the LLM-DM would otherwise have to pull punches.

### House rules baked into v1

| Rule | Default | Rationale |
|---|---|---|
| Death and Dismemberment | On (BFRPG-compatible OSR table) | Adds nuance to "0 HP = dead"; survivable scars over insta-death |
| Death's Door (negative HP buffer) | Off | Keep gritty tone |
| Fast healing (1d3/day) | On | Keeps 2-player parties viable |
| Variable weapon damage | On | Standard BFRPG option |
| Ascending AC only | On | Simpler for new players |
| XP for treasure | On | Encourages exploration |

These are stored as a `house_rules` JSON column on the campaign and surfaced in the DM's system prompt so it adjudicates consistently.

---

## 5. Database schema

### Conventions
- Primary keys are UUIDv7 (time-ordered) generated in Python and stored as `TEXT(36)`.
- `created_at`, `updated_at` are ISO-8601 strings stored as `TEXT`, with `DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))`. SQLAlchemy's `DateTime` type handles round-tripping transparently.
- JSON payloads stored as `TEXT` (with `CHECK(json_valid(col))` where worth enforcing). Querying via `json_extract` is available but reserved for diagnostic use; business logic deserialises to Python objects.
- Booleans stored as `INTEGER` 0/1 (SQLite has no native bool — SQLAlchemy maps `Boolean` to this).
- Foreign keys are declared and enforced; every connection runs `PRAGMA foreign_keys = ON;` (SQLite defaults to off).
- Soft deletes via `deleted_at` (TEXT, nullable) rather than `DELETE`.
- Embeddings stored as `BLOB` — a serialised NumPy `float32` array. 1024-dim → 4 KB per row.

### Required pragmas (set on every connection)

```python
# app/db/session.py — applied via SQLAlchemy connection events
PRAGMAS = """
    PRAGMA journal_mode = WAL;          -- concurrent readers, single writer, no blocking
    PRAGMA synchronous = NORMAL;        -- WAL-safe, ~3x faster than FULL
    PRAGMA foreign_keys = ON;
    PRAGMA busy_timeout = 5000;         -- wait 5s on lock contention before raising
    PRAGMA cache_size = -65536;         -- 64 MB page cache
    PRAGMA temp_store = MEMORY;
    PRAGMA mmap_size = 268435456;       -- 256 MB memory-mapped I/O
"""
```

WAL mode is non-negotiable: it's what makes 2–4 concurrent players viable. Without it, every write blocks every read.

### Core tables (DDL)

```sql
-- Users / accounts
CREATE TABLE users (
    id           TEXT PRIMARY KEY,                          -- UUIDv7 hex
    username     TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email        TEXT UNIQUE COLLATE NOCASE,
    pwd_hash     TEXT NOT NULL,
    is_admin     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Campaigns: a persistent world + party
CREATE TABLE campaigns (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    owner_id      TEXT NOT NULL REFERENCES users(id),
    ruleset       TEXT NOT NULL DEFAULT 'bfrpg',
    house_rules   TEXT NOT NULL DEFAULT '{}'  CHECK(json_valid(house_rules)),
    world_state   TEXT NOT NULL DEFAULT '{}'  CHECK(json_valid(world_state)),
    long_summary  TEXT,
    module_id     TEXT REFERENCES modules(id),
    module_state  TEXT NOT NULL DEFAULT '{}'  CHECK(json_valid(module_state)),
    image_style   TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE campaign_members (
    campaign_id  TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL REFERENCES users(id),
    role         TEXT NOT NULL CHECK (role IN ('owner','player')),
    PRIMARY KEY (campaign_id, user_id)
);

-- Characters (PCs)
CREATE TABLE characters (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id),
    campaign_id         TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    race                TEXT NOT NULL,
    class_name          TEXT NOT NULL,
    level               INTEGER NOT NULL DEFAULT 1,
    xp                  INTEGER NOT NULL DEFAULT 0,
    hp_current          INTEGER NOT NULL,
    hp_max              INTEGER NOT NULL,
    ac                  INTEGER NOT NULL,
    str_score           INTEGER NOT NULL,
    int_score           INTEGER NOT NULL,
    wis_score           INTEGER NOT NULL,
    dex_score           INTEGER NOT NULL,
    con_score           INTEGER NOT NULL,
    cha_score           INTEGER NOT NULL,
    gold                INTEGER NOT NULL DEFAULT 0,
    alignment           TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'alive',
    sheet               TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(sheet)),
    canonical_image_id  TEXT REFERENCES generated_images(id),
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX idx_characters_campaign ON characters(campaign_id);

-- Inventory
CREATE TABLE inventory_items (
    id            TEXT PRIMARY KEY,
    character_id  TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    item_type     TEXT NOT NULL,
    quantity      INTEGER NOT NULL DEFAULT 1,
    equipped      INTEGER NOT NULL DEFAULT 0,
    properties    TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(properties)),
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE spells_known (
    id            TEXT PRIMARY KEY,
    character_id  TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    spell_name    TEXT NOT NULL,
    spell_level   INTEGER NOT NULL,
    prepared      INTEGER NOT NULL DEFAULT 0
);

-- Sessions
CREATE TABLE sessions (
    id                   TEXT PRIMARY KEY,
    campaign_id          TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    started_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    ended_at             TEXT,
    summary              TEXT,
    current_location_id  TEXT,
    state                TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(state))
);

-- Every utterance, action, and DM narration
CREATE TABLE session_messages (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    sender_kind   TEXT NOT NULL,
    sender_id     TEXT,
    audience      TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(audience)),
    content       TEXT NOT NULL,
    image_id      TEXT,
    dice_rolls    TEXT,                          -- JSON or NULL
    tool_calls    TEXT,                          -- JSON or NULL
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX idx_messages_session_time ON session_messages(session_id, created_at);

-- NPCs
CREATE TABLE npcs (
    id                  TEXT PRIMARY KEY,
    campaign_id         TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    description         TEXT,
    stats               TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(stats)),
    location_id         TEXT,
    alive               INTEGER NOT NULL DEFAULT 1,
    canonical_image_id  TEXT REFERENCES generated_images(id),
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Locations
CREATE TABLE locations (
    id            TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    parent_id     TEXT REFERENCES locations(id),
    name          TEXT NOT NULL,
    description   TEXT,
    image_id      TEXT,
    metadata      TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(metadata))
);

-- Encounters
CREATE TABLE encounters (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    monsters      TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(monsters)),
    initiative    TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(initiative)),
    round_number  INTEGER NOT NULL DEFAULT 1,
    current_turn  INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Long-term memory: vectorised facts about the world
CREATE TABLE world_facts (
    id                TEXT PRIMARY KEY,
    campaign_id       TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    fact              TEXT NOT NULL,
    embedding         BLOB NOT NULL,                        -- np.float32 array bytes
    embedding_dim     INTEGER NOT NULL,                     -- 1024 typical
    tags              TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(tags)),
    importance        INTEGER NOT NULL DEFAULT 5,
    source_session_id TEXT,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX idx_world_facts_campaign ON world_facts(campaign_id);
-- No vector index. Retrieval is brute-force cosine in NumPy, scoped per campaign.
-- See §7 "Memory tiers" for the retrieval routine.

-- Generated images
CREATE TABLE generated_images (
    id                TEXT PRIMARY KEY,
    campaign_id       TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    kind              TEXT NOT NULL,
    prompt            TEXT NOT NULL,
    prompt_hash       TEXT NOT NULL UNIQUE,
    file_path         TEXT NOT NULL,
    width             INTEGER,
    height            INTEGER,
    source_image_id   TEXT REFERENCES generated_images(id),
    edit_instruction  TEXT,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Audit log of dice rolls (server-authoritative)
CREATE TABLE dice_rolls (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    actor_kind    TEXT NOT NULL,
    actor_id      TEXT,
    expression    TEXT NOT NULL,
    individual    TEXT NOT NULL CHECK(json_valid(individual)),
    total         INTEGER NOT NULL,
    purpose       TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Modules (see §10 for full design)
CREATE TABLE modules (
    id                 TEXT PRIMARY KEY,
    author_id          TEXT NOT NULL REFERENCES users(id),
    name               TEXT NOT NULL,
    description        TEXT,
    min_level          INTEGER,
    max_level          INTEGER,
    tone               TEXT,
    estimated_sessions INTEGER,
    content            TEXT NOT NULL CHECK(json_valid(content)),
    image_manifest     TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(image_manifest)),
    source_session_id  TEXT REFERENCES sessions(id),
    public             INTEGER NOT NULL DEFAULT 0,
    version            INTEGER NOT NULL DEFAULT 1,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
```

### Concurrency notes

WAL mode allows arbitrarily many readers concurrent with one writer. Writers serialise on a single mutex; under our load (2–4 humans + DM + image worker) write contention is mild — typical bursts measure in microseconds.

The one rule: **never hold a write transaction open across an LLM streaming call.** A 30-second write transaction would block every other writer (DM tool-call commits, dice roll inserts, image-ready notifications) for the whole duration. Discipline:

- Open a transaction, persist player input, commit. *Then* call the LLM.
- Stream tokens with no transaction held.
- On `narration_complete`, open a new transaction, persist the message + tool-call mutations atomically, commit.

SQLAlchemy's default per-statement autocommit makes the wrong thing easy; we use explicit `async with session.begin():` blocks scoped tightly around mutations.

---

## 6. Rules engine

A pure-Python module that owns all mechanical adjudication. The LLM never resolves a die — it asks the engine.

### Module surface
```python
# app/game/rules.py
def ability_modifier(score: int) -> int: ...
def attack_roll(attacker, target, weapon) -> AttackResult: ...
def saving_throw(character, save_kind: str, dc: int) -> SaveResult: ...
def ability_check(character, ability: str, dc: int) -> CheckResult: ...
def apply_damage(character, amount: int) -> DamageResult: ...
def hp_at_level_up(character) -> int: ...
def xp_for_treasure(gp_value: int) -> int: ...
def encounter_xp(monsters: list[Monster]) -> int: ...

# app/game/dice.py
def roll(expression: str, *, advantage: bool = False) -> Roll: ...
# Supports: NdM, NdM+K, NdMkhK (keep highest), NdMklK, advantage/disadvantage shortcuts
```

### Server-authoritative execution

Every state change goes through the engine and is logged in `dice_rolls` and persisted on the relevant entity. The LLM's role is to:
1. Decide *what* to ask for (an attack, a save, a check) and against *what target* (DC, AC).
2. Narrate the outcome the engine returns.

It never decides if the dice succeeded.

---

## 7. LLM integration

### Client
```python
# app/llm/client.py
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://svrai01.mcconaghygroup.internal:8000/v1",
    api_key="not-needed",  # vLLM accepts any string
)

async def stream_dm(messages, *, tools=None):
    return await client.chat.completions.create(
        model="nemotron-3-super",
        messages=messages,
        tools=tools,
        temperature=0.85,
        max_tokens=1024,
        stream=True,
    )
```

### Tool calling

The vLLM endpoint is launched with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, which parses tool calls from Nemotron's output into OpenAI-compatible `tool_calls` in the response. Native tool calling is therefore the primary path — no JSON-block fallback as default. Tools are passed to the API via the standard `tools` parameter on `chat.completions.create(...)` and the orchestrator dispatches on `response.choices[0].message.tool_calls`.

**Watch-item: long-input failure mode.** The `qwen3_coder` parser has a documented failure mode on long inputs containing tool calls — under certain conditions it can emit an infinite stream of `!` tokens. The fix when it bites is to switch the vLLM flag to `--tool-call-parser qwen3_xml` (a newer parser with the same output contract). The orchestrator should include a runaway-token detector: if the streamed response contains >50 consecutive identical tokens, abort the request and surface the error.

**Defensive secondary path.** As a belt-and-braces measure, the orchestrator also accepts JSON tool-call blocks in `response.choices[0].message.content` as a fallback — if `tool_calls` is empty but the content contains a fenced \`\`\`json block matching the tool schema, we extract it. This catches occasional parser misses without requiring a config change. Same tool surface either way:

```python
TOOLS = [
    {
        "name": "request_dice_roll",
        "description": "Ask the engine to roll dice. Use for any check, save, attack, or damage roll.",
        "parameters": {
            "expression": "string, e.g. '1d20+3'",
            "purpose": "string, human-readable",
            "actor": "character_id or 'dm'",
            "target": {"kind": "ac|dc|none", "value": "int"}
        }
    },
    {"name": "apply_damage",        "parameters": {"target_id": "uuid", "amount": "int", "source": "string"}},
    {"name": "heal",                "parameters": {"target_id": "uuid", "amount": "int"}},
    {"name": "award_xp",            "parameters": {"character_ids": "[uuid]", "amount": "int", "reason": "string"}},
    {"name": "award_treasure",      "parameters": {"character_ids": "[uuid]", "items": "[Item]", "gold": "int"}},
    {"name": "transition_location", "parameters": {"location_id": "uuid", "description": "string"}},
    {"name": "spawn_npc",           "parameters": {"name": "string", "stats": "object"}},
    {"name": "generate_scene_image","parameters": {"prompt": "string", "kind": "scene|npc|item"}},
    {"name": "whisper",             "parameters": {"character_id": "uuid", "content": "string"}},
    {"name": "start_encounter",     "parameters": {"name": "string", "monsters": "[Monster]"}},
    {"name": "end_encounter",       "parameters": {"encounter_id": "uuid", "outcome": "string"}},
    {"name": "mark_beat",           "parameters": {"beat_id": "string", "summary": "string"}},
    {"name": "reveal_secret",       "parameters": {"secret_id": "string"}},
]
```

### Prompt structure

The system prompt is composed at every turn from layered sources:

```
[ROLE]
You are the Dungeon Master for a Basic Fantasy RPG game. You narrate vividly,
adjudicate fairly, and maintain a gritty tone. Player characters can die —
do not pull punches, but do telegraph danger. Never roll your own dice;
always call request_dice_roll. Never declare HP changes; always call
apply_damage or heal.

[RULES SUMMARY]
<compressed BFRPG rules: ~1500 tokens covering AC, saves, classes,
spell-casting, combat sequence, conditions>

[HOUSE RULES]
- Death and Dismemberment table on. Critical hits roll on the table at 0 HP.
- Variable weapon damage on.
- XP for recovered treasure (1 XP per 1 gp).

[CAMPAIGN]
Name: <campaign.name>
Long-term context: <campaign.long_summary>

[CURRENT LOCATION]
<location.name> — <location.description>

[ACTIVE PCs]
<for each: name, race/class/level, HP cur/max, AC, key abilities, status>

[ACTIVE NPCs IN SCENE]
<for each: name, attitude, brief description>

[RECENT TURNS]
<last N=20 messages verbatim>

[SESSION SO FAR]
<sessions.summary, regenerated every 20 turns>

[RELEVANT WORLD FACTS]
<top-K=5 from in-process NumPy cosine similarity over campaign-scoped embeddings>

[ACTIVE ENCOUNTER]
<if any: round, initiative order, monster HP visible to DM>
```

### Memory tiers

| Tier | Storage | Refresh | Size budget |
|---|---|---|---|
| Verbatim | `session_messages` last N=20 | Per turn | ~3k tokens |
| Session summary | `sessions.summary` | Every 20 turns, async | ~500 tokens |
| Campaign summary | `campaigns.long_summary` | End of session, async | ~1k tokens |
| World facts (vector) | `world_facts` (BLOB embeddings) + NumPy retrieval | On significant events | top-K retrieval per turn |

A separate "fact extractor" call runs after each player action: it asks the LLM "did anything happen here that should be remembered long-term?" and writes any returned facts to `world_facts` with embeddings.

### Vector retrieval (NumPy brute-force)

No vector index — at our scale (a long campaign accumulates 500–2000 world facts at most), brute-force cosine similarity over a per-campaign embedding matrix is sub-5 ms and adds zero operational complexity.

```python
# app/llm/memory.py
import numpy as np

class WorldFactRetriever:
    """Per-campaign in-memory cache of (id, fact, embedding) tuples.
    Loaded on first use, invalidated on insert. Memory cost: ~10 MB
    per campaign at 2500 facts × 1024 dims × 4 bytes."""

    async def topk(self, campaign_id: str, query_emb: np.ndarray, k: int = 5):
        ids, facts, mat = await self._load(campaign_id)   # mat is (N, 1024)
        if mat.shape[0] == 0:
            return []
        # Cosine similarity (assumes embeddings pre-normalised on insert)
        scores = mat @ query_emb
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(ids[i], facts[i], float(scores[i])) for i in top]
```

Embeddings are L2-normalised before insert so cosine reduces to a dot product. The cache is per-campaign and lazy: load on first retrieval, invalidate on `world_facts` insert. Memory ceiling per active campaign is roughly 10 MB.

If retrieval ever becomes a bottleneck (unlikely; would need ~50k+ facts per campaign), drop in `sqlite-vec` without changing the schema beyond an index addition.

### Token budget (rough)

| Slot | Tokens |
|---|---|
| System role + rules + house rules | ~2,000 |
| Campaign + location + PCs + NPCs | ~1,500 |
| Recent verbatim turns | ~3,000 |
| Session summary | ~500 |
| Retrieved world facts | ~500 |
| Output buffer | ~1,024 |
| **Total** | **~8.5k** |

Comfortably inside any reasonable Nemotron context window with headroom for combat-heavy turns.

---

## 8. Image generation

### Backend: FLUX.1 [dev] + FLUX.1 Kontext [dev] @ svrai01:11437

The image service running on `svrai01` exposes two endpoints — text-to-image (FLUX.1 [dev]) and instruction-based image editing (FLUX.1 Kontext [dev]) — pinned to GPU 1 (RTX 5090, 32GB) via `CUDA_VISIBLE_DEVICES=1`.

> **Naming note.** The systemd unit description and the service docstring both refer to the model as "FLUX.2 Dev", but the model constants in `flux_service.py` are `black-forest-labs/FLUX.1-dev` and `black-forest-labs/FLUX.1-Kontext-dev`. The constants are authoritative — that's what `diffusers` actually pulls from HuggingFace. This spec assumes FLUX.1; if you elect to upgrade to FLUX.2, see §14.

Why this combination is a strong fit for D&D:

- **FLUX.1 [dev]** — excellent prompt adherence and detail at scene-illustration scale; rectified-flow transformer with T5-XXL + CLIP text conditioning.
- **FLUX.1 Kontext [dev]** — instruction-based editing of an existing image. "Same character, now wearing battle-worn plate, in a torchlit crypt." This is how we get character/scene consistency across a campaign; see below.
- **Open weights, self-hosted on internal infrastructure** — no per-image cost, no external dependency, no data leaves the LAN.
- **fp8 layerwise casting on the transformer** — keeps both pipelines around 19GB VRAM, leaving headroom on the 5090.

### Performance characteristics (read this before designing the UX)

The service unloads its pipeline after every request (`finally: unload_all_pipelines()`) so GPU 1 can hot-swap with Ollama Gemma 3. That has real consequences:

| Phase | Approx. duration |
|---|---|
| Cold pipeline load (fp8 transformer + T5-XXL onto GPU) | ~15–30 s |
| Generation, FLUX.1 [dev], 1024×1024, 28 steps | ~8–15 s |
| Generation, Kontext, edit at source resolution, 28 steps | ~10–18 s |
| Encode + return base64 PNG | <1 s |
| **Per-request total** | **~25–45 s** |

There's also an `asyncio.Lock` (`pipe_lock`) inside the service: requests serialise. Two simultaneous /generate calls do not run concurrently.

Implications for the DM behaviour:
- The system prompt explicitly tells the DM to be sparing with images. Generate for major locations, climactic beats, first appearances of significant NPCs — not every minor moment.
- Hash-based deduplication (existing `generated_images.prompt_hash` row → reuse) becomes load-bearing. Repeat scenes cost 0 seconds.
- The placeholder-then-replace UX in the WebSocket protocol makes the latency invisible to players: the DM keeps narrating while the image renders.

### API contract (concrete)

The service is a FastAPI app on port 11437. Endpoints:

```
GET  /health
POST /generate     # FLUX.1 [dev] text-to-image
POST /edit         # FLUX.1 Kontext [dev] instruction-based edit
```

All three are synchronous: POST returns when the image is ready (or after pipeline-load + generation time elapses, ~30s typical). No polling, no webhooks.

Request/response shapes (verbatim from the service):

```python
# POST /generate
{
  "prompt": str,                # required
  "negative_prompt": str = "",
  "width": int = 1024,          # 256..2048
  "height": int = 1024,         # 256..2048
  "num_inference_steps": int = 28,   # 1..50
  "guidance_scale": float = 3.5,     # 1.0..20.0
  "seed": int | null = null
}

# POST /edit
{
  "prompt": str,                # edit instruction, e.g. "add a hood"
  "image_base64": str,          # source image, base64-encoded PNG/JPEG
  "num_inference_steps": int = 28,
  "guidance_scale": float = 2.5,
  "seed": int | null = null
}

# Response (both endpoints)
{
  "image_base64": str,                 # PNG, base64-encoded
  "seed_used": int,
  "generation_time_seconds": float,
  "filepath": str | null               # path on the FLUX server's filesystem; ignore from our side
}
```

### Client wrapper

```python
# app/images/client.py
import base64
import httpx
from app.config import settings

class FluxClient:
    def __init__(self):
        self.base = settings.flux_base_url   # http://svrai01.mcconaghygroup.internal:11437
        # Generous timeout: cold pipeline load + generation can run to a minute.
        self.timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{self.base}/health")
            r.raise_for_status()
            return r.json()

    async def generate(self, prompt: str, *, negative_prompt: str = "",
                       width: int = 1024, height: int = 1024,
                       steps: int = 28, guidance: float = 3.5,
                       seed: int | None = None) -> tuple[bytes, int]:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width, "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base}/generate", json=payload)
            r.raise_for_status()
            data = r.json()
        return base64.b64decode(data["image_base64"]), data["seed_used"]

    async def edit(self, prompt: str, source_png: bytes, *,
                   steps: int = 28, guidance: float = 2.5,
                   seed: int | None = None) -> tuple[bytes, int]:
        payload = {
            "prompt": prompt,
            "image_base64": base64.b64encode(source_png).decode(),
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base}/edit", json=payload)
            r.raise_for_status()
            data = r.json()
        return base64.b64decode(data["image_base64"]), data["seed_used"]
```

The DM orchestrator only ever sees `flux.generate(...)` / `flux.edit(...)`. Everything FLUX-specific is contained in this module.

### Worker pipeline

1. DM emits a `generate_scene_image` or `restyle_npc` tool call with a prompt and a `kind` (scene / npc / item / map).
2. Image worker hashes `(campaign_id, kind, prompt + style_suffix, [reference_image_id])`. If `generated_images` already has a row with that hash, reuse it — saves 30–45s.
3. Otherwise enqueue a job in Redis (`LPUSH images:queue`).
4. Image worker (its own systemd unit, single concurrency) pops the job. For txt2img, calls `flux.generate(...)`; for an edit, fetches the reference image bytes from disk and calls `flux.edit(...)`.
5. Worker writes the PNG to `/var/lib/dungeon-master/images/<uuid>.png` and inserts a `generated_images` row.
6. Worker publishes `image:ready:<id>` on Redis pub/sub. Session WS handler broadcasts `image_ready` to the table; the placeholder card in the UI swaps to the rendered image.

Single concurrency at the worker level matches the service's internal lock — no point queueing more.

### Generation parameters per kind

Defaults align with FLUX.1 [dev] best practice (guidance ~3.5) and the service's actual defaults.

| Kind | Resolution | Steps | Guidance | Notes |
|---|---|---|---|---|
| `scene` | 1280×768 | 28 | 3.5 | Wide aspect for rooms/landscapes |
| `npc` | 768×1024 | 32 | 3.5 | Portrait aspect; more steps for face fidelity |
| `item` | 1024×1024 | 24 | 3.5 | Square; trim steps for faster turnaround |
| `map` | 1280×1280 | 36 | 4.0 | More steps for typography/legibility |

`negative_prompt` is set per-campaign (e.g. "modern objects, photographic, watermark, text artefacts, extra fingers") and prepended to every request.

### Style consistency

A campaign-level `image_style` field, set at campaign creation ("dark fantasy oil painting, muted palette, candlelight, painterly brushwork"), is appended to every prompt. FLUX.1's prompt adherence is strong enough that a stable style suffix delivers a stable look across a session.

### Character & NPC consistency via Kontext (phase 5+)

FLUX.1 doesn't have FLUX.2's multi-image reference, but Kontext gets us to the same place via a different path:

1. **Canonical portrait.** When a PC is rolled or a recurring NPC is spawned, the worker calls `/generate` once with a portrait prompt derived from the character's description. The result is stored as the canonical image and linked from `characters.canonical_image_id` / `npcs.canonical_image_id`.
2. **Contextual edits.** When a scene calls for that character in a new context, instead of /generate from scratch, the worker calls `/edit` with the canonical portrait as the source and an instruction like *"Same character, now standing in a torchlit crypt, sword drawn, blood on her armour, kneeling beside a fallen companion."* Kontext preserves the character's identity (face, hair, build, signature gear) while rewriting the surroundings and pose.
3. **Cache aggressively.** The hash for an edit incorporates the canonical image id, so the same character in the same scene reuses the existing render.

Schema additions to support this are folded into the consolidated DDL in §5: `characters.canonical_image_id`, `npcs.canonical_image_id`, `generated_images.source_image_id`, and `generated_images.edit_instruction`.

Phase 5 ships the canonical-portrait-on-creation path; phase 6 adds Kontext-driven scene edits.

### Throttling & failure

- The service serialises with its own lock; the worker also runs single-concurrency. No need for application-side rate limiting.
- A 503 / OOM from the service triggers an exponential backoff retry (3 attempts, 5s/15s/45s) before marking the job failed and emitting a `image_failed` WS event. The DM's narration is not blocked by failure — the player just sees "(scene image unavailable)" in place of the card.
- The `/health` endpoint is polled every 30s by a watchdog. If unreachable for >2 min, image generation is marked degraded and the DM is told (in-prompt) to omit `generate_scene_image` calls until further notice.

### GPU placement

Confirmed: Nemotron 3 Super runs on GPU 0 (RTX Pro 6000 Blackwell, 96 GB), FLUX.1 + Ollama Gemma 3 share GPU 1 (RTX 5090 Blackwell, 32 GB). Different PCIe devices, no contention. Image generation does not block narration.

A consequence worth designing around: the Pro 6000's headroom means we can be generous with Nemotron's context window before falling back on summarisation. The verbatim-turn budget in §7 is set conservatively at N=20; with 96 GB and a sensible KV cache config we can likely push that to N=40+ for a smoother in-session feel. Worth tuning during phase 2 once we see real KV usage patterns.

### Storage & serving

- Files on disk under `/var/lib/dungeon-master/images/` on the *web app host* (chown app user, mode 0750). Note the FLUX service also writes to its own `/opt/svrai/generated_images/` on `svrai01` — that's the service's local cache and is not what we serve. We pull base64 over the wire and persist locally.
- Served by nginx via an `internal` location with `X-Accel-Redirect`, so the FastAPI app authorises access (only campaign members see a campaign's images).
- `generated_images` rows include the prompt, parameters, seed, and source image (for edits) — enough to "regenerate this scene with a different seed" or trace provenance.

---

## 9. Multiplayer & real-time

### Connection

Each client opens `WSS /ws/session/{session_id}` after auth. The server:
1. Verifies the user is a member of the campaign.
2. Subscribes the connection to the Redis channel `session:{session_id}`.
3. Sends a state snapshot (last 50 messages, active encounter, current scene).

### Message types (server → client)

| Type | Payload |
|---|---|
| `narration_chunk` | streamed token chunk from DM |
| `narration_complete` | full message, with tool_calls executed |
| `pc_action` | another player's action |
| `whisper` | private DM message (only sent to the target client) |
| `dice_roll` | a roll the engine performed |
| `state_update` | character/encounter state delta |
| `image_pending` | image_id, placeholder |
| `image_ready` | image_id, URL |
| `presence` | who is connected |

### Turn order

- **Out of combat:** free-form. Any player can post an action; the DM responds. No queue.
- **In combat:** initiative is rolled at encounter start, stored in `encounters.initiative`. The current actor is highlighted in the UI; only that PC can submit combat actions. Other players can still chat / whisper.
- The DM always acts as monsters and NPCs on their initiative slots.

### Whispers

The `whisper` tool sends a message only to the target character's user. The full whisper is stored in `session_messages` with `audience=[character_id]` — visible to the DM in the prompt history (so it stays consistent), invisible to other players in the UI.

---

## 10. Adventure modules

### Concept

A module is a reusable adventure: locations, NPCs, encounters, plot beats, secrets, treasure. Save once, run repeatedly with different parties — or the same party with new characters.

The crucial property: a module is a *skeleton*, not a script. Each playthrough is genuinely different because Nemotron narrates fresh every time. The same module played twice produces two different stories at the same campfire. This is a feature unique to LLM-driven play that traditional D&D modules can't offer.

### Module structure

The module content is a single JSON document on the `modules` row:

```json
{
  "synopsis": "A border keep on the edge of a goblin-infested wilderness. The party arrives seeking work...",
  "level_range": [1, 3],
  "estimated_sessions": 4,
  "starting_hook": "The party has been hired by Castellan Thorvald to investigate goblin raids.",
  "starting_location_id": "loc_keep",
  "tone": "gritty, isolated, morally grey",
  "image_style": "dark fantasy oil painting, muted palette, candlelight",

  "locations": [
    {
      "id": "loc_keep",
      "name": "The Keep on the Borderlands",
      "description": "A squat stone fortification...",
      "areas": [
        {"id": "area_gate", "name": "Main Gate", "description": "...", "secrets": ["..."]}
      ],
      "canonical_image_id": "img_uuid"
    }
  ],

  "npcs": [
    {
      "id": "npc_castellan",
      "name": "Castellan Thorvald",
      "description": "Greying veteran, missing two fingers on his left hand.",
      "motivation": "Wants the goblin threat gone before his lord visits in autumn.",
      "starting_location_id": "loc_keep",
      "stats_block": {"hd": 4, "ac": 16, "hp": 22, "morale": 9},
      "sample_dialogue": ["...", "..."],
      "secrets": ["He bribed the goblins three years ago to attack a rival keep."],
      "canonical_image_id": "img_uuid"
    }
  ],

  "encounters": [
    {
      "id": "enc_patrol",
      "name": "Goblin patrol",
      "trigger": "When the party first leaves the keep heading north",
      "monsters": [{"name": "Goblin", "count": "4-6", "tactics": "ambush from rocks"}],
      "treasure": "1d6 sp each, 1 in 6 carries a crude map fragment"
    }
  ],

  "plot_beats": [
    {
      "id": "beat_1",
      "title": "Discover the bribery",
      "trigger_hint": "Speaking with the goblin shaman, or finding old letters in the castellan's quarters",
      "outcome": "The party knows Thorvald's secret",
      "secrets": ["..."],
      "leads_to": ["beat_3", "beat_4"]
    }
  ],

  "secrets": [
    {"id": "sec_1", "content": "The 'goblin caves' are an old elven tomb the goblins squat in.", "reveal_when": "On entering area cave_3"}
  ],

  "treasure_pools": [
    {"id": "pool_caves", "value_gp": 240, "items": ["Potion of Healing", "Silver dagger (15gp)"]}
  ],

  "endings": [
    {"id": "end_resolved", "trigger": "Goblin threat eliminated and bribery exposed", "outcome": "Thorvald flees, captain takes command, party paid double"},
    {"id": "end_complicit", "trigger": "Threat eliminated but bribery hidden", "outcome": "Standard payment, Thorvald owes the party a favour"}
  ]
}
```

### Schema additions

The `modules` table and the `campaigns.module_id` / `campaigns.module_state` columns are listed in the consolidated DDL in §5. Shape of `module_state`:

```jsonc
{
  "beats_hit":         ["beat_1", "beat_2"],
  "beats_pending":     ["beat_3", "beat_4"],
  "secrets_revealed":  ["sec_1"],
  "encounters_run":    ["enc_patrol"],
  "endings_reached":   []
}
```

Note that the level range is stored as two columns (`min_level`, `max_level`) rather than a single range type — SQLite doesn't have ranges. Trivial in application code.

### Authoring paths

Two ways a module is created. We ship #1 in phase 8; #2 is a stretch.

1. **Auto-extract from a completed session.** Primary path. At session-end the user clicks "Save as module". An extraction LLM call digests `session_messages` + locations + npcs + encounters + world_facts and emits a structured module JSON matching the schema above. The user reviews and lightly edits before saving.
2. **From scratch.** A blank module with an editor UI. Deferred — almost nobody starts here in practice.

Both paths funnel into the same editor for review/refinement before the module is saved as a versioned artefact.

### Extraction

The extraction is a single LLM call to Nemotron with a tightly structured prompt:

```
You are an adventure module designer. Given the completed session below, extract
a reusable adventure module that other parties could play.

REQUIREMENTS:
- Output valid JSON matching the schema provided.
- Strip player-specific details. Do not say "the party did X"; say "when the party
  encounters X" or "if the party chooses Y".
- Preserve atmosphere, NPC voice, and signature scenes.
- Identify implicit plot beats — the moments the story turned — and list them
  with triggers, not scripts.
- Note DM-only secrets that were revealed in play; mark them as secrets in the
  module so future DMs know what the players don't.
- Set a level range based on the encounters and party power.
- Estimate session count from the actual session log length.

SCHEMA: <full module schema>

SESSION LOG:
<session_messages joined>

WORLD STATE AT END OF SESSION:
<locations, npcs, encounters, world_facts as JSON>

CHARACTER ARCS:
<for each PC: starting state, key choices, ending state>
```

The output is parsed, validated against a Pydantic model, and presented in the editor. If the LLM emits malformed JSON we retry with a corrective prompt up to three times before falling back to a partial template the user can fill in manually.

Canonical NPC and location images already in `generated_images` are linked into the module's `image_manifest` so they ship with the module. On export, image params (prompt + seed + parameters) are included so a recipient who lacks the image files can regenerate them via their own FLUX endpoint.

### Loading a module into a campaign

When a new campaign is created with `module_id` set:

1. **Deep-copy structural content.** Locations, NPCs, encounters, world_facts from the module are copied as new rows in the campaign's tables, with fresh UUIDs. Each module entity carries a back-reference (`source_module_entity_id`) for analytics, but the campaign owns mutable copies — playthrough state never bleeds back into the module.
2. **Seed world facts.** Module-level world_facts are embedded and inserted, ready for retrieval via the per-campaign NumPy similarity routine.
3. **Initialise module_state.** All plot beats start in `beats_pending`, no secrets revealed, no encounters run.
4. **Inject the module guide into the DM system prompt** (see next subsection).
5. **Place the party at `starting_location_id`** and seed the first scene with the module's `starting_hook`.

### Module guide in the DM system prompt

When a campaign has a loaded module, the DM's system prompt gains a new layered section:

```
[MODULE: Keep on the Borderlands]
Synopsis: <module.synopsis>
Tone: <module.tone>

[KEY NPCs IN THIS MODULE]
- Castellan Thorvald (loc_keep) — greying veteran, motivation: ...
- ...

[KEY LOCATIONS]
- The Keep (loc_keep), the Goblin Caves (loc_caves), ...

[PLOT BEATS — PENDING]
- beat_3: <trigger_hint> → <outcome>
- beat_4: ...

[PLOT BEATS — ALREADY HIT]
- beat_1: discover the bribery
- beat_2: meet the shaman

[SECRETS — DM ONLY, DO NOT REVEAL UNLESS TRIGGERED]
- sec_2: the goblin caves are an old elven tomb (reveal on entering cave_3)

[GUIDANCE]
You are running a published-style module. Use the locations, NPCs, and beats
above as your skeleton, but improvise freely around player choices. Steer
gently toward unfulfilled beats when the party seems aimless; never railroad.
Call mark_beat when a beat triggers. Call reveal_secret when one comes out.
```

### Beat tracking

Two new tool calls join the existing roster:

```python
{"name": "mark_beat",     "parameters": {"beat_id": "string", "summary": "string"}}
{"name": "reveal_secret", "parameters": {"secret_id": "string"}}
```

When the LLM emits one of these, the backend updates `campaigns.module_state` and feeds the result back into the next prompt so the DM's view of remaining beats stays current. A beat is *the LLM's call to make* — it decides when player action satisfies a beat's trigger. The backend just records and tracks.

A `module_state.endings_reached` of one or more entries triggers a soft prompt: "The module's resolution conditions have been met. End the campaign?" The user (DM-owner) decides; the campaign isn't auto-closed.

### Export / import

Modules are portable as a single JSON file:

```json
{
  "format_version": "1.0",
  "module": { ...full modules row... },
  "image_manifest": [
    {
      "id": "img_uuid",
      "role": "npc:castellan",
      "prompt": "...",
      "params": {"width": 768, "height": 1024, "steps": 32, "guidance": 3.5, "seed": 42}
    }
  ]
}
```

- **Export:** `GET /api/modules/{id}/export` returns the JSON.
- **Import:** `POST /api/modules/import` validates and stores. If the importer's FLUX endpoint is reachable, the loader optionally regenerates each manifest image so the module ships with art on the receiving side too.

This makes modules shareable as a single file across installations — useful for swapping content with friends running their own Dungeon Master instances.

### Variation across runs

Worth restating because it's the central design payoff: a module played a second time produces a genuinely different story. Same skeleton, same beats, same NPCs — but the LLM's prose, the dice, the players' choices, and the order in which beats trigger all vary. The encounters described as "a goblin patrol of 4–6, ambushing from the rocks" leaves room for the LLM to dial intensity to the table. Plot beats are *triggers and outcomes*, not blow-by-blow scripts.

This is also why we don't try to make modules deterministic. We're not building a saved game; we're building a campaign-in-a-bottle that breathes differently every time it's uncorked.

### Bundled starter module

To give first-run users something to play immediately, the v1 build ships with one pre-authored module sourced from the Basic Fantasy community catalogue. Candidates (all CC BY-SA via the Basic Fantasy Project):

- ***Morgansfort: The Western Lands Campaign*** — the canonical BFRPG starter. A keep on a frontier, surrounding dungeons keyed at low levels (1–3), several factions, room for 4–8 sessions of play. Strong fit for the gritty/deadly tone.
- ***BF1: Morgansfort* + *BF2: Fortress, Tomb, and Tower*** — pair of low-level adventures, slightly more tightly structured.
- ***Adventure Anthology One*** — collection of one-shots if shorter sessions are preferable.

Default pick is *Morgansfort* unless implementation reveals a licensing issue. The module is hand-authored once during phase 8 (rather than extracted from a played session) — the structured JSON is committed to the repo at `data/bfrpg/modules/morgansfort.json`, with image manifest references to portraits/scenes generated once and stored alongside. New users see it as a pre-loaded option in the campaign-creation dropdown.

---

## 11. API surface

### REST (JSON)

```
POST   /api/auth/register
POST   /api/auth/login
POST   /api/auth/logout
GET    /api/me

GET    /api/campaigns                         # list mine
POST   /api/campaigns                         # create
GET    /api/campaigns/{id}
PATCH  /api/campaigns/{id}
POST   /api/campaigns/{id}/invite             # generate invite code
POST   /api/campaigns/join                    # redeem invite code

GET    /api/campaigns/{id}/characters
POST   /api/campaigns/{id}/characters         # roll up a new PC
GET    /api/characters/{id}
PATCH  /api/characters/{id}
POST   /api/characters/{id}/level-up

POST   /api/campaigns/{id}/sessions           # start a new session
POST   /api/sessions/{id}/end
GET    /api/sessions/{id}/messages?limit=&before=
GET    /api/sessions/{id}/export              # full transcript download

POST   /api/sessions/{id}/extract-module      # kick off extraction LLM call
GET    /api/modules                           # list mine + public
POST   /api/modules                           # create from extracted draft
GET    /api/modules/{id}
PATCH  /api/modules/{id}                      # edit content
DELETE /api/modules/{id}
GET    /api/modules/{id}/export               # download as portable JSON
POST   /api/modules/import                    # upload module JSON

POST   /api/dice/roll                         # client-side flavor roll (non-authoritative)
```

### WebSocket

```
WSS    /ws/session/{session_id}
```

Bidirectional. Client → server messages: `pc_action`, `whisper_to_dm`, `out_of_band_chat`, `ping`.

---

## 12. Project layout

```
dungeon-master/
├── pyproject.toml
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
├── app/
│   ├── main.py                 # FastAPI app factory
│   ├── config.py               # pydantic-settings
│   ├── deps.py                 # FastAPI dependencies
│   ├── db/
│   │   ├── base.py
│   │   ├── session.py
│   │   ├── models.py
│   │   └── crud/
│   ├── api/
│   │   ├── auth.py
│   │   ├── campaigns.py
│   │   ├── characters.py
│   │   ├── sessions.py
│   │   └── ws.py
│   ├── llm/
│   │   ├── client.py           # vLLM client wrapper
│   │   ├── prompts.py          # prompt builders
│   │   ├── tools.py            # tool schemas + dispatch
│   │   ├── memory.py           # summarisation + fact extraction
│   │   └── rules_text.py       # condensed BFRPG rules text for system prompt
│   ├── game/
│   │   ├── rules.py            # BFRPG resolution
│   │   ├── dice.py
│   │   ├── combat.py
│   │   ├── classes.py          # Fighter, Cleric, Magic-User, Thief
│   │   ├── races.py
│   │   ├── items.py            # equipment catalogue
│   │   ├── monsters.py         # bestiary
│   │   └── chargen.py          # roll up a new PC
│   ├── images/
│   │   ├── client.py           # backend abstraction
│   │   └── worker.py           # async queue consumer
│   ├── realtime/
│   │   ├── hub.py              # session WS hub
│   │   ├── presence.py
│   │   └── pubsub.py           # Redis pub/sub
│   ├── orchestrator/
│   │   └── dm.py               # the DM turn loop
│   ├── templates/
│   │   ├── base.html
│   │   ├── table.html          # the play screen
│   │   └── ...
│   └── static/
│       ├── css/
│       ├── js/
│       └── img/
├── deploy/
│   ├── nginx.conf
│   ├── dungeon-master.service
│   ├── dungeon-master-imageworker.service
│   └── selinux/
│       └── dungeon-master.te
├── data/
│   └── bfrpg/
│       ├── classes.yaml
│       ├── spells.yaml
│       ├── monsters.yaml
│       └── equipment.yaml
└── tests/
```

---

## 13. Deployment on AlmaLinux 10.1

### Security posture

Dungeon Master is deployed on a **trusted internal LAN** with no public-internet exposure. This shapes several decisions throughout the spec:

- **Auth** is a simple username + bcrypt-hashed-password store in the `users` table. No SSO, no OAuth, no MFA, no aggressive password policy. Session cookies signed with a server-side secret; default 30-day TTL.
- **TLS** uses a self-signed certificate (see below). Browsers warn on first connect; players accept once.
- **Inter-service auth** is not enforced. The DM app talks to vLLM (`:8000`) and FLUX (`:11437`) over plain HTTP without API tokens. firewalld restricts those ports to the internal interface only.
- **The FLUX service on `svrai01`** is left as currently configured (no auth, HF token in unit file). Out of scope for this project. If posture changes later, both can be hardened independently.

If the deployment surface ever expands beyond the trusted LAN, all four of these need revisiting. Flagging here so the assumption is explicit.

### Packages

```bash
sudo dnf install -y python3.12 python3.12-devel \
    valkey nginx \
    sqlite \
    gcc git
# Valkey replaces Redis on AlmaLinux 10 — Redis was dropped from the
# base repos after the 2024 SSPL relicensing. Valkey is the Linux
# Foundation fork, wire-compatible: redis-py and redis://... URLs
# work unchanged. Service unit is valkey.service, config at
# /etc/valkey/valkey.conf, default port 6379, default bind 127.0.0.1.
# No PostgreSQL, no pgvector. SQLite ships with Python's stdlib;
# the `sqlite` CLI package is for manual inspection / backups.
```

### Database

```bash
# The DB file is created automatically by `alembic upgrade head` on first run.
# Just ensure the directory exists with correct ownership:
sudo install -d -m 0750 -o dungeonmaster -g dungeonmaster /var/lib/dungeon-master
# Path: /var/lib/dungeon-master/dm.db
# WAL files (dm.db-wal, dm.db-shm) live alongside it.
```

That's the entire database setup — one directory and a migration command. No daemon to start, no users to create, no extensions to enable, no `pg_hba.conf` to wrangle.

### App user and directories

```bash
sudo useradd -r -m -d /var/lib/dungeon-master -s /sbin/nologin dungeonmaster
sudo mkdir -p /etc/dungeon-master /var/log/dungeon-master /var/lib/dungeon-master/images
sudo chown -R dungeonmaster:dungeonmaster /var/lib/dungeon-master /var/log/dungeon-master
```

### Application install

```bash
cd /opt
sudo git clone <repo> dungeon-master
sudo chown -R dungeonmaster:dungeonmaster dungeon-master
sudo -u dungeonmaster bash -lc 'cd /opt/dungeon-master && uv sync'
sudo -u dungeonmaster bash -lc 'cd /opt/dungeon-master && uv run alembic upgrade head'
# ↑ creates /var/lib/dungeon-master/dm.db with full schema
```

### systemd units

`/etc/systemd/system/dungeon-master.service`:

```ini
[Unit]
Description=Dungeon Master web app
After=network.target valkey.service

[Service]
Type=notify
User=dungeonmaster
Group=dungeonmaster
WorkingDirectory=/opt/dungeon-master
EnvironmentFile=/etc/dungeon-master/env
ExecStart=/opt/dungeon-master/.venv/bin/gunicorn \
    -w 1 -k uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8001 \
    app.main:app
Restart=on-failure
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/dungeon-master /var/log/dungeon-master

[Install]
WantedBy=multi-user.target
```

A second unit `dungeon-master-imageworker.service` runs the image queue consumer.

> **Why one gunicorn worker, not four?** SQLite serialises writes via a single in-process WAL writer; multiple processes would still serialise but with extra IPC overhead and more lock contention. One worker with many async tasks (FastAPI's native model) is the right shape for SQLite. If we ever need to scale beyond one machine, we migrate the DB first and the workers second — not the other way around.

### TLS — self-signed cert (trusted LAN)

Operating on a trusted internal LAN, no public exposure, so a self-signed certificate is sufficient. One-shot generation:

```bash
sudo openssl req -x509 -nodes -newkey rsa:4096 \
    -keyout /etc/pki/tls/private/dm.key \
    -out    /etc/pki/tls/certs/dm.crt \
    -days 1825 \
    -subj "/CN=dm.mcconaghygroup.internal" \
    -addext "subjectAltName=DNS:dm.mcconaghygroup.internal,DNS:dm,IP:<server-ip>"
sudo chmod 600 /etc/pki/tls/private/dm.key
sudo chown root:nginx /etc/pki/tls/private/dm.key
```

Players will see a browser warning the first time and need to add an exception. For a smoother experience later, sign with an internal CA so players' machines can trust it once and forget — but not a v1 priority.

### nginx

`/etc/nginx/conf.d/dungeon-master.conf`:

```nginx
upstream dm_app { server 127.0.0.1:8001; }

server {
    listen 443 ssl http2;
    server_name dm.mcconaghygroup.internal;
    ssl_certificate     /etc/pki/tls/certs/dm.crt;
    ssl_certificate_key /etc/pki/tls/private/dm.key;

    client_max_body_size 10m;

    location /ws/ {
        proxy_pass http://dm_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
    }

    location /static/ {
        alias /opt/dungeon-master/app/static/;
        expires 7d;
    }

    location /images/ {
        internal;
        alias /var/lib/dungeon-master/images/;
    }

    location / {
        proxy_pass http://dm_app;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SELinux

```bash
sudo setsebool -P httpd_can_network_connect 1
# If the app binds outside default ports, label them:
sudo semanage port -a -t http_port_t -p tcp 8001
```

A custom policy module for the app's read/write paths under `/var/lib/dungeon-master` is included in `deploy/selinux/`.

### firewalld

```bash
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --add-service=http      # for redirect to https
sudo firewall-cmd --reload
# vLLM port 8000 stays internal — no firewall opening needed unless cross-zone
```

### Backups

Nightly cron, run as the `dungeonmaster` user:
```
0 2 * * * /usr/bin/sqlite3 /var/lib/dungeon-master/dm.db ".backup /var/backups/dm/dm-$(date +\%F).db"
0 3 * * * /usr/bin/rsync -a /var/lib/dungeon-master/images/ /var/backups/dm/images/
```

`.backup` is the only safe way to copy a live SQLite database — it takes a coordinated snapshot through the SQLite API rather than a raw filesystem copy, which would race with WAL writes. The output is a complete, consistent `.db` file you can `cp` anywhere.

Weekly: keep the last 8 daily snapshots and one weekly going back 12 weeks. A whole campaign with hundreds of sessions is well under 1 GB; storage isn't a constraint.

---

## 14. Phased implementation plan

| Phase | Scope | Estimate |
|---|---|---|
| **0. Bootstrap** | Repo, pyproject, FastAPI skeleton, Alembic, base templates, auth, AlmaLinux deploy of an empty hello-world. | 3–5 days |
| **1. BFRPG engine** | Rules module, dice, character generation, classes, equipment, monsters loaded from YAML. Unit tests for every resolution path. No LLM yet. | 1–2 weeks |
| **2. DM core (single-player, text-only)** | vLLM client, prompt builder, JSON tool protocol, server-authoritative turn loop, single-player session UI. | 2 weeks |
| **3. Memory** | Session summaries, world facts extraction, NumPy cosine retrieval per campaign. | 1 week |
| **4. Multiplayer** | WebSocket hub, Redis pub/sub, presence, whispers, initiative-driven turn order. | 1–2 weeks |
| **5. Image generation** | Backend client, async queue, scene/NPC/item images, style consistency, X-Accel-Redirect serving. | 1 week |
| **6. UX polish** | Character sheet view, encounter tracker, Markdown session-log export (PDF deferred), responsive layout. | 1–2 weeks |
| **7. Hardening** | Rate limiting, audit logging, SELinux policy module, backup automation, monitoring. | 1 week |
| **8. Adventure modules** | `modules` schema, extraction LLM call + prompt, module editor UI, loader (deep-copy + system-prompt injection), beat tracker (`mark_beat` / `reveal_secret`), export/import as JSON, **bundled Morgansfort starter module hand-authored and pre-loaded**. | 2 weeks |

A working internal demo is achievable at the end of phase 2 (~3 weeks in). Multiplayer + images by the end of phase 5 (~6–7 weeks). Modules ship at the end of phase 8 (~10–12 weeks total).

---

## 15. Decisions log

All design questions raised during specification have been resolved. Recording them here for posterity.

| Decision | Resolution |
|---|---|
| Image model — FLUX.1 vs FLUX.2 upgrade | Stay on FLUX.1 [dev] + FLUX.1 Kontext [dev] as currently deployed. Comment-vs-code drift in `flux_service.py` to be cleaned up at convenience. |
| Image model license posture | Non-commercial use under FLUX [dev] Non-Commercial License. |
| `flux_service.py` hardening (HF token in unit file, no auth, 0.0.0.0 bind) | Deferred — operating on trusted LAN. Out of scope for this project. |
| TLS certificates | Self-signed via `openssl req -x509`, 5-year validity. Browser exception accepted by players on first connect. |
| Auth | Local accounts: username + bcrypt-hashed password in the `users` table, server-signed session cookies. No SSO. Low security posture appropriate to trusted LAN. |
| Bundled starter content | Hand-authored *Morgansfort* (Basic Fantasy Project, CC BY-SA) ships as `data/bfrpg/modules/morgansfort.json`. Falls back to another BFRPG-licensed adventure if a licensing snag emerges. |
| Nemotron tool calling | Native via vLLM `--tool-call-parser qwen3_coder`. Defensive fallback: orchestrator also accepts JSON-block tool calls in message content. Watch-item: `qwen3_coder` has a documented infinite-`!` failure mode on long inputs; runaway-token detector in the orchestrator catches it; switch to `--tool-call-parser qwen3_xml` if it bites. |
| Session export format | Markdown for v1 (simple dump of `session_messages` with formatting). PDF deferred to v2. |
| GPU placement | Confirmed: Nemotron on RTX Pro 6000 (GPU 0, 96GB), FLUX + Ollama Gemma 3 share RTX 5090 (GPU 1, 32GB). No contention; image generation runs truly in background relative to narration. |
| Database engine | **SQLite 3 (WAL mode)** rather than PostgreSQL. Single-file, daemon-free, sufficient for 2–4 players and consistent with the trusted-LAN, single-server deployment shape. Eliminates ~5 deployment steps and the build-pgvector-on-AL10 dependency. |
| Vector retrieval | **NumPy brute-force cosine** over per-campaign BLOB embeddings. Sub-5 ms at expected scale (≤2k facts per campaign). Drop-in `sqlite-vec` upgrade path if needed. |

### Things to verify in flight (not blockers)

These are checks to run during phase 0 or early phase 2 — known unknowns that don't require resolution before starting:

- **Nemotron context window in production.** Phase 2 should empirically confirm we have enough KV cache headroom on the Pro 6000 to push verbatim turns from N=20 to N=40+.
- **FLUX cold-load time in practice.** ~15–30s estimate is from secondary sources; actual numbers on your hardware may differ. Worth measuring once and tuning the worker timeout accordingly.
- **Nemotron's compliance with the "DM narrates, backend adjudicates" rule.** Even with system-prompt instructions, LLMs sometimes try to roll dice in narration. Phase 2 needs an integration test that forces the model into corner cases (combat, surprise damage, healing) and verifies it routes through tools.
