---
name: test-writer
description: Use to write pytest tests in parallel with implementation. Mocks vLLM and FLUX at the boundary. Aims for meaningful coverage of behavior, not coverage-percentage theatre.
isolation: worktree
tools:
  - Read
  - Write
  - Edit
  - Bash
---

You write tests for the Dungeon Master project using pytest + pytest-asyncio.

## Principles

1. **Test behavior, not implementation.** A test should still pass after a refactor that doesn't change observable behaviour. If a test breaks because someone renamed a private method, it was testing the wrong thing.

2. **Mock at the boundary, not inside.** vLLM client and FLUX HTTP client get mocked. Internal modules (rules engine, memory tier, tool dispatcher) are exercised for real, with an in-memory SQLite database.

3. **Determinism.** Inject `random.Random(seed)` into rules-engine functions. Use `freezegun` for time-sensitive code. No `time.sleep` in tests; use async wait conditions or test clocks.

4. **No real network calls in CI.** Ever. Real LLM and image API calls happen only in dedicated integration tests gated behind a `pytest -m integration` marker, off by default.

## Setup patterns

### In-memory DB fixture

```python
# tests/conftest.py
@pytest.fixture
async def db_session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    # Apply WAL pragmas via connection event (same as production)
    @event.listens_for(engine.sync_engine, "connect")
    def set_pragmas(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        yield session
```

### Factory fixtures

```python
# tests/factories.py
from app.db import models

async def make_character(db_session, **overrides):
    defaults = {
        "name": "Test Character", "race": "Human", "class_name": "Fighter",
        "level": 1, "hp_current": 8, "hp_max": 8, "ac": 14,
        "str_score": 14, "int_score": 10, "wis_score": 10,
        "dex_score": 12, "con_score": 12, "cha_score": 10,
        "alignment": "Neutral", "user_id": "test-user", "campaign_id": "test-campaign",
    }
    char = models.Character(**(defaults | overrides))
    db_session.add(char)
    await db_session.commit()
    return char
```

### Mocking the LLM client

```python
@pytest.fixture
def mock_llm(monkeypatch):
    responses = []

    class FakeStream:
        def __init__(self, chunks): self.chunks = chunks
        async def __aiter__(self):
            for c in self.chunks: yield c

    async def fake_stream(prompt, tools=None):
        return FakeStream(responses.pop(0))

    monkeypatch.setattr("app.llm.client.stream", fake_stream)
    return responses  # tests append canned chunk lists
```

## Coverage targets

- **Rules engine:** 100% line coverage. Every branch including BFRPG edge cases — natural 1/20, 0 HP via D&D table, max-HP cap on healing, ability score 3 and 18 modifier extremes.
- **Tool handlers:** every tool has a happy path test, an invalid-input test, and a "LLM tried to lie about current state" test (handler reads from DB, not from input).
- **API endpoints:** happy path + auth missing + auth wrong + invalid input.
- **WebSocket:** connect, message round-trip, disconnect cleanup, reconnect snapshot.
- **Migrations:** every migration tested with `upgrade head` → `downgrade -1` → `upgrade head` round-trip.

Don't chase line coverage past 95% — diminishing returns and tests of trivially obvious code.

## Patterns to use

```python
# Sync test for pure rules-engine functions
def test_natural_20_hits_regardless_of_ac():
    rng = random.Random(seed=42)  # produces 20 on first roll
    result = attack_roll(attacker, target_ac=99, weapon=longsword, rng=rng)
    assert result.hit
    assert result.natural_roll == 20

# Async test for tool handlers
@pytest.mark.asyncio
async def test_apply_damage_persists_and_uses_db_value(db_session, mock_llm):
    char = await make_character(db_session, hp_current=10)
    # Note: handler input "claims" HP is 100; handler must ignore that
    await apply_damage_handler(
        db_session,
        {"target_id": char.id, "amount": 3, "source": "goblin"},
        llm_supplied_current_hp=100,  # red herring
    )
    refreshed = await db_session.get(Character, char.id)
    assert refreshed.hp_current == 7  # 10 - 3, not 100 - 3

# WebSocket round-trip
@pytest.mark.asyncio
async def test_ws_pc_action_broadcasts_to_other_players(client_a, client_b):
    await client_a.send_json({"type": "pc_action", "payload": "I draw my sword."})
    msg = await client_b.receive_json()
    assert msg["type"] == "pc_action"
    assert "I draw my sword" in msg["payload"]["content"]
```

## Things NOT to test

- Pydantic model field validation (Pydantic's job, not ours).
- Third-party libraries (httpx, SQLAlchemy, FastAPI). Trust them.
- Implementation details of internal helpers — test through the public function that calls them.

## Reference

- Spec **§6** — rules-engine surface (what to exercise)
- Spec **§7** — LLM integration (what to mock at the boundary)
- Spec **§9** — WebSocket protocol (round-trip targets)
