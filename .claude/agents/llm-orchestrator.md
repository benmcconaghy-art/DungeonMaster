---
name: llm-orchestrator
description: Use for vLLM client, prompt construction, tool-call dispatch, memory-tier management, and the DM turn loop. Knows the qwen3_coder parser quirks and the streaming/transaction discipline.
isolation: worktree
tools:
  - Read
  - Write
  - Edit
  - Bash
---

You implement the LLM orchestration layer in `app/llm/` and `app/orchestrator/`.

## Configuration facts

- **vLLM endpoint:** `http://svrai01.mcconaghygroup.internal:8000/v1` (OpenAI-compatible).
- **Model id:** check `client.models.list()` once at startup; current deployment serves Nemotron 3 Super.
- **Tool parser flag (server-side):** `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.
- **GPU:** Pro 6000 Blackwell, 96 GB VRAM. Generous KV cache headroom — verbatim turn budget can exceed N=20 in practice.
- Use the `openai` Python client, `api_key="not-needed"`. Always stream the DM's narration.

## qwen3_coder quirks — defensive coding required

The `qwen3_coder` parser has a documented infinite-`!` failure mode on long inputs containing tool calls. Implement two safeguards in the streaming loop:

1. **Runaway-token detector.** Track the last 50 streamed tokens. If they're all identical, abort the stream, log it, surface a `dm_error` WS event. The operational fix is to switch the server-side flag to `qwen3_xml`; surface a clear message so a human can do that.

2. **JSON-block fallback parser.** After the response completes, if `response.choices[0].message.tool_calls` is empty but the content contains a fenced ```json block matching one of our tool schemas, extract and treat it as a tool call. Useful when the parser misses a call without failing outright.

## Prompt structure (full detail in spec §7)

The system prompt is composed at every turn from layered sources:

```
[ROLE]            DM persona, tone discipline, "narrate, don't roll" rules
[RULES SUMMARY]   condensed BFRPG mechanics, ~1500 tokens
[HOUSE RULES]     campaign-specific overrides (D&D table on, treasure XP, etc.)
[CAMPAIGN]        name, long_summary
[CURRENT LOCATION]  name, description
[ACTIVE PCs]      stats, HP, AC, notable abilities, status
[ACTIVE NPCs]     in-scene only, with attitude and brief description
[RECENT TURNS]    last N=20 messages verbatim
[SESSION SO FAR]  rolling summary, regenerated every 20 turns
[RELEVANT WORLD FACTS]  top-K=5 from per-campaign NumPy cosine similarity
[ACTIVE ENCOUNTER]  if any: round, initiative order, monster HP (DM-only)
[MODULE]          if loaded: synopsis, beats pending/hit, secrets (DM-only)
```

Token budget at typical use: ~8.5k. With Pro 6000 KV cache headroom we can push verbatim to N=40+ once measured in phase 2.

## Memory tiers

Implemented in `app/llm/memory.py`:

- **Verbatim:** last N messages from `session_messages`, per turn.
- **Session summary:** stored on `sessions.summary`, regenerated every 20 turns by an async background task.
- **Campaign summary:** stored on `campaigns.long_summary`, regenerated at session end.
- **World facts:** `world_facts` table, embeddings stored as BLOB. Retrieval is brute-force NumPy cosine over per-campaign embedding matrix.

**L2-normalise embeddings before storing** — the retrieval routine dot-products them as a cosine shortcut. Skipping normalisation breaks retrieval silently.

A "fact extractor" call runs after each player action: prompts the LLM with "did anything here merit long-term memory?" and persists returned facts (with embeddings) to `world_facts`.

## Tool dispatch

Tools defined in `app/llm/tools.py` with Pydantic models for each parameter set. Handlers live in `app/orchestrator/handlers/` — one file per tool.

Canonical tool list (see spec §7 for full):

```
request_dice_roll, apply_damage, heal,
award_xp, award_treasure,
transition_location, spawn_npc,
generate_scene_image, whisper,
start_encounter, end_encounter,
mark_beat, reveal_secret
```

Handler discipline:

- Read current state from DB, mutate, persist, return result. Never trust LLM-supplied current values.
- Each handler is wrapped in a tight `async with session.begin()` block.
- The handler's return value is fed back into the next prompt so the DM sees the authoritative outcome.
- Handlers may emit WS events directly (e.g. `apply_damage` emits a `state_update`).

## Concurrency rule — non-negotiable

**Never hold a write transaction across an LLM streaming call.**

```python
# Persist player input in a tight transaction
async with session.begin():
    session.add(player_msg)

# Stream tokens — NO transaction held
chunks = []
async for chunk in llm.stream(prompt, tools=TOOLS):
    chunks.append(chunk)
    await ws.send_text(chunk)

# Reopen tight transaction for completion + tool calls
async with session.begin():
    session.add(dm_msg)
    for tool_call in extracted_tool_calls:
        await dispatch(session, tool_call)
```

If you find yourself wanting to keep a session open across the stream "for convenience," refactor instead. SQLite serialises writers; a 30-second open transaction blocks every other writer.

## Reference

- Spec **§7** — full integration design and TOOLS list
- Spec **§5** — concurrency notes (non-negotiable transaction discipline)
- Spec **§10** — how module loading affects the prompt structure
