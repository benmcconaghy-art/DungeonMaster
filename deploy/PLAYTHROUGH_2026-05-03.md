# Playthrough Findings — 2026-05-03

First real playthrough of the Dungeon Master system. Two sessions
in one campaign ("Testing"), Halfling Thief (Lila Lockpick) and
Dwarf Fighter (Slowhand). Approximately 90 minutes of real play
across multiple sessions. Several bugs were discovered and fixed
in real-time; others remain as architectural prep material for
Phase 8.

## Fixed during playthrough

### Phase 5 close-out (4ba0233): image serving route missing
Symptom: character portraits and scene images returned 404 in
the browser despite being persisted to disk and database.
Cause: Phase 5 added the FLUX worker, schema, and templates,
but the FastAPI route to serve generated images at
`/api/images/{id}.png` was never wired. Spec §8 mentioned
X-Accel-Redirect for production; the dev path was simply
missing.
Fix: GET /api/images/{id}.png handler with FileResponse,
campaign-membership auth, path-traversal defence, 404-on-no-
permission to avoid spoilers.

### Phase 6.5 (6f70058): chargen UI didn't exist
Symptom: dashboard's "Roll new character" affordance pointed
at a route that didn't exist. Players had to use curl directly
to create characters.
Cause: Phase 6 polished the affordances but the underlying
chargen view was deferred without a clear trigger to land it.
Real play surfaced it as immediately blocking.
Fix: app/templates/chargen.html with progressive-reveal
sections (abilities → race → class → alignment → name).
POST /api/chargen/roll-abilities for server-side rolling.
Eligibility computed client-side from data tables; mechanics
stay server-authoritative.

### Phase 6.6 (bf8b6ae): "Start Session" was a non-interactive span
Symptom: clicking Start Session on the dashboard did nothing.
Cause: the design's resume-session form had an internal
button only on the arrow icon; the rest of the card was
non-interactive. Both the resume and start variants had the
same gap.
Fix: form wraps the entire card body as a real button.
Empty-state and active-state both verified.

### Phase 6.8 (7a456ce): four playthrough bugs in one commit
- Bug 1 (streaming UI ghost bubbles): per-iteration
  stream_id added to NarrationChunk/NarrationComplete, JS
  dispatcher keyed by stream_id rather than frame adjacency.
- Bug 2 (dice parser): regex extended to accept `NdM*K`,
  `NdM+K`, `NdM-K`, parens optional. BFRPG starting-gold
  idiom `3d6*10` now parses.
- Bug 3 (auto-greeting): take_turn(opening=True) auto-fires
  on session creation, persists [Session begins] as a
  system-role message. Shared dispatch helper extracted to
  app/orchestrator/dispatch.py.
- Bug 4 (transition_location ID exposure): tool now accepts
  location_id OR name; handler does fuzzy-match-or-create.
  System prompt explicitly forbids asking the player for IDs.

## Open — still surfacing during play

### Tool-error history poisons subsequent prompts
Symptom (one form): malformed tool args (e.g. dict literal
instead of JSON) get parsed unsuccessfully, the error result
gets embedded in the conversation history, vLLM rejects the
next completion request with HTTP 400 because the message
history contains a malformed tool-call. Session wedges; only
recovery is ending the session.
Symptom (another form): tool dispatch succeeds the first
time, fails the second time (e.g. dice expression `3d6*10`
in the Phase 5 era), failure history bloats the prompt and
biases the model toward over-reasoning, leading to empty
completions on subsequent turns.
Architectural shape: the orchestrator's "graceful tool-error"
pattern (catch the error, surface it as a tool result, let
the model retry) assumes the error result is benign content.
In practice, certain error shapes either (a) make the next
prompt invalid, or (b) pollute the prompt enough to
destabilise generation.
For Phase 8: needs a "scrub error history before next prompt"
or "limit tool-error retries to N before falling back to a
clean retry without history" pattern. Modules will be exposed
to this — every module session has tool calls.

### Memory pacing across session boundaries
Symptom: cross-session continuity worked for major NPCs and
locations (Jeb the smith, his smithy) but felt thin —
opening scenes set the place but didn't carry forward
mid-session details (the price negotiation, the specific gold
counts). Session-end summarisation may not have fired on
manual session-close (`summary` was empty in sessions table).
For Phase 8: modules need explicit "load-state" semantics
distinct from continuation. The session-end summariser path
needs to be reliable. Worth verifying it actually runs on
clean session-end (UI-driven) vs forced session-end (DB
update).

### Whisper UX never naturally surfaced
Symptom: across 90 minutes of play with multiple "secret
observation" prompts ("Slowhand looks for tracks no one else
would notice"), no whispers occurred organically. The whisper
feature works mechanically (verified in tests), but the DM's
prompt isn't pushing it to use whispers as a first-class
narrative tool.
For Phase 8: prompt engineering. The DM should be encouraged
to use whispers for private observations, secret information,
and player-specific consequences. May need example shots in
the system prompt or explicit instructions about when whispers
are appropriate.

### Multi-character commerce wedges Nemotron
Symptom: scenes involving multiple PCs interacting with a
single NPC over a transactional task (Slowhand and Lila both
buying equipment from Jeb) trigger empty-completion cascades.
Single-PC interactions resolve cleanly; commerce specifically
seems to wedge.
Hypothesis: combination of multi-character coordination +
arithmetic (gold, prices, change) + tool dispatches (dice for
discount checks) + reasoning_mode=full creates over-reasoning
loops that exhaust the response budget.
For Phase 8: not directly module-related but exacerbated by
modules with shop NPCs. May want a different reasoning_mode
for transactional scenes, or explicit prompt structure for
"resolve commerce in narration without mechanical adjudication
unless dice are needed."

### Auto-greeting works but error path is silent
Symptom (potential): if the auto-greeting turn fails (empty
completion, vLLM error), the player sees the same "preparing
the scene" placeholder forever with no recovery path.
Not observed in this playthrough — the auto-greeting worked
on every fresh session — but the failure mode is theoretically
present.
For Phase 8 prep: surface auto-greeting failures to the
client as an explicit error state with a manual retry option.

## Architectural observations

### Phase 7 observability paid off
Every bug above was diagnosable from the structured logs
without spelunking. request_id propagation, llm_complete
records with prompt_tokens/completion_tokens/outcome, the
tool dispatch logs — all surfaced cause-and-effect cleanly.
Without Phase 7's logging, several of these bugs would have
been "the system is acting weird" rather than "Nemotron
returned a 400 on the next request because the prior tool
call's args were malformed."

### Design system held in real use
Phase 6's polish work survived contact with reality. The
table view's narration prose is genuinely pleasant to read.
The character sheet renders BFRPG data correctly. The chargen
flow felt natural despite being a same-day add. Nothing
visual needed correction during play.

### "Verified in tests" vs "verified in play"
Five of the seven fixed bugs were not caught by the test
suite. Phase 7's PHASE_7_VERIFICATION.md runbook anticipated
this distinction; this playthrough confirmed it. The pattern:
tests verify the happy path of code paths the developer
thought of; play verifies whatever the system actually does.
Both matter.

### Empty states need explicit tests
The "Start Session" button bug (6.6) and the chargen 404
(6.5) were both empty-state bugs — the populated case worked,
the empty case wasn't wired. Phase 6.6 added a Code
Conventions note: "for any conditional affordance, test both
branches." This playthrough validates that as durable
guidance.
