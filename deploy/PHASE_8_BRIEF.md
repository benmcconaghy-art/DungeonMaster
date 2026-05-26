Phase 8 — Adventure Modules. Fresh CC session, but the
architecture is settled from a prior session's read-back.
Implement directly per the plan below.

Foundation: Phase 7 (commit 2a5d658) plus eight playthrough
close-out commits (Phase 5 image route, 6.5 chargen, 6.6
start session, 6.8 streaming/dice/auto-greet/location, 6.9
tool-error hygiene, 6.10 speaker attribution, 6.11 empty
completion recovery + bubble suppression, 6.12 revival and
status tools, 6.13 character presentation). The system
survives real play; AGENTS.md has 18 Critical Invariants.
Phase 8 is the final architectural phase per spec §14.

## Step 1: Read

1. AGENTS.md in full. All 18 Critical Invariants and the
   Follow-ups parking lot matter for this phase.

2. dungeon-master-spec.md sections §3, §6, §7, §10, §14.
   §10 is the canonical module spec; §14 is the phase
   roadmap.

3. The playthrough findings docs in deploy/ (PLAYTHROUGH_
   2026-05-03.md and any subsequent findings files).

4. Survey for context:
   - app/db/models.py — Campaign, Location, NPC,
     Character, SessionMessage, GeneratedImage. The
     modules table and campaigns.module_id should already
     exist as stubs.
   - app/orchestrator/dm.py — take_turn loop with Phase
     6.9's _classify_tool_call gate, Phase 6.11's two-tier
     empty-completion recovery
   - app/llm/prompts.py — _ROLE_TEXT, _render_system,
     _render_characters
   - app/llm/tools.py — mark_beat and reveal_secret stubs
     with implemented=False
   - app/orchestrator/handlers/ — current handler patterns
   - app/api/campaigns.py — campaign creation
   - data/bfrpg/ — existing static data

5. Confirm the surface you'll work against matches what's
   described below. If something has shifted (e.g. the
   modules stub schema is different from spec), flag it
   before implementing.

## Step 2: Architecture (settled, do not re-derive)

This was settled in a prior session. Confirm you can
implement it; flag any blockers, but don't re-argue the
design space.

### Module schema (Pydantic v2 ModuleContent)
{
"format_version": "1.0",
"synopsis": "...",
"tone": "...",
"image_style": "...",
"image_negative_prompt": "...",
"level_range": [1, 3],
"estimated_sessions": 6,
"starting_hook": "...",
"starting_location_symbol": "loc_...",
"locations":  [{"symbol", "name", "description",
"parent_symbol"?, "image_role"?,
"metadata"}],
"npcs":       [{"symbol", "name", "description",
"motivation", "starting_location_symbol",
"stats", "sample_dialogue",
"image_role"?, "secrets"?}],
"encounters": [{"symbol", "name", "trigger_hint",
"monsters": [{"name", "count", "tactics"}],
"treasure_hint"}],
"plot_beats": [{"symbol", "title", "trigger_hint",
"outcome", "leads_to"?, "dm_notes"}],
"secrets":    [{"symbol", "content", "reveal_when",
"leads_to_beat"?}],
"endings":    [{"symbol", "trigger", "outcome"}],
"world_facts": [{"fact", "tags", "importance": 1..10}]
}

Symbolic IDs (snake_case, namespaced: loc_, npc_, enc_,
beat_, sec_, end_) live in module JSON. Loader mints fresh
UUIDv7 per symbol; writes campaigns.module_state.symbolic_
id_map for runtime resolution. Beat/secret tools use
symbols, not UUIDs.

image_manifest (sibling JSON column, per spec): list of
{"role": "npc:npc_castellan", "prompt", "params",
"prompt_hash"}. Portable shape — prompt + params + hash,
NO foreign key to GeneratedImage. Same-install reload wins
dedup via prompt_hash; cross-install regenerates from prompt.
This matches spec §10's portability contract.

campaigns.module_state:
{
"module_id": "<uuid>",
"symbolic_id_map": {"loc_gates": "<uuid7>", ...},
"beats_hit": [],
"beats_pending": ["beat_arrive", ...],
"secrets_revealed": [],
"encounters_run": [],
"endings_reached": []
}

Spoiler discipline: player API serialisers never expose
module.content.secrets or plot_beats[i].dm_notes. DM system
prompt's module section receives them.

### Extraction

Manual trigger. POST /api/sessions/{id}/extract-module
(campaign owner only, requires ended_at IS NOT NULL).
Handler:
1. Read session_messages + locations + npcs + encounters +
   world_facts + characters.
2. Build tightly-structured prompt per spec §10.
3. Call client.complete() with reasoning_mode="full"
   (extraction is salience judgement; matches the fact
   extractor pattern).
4. JSON-validate against ModuleContent. On ValidationError,
   retry with corrective prompt up to 3 retries total.
5. On success: insert modules row with author_id=user.id,
   source_session_id=session_id, public=false.

Privacy at extraction: prompt strips played-specific PCs/
dice/chat. Filter session_messages to sender_kind IN
('player','dm') so other system notes don't leak.

### Loading

POST /api/campaigns/from-module taking {module_id, name,
image_style_override?}.

Transaction:
1. Validate module.content against ModuleContent.
2. Insert Campaign with module_id, empty module_state.
3. Mint UUIDv7 per symbol; build symbolic_id_map.
4. Insert locations (parent_id via map), NPCs (location_id
   via map). Encounters deferred — only instantiated when
   start_encounter fires.
5. Insert world_facts, embed each.
6. Initialise module_state with all beats pending,
   symbolic_id_map populated.
7. Insert CampaignMember owner row.
8. Commit.

Post-commit (out-of-transaction):
9. For each image_manifest entry: query GeneratedImage by
   prompt_hash; if found, link via canonical_image_id /
   image_id; if not, enqueue ImageJob. Worker dedups by
   hash.
10. Return 201 immediately. First-load is ~32 portraits ×
    ~17s = ~9 min; user can enter the campaign immediately
    via image_pending placeholders.

Idempotence: loading into a campaign with module_id already
set returns 409.

### Beat tracking

Two handlers in app/orchestrator/handlers/:
- mark_beat.py: validate beat_id in beats_pending (no-op +
  structured note if already hit), move to beats_hit,
  persist. ToolResult with side_effects {kind:
  "beat_marked", beat_id, summary}.
- reveal_secret.py: same shape for secrets.

Both go through _classify_tool_call. Flip implemented=True
in TOOLS. DM uses symbolic IDs (beat_jeb_tip), validates
against module_state.

LLM-judged: system prompt tells DM "call when triggered" —
no mechanical conditions.

### System prompt module section

New _render_module_section(db, campaign) helper, keyed off
campaign.module_id. Renders into the [MODULE] block of
_render_system:
- Synopsis + tone + image_style
- [KEY NPCs IN THIS MODULE]: each NPC's name, description,
  starting location, current state
- [KEY LOCATIONS]: name, description, current state
- [PLOT BEATS — PENDING]: title + trigger_hint per pending
  beat, with symbolic ID
- [PLOT BEATS — HIT]: short list of already-fired beats
- [SECRETS — DM-ONLY DO NOT REVEAL]: pending secrets with
  reveal_when
- [SECRETS — REVEALED]: secrets that have come out (no
  longer DO NOT REVEAL)
- [GUIDANCE]: when to call mark_beat / reveal_secret

## Step 3: Architectural fixes integral to module work

Four issues the playthrough surfaced. They block module
work and land first.

### Commit 1: Prompt revisions (E.1 + E.3 + E.4)

Edit _ROLE_TEXT in app/llm/prompts.py:

E.1 — Scene art guidance. New IMAGES block:
"Call generate_scene_image when the party first arrives at
a major location, when a climactic beat fires, or when a
dungeon-room reveal lands. Use generate_scene_image with
kind='scene' for environments and kind='npc' (via
spawn_npc's auto_portrait) for character introductions.
Don't call generate_scene_image for character portraits —
spawn_npc handles that. Scene art evokes mood; don't
generate one per beat."

E.3 — Length/pacing. Strengthen PACING block:
"Respond in 2-4 short paragraphs (~200-350 words). One
beat per response. Compress, don't truncate — wrap on a
natural sentence boundary rather than mid-thought."

Also: drop max_tokens from 1024 → 768 on the ordinary
streaming path. The Phase 6.11 recovery retry path stays
at 2048 (critical — do not unify these).

Also: markdown renderer client-side via markdown-it from
a static asset (~50KB). Accumulate plain text during
narration_chunk; render to HTML at narration_complete.
Do NOT add a Python markdown dep — this is presentation
layer, belongs client-side.

E.4 — Whisper UX. New WHISPERS block:
"Use whisper for class-specific observations (Thief detects
tracks no one else sees; Cleric senses unholy presence),
private NPC tells (a barkeep's nod, a stranger's wink),
character-specific consequences (a curse only the affected
character feels). Default to public narration; whisper is
for moments where information should NOT be common
knowledge. When a player makes a perception/listen check
whose result is private, narrate publicly that they look,
then whisper the private detail."

Tests: prompt-snapshot assertions. No integration test
needed.

After commit: restart dev server, 10-minute play sanity
check to verify the prompt changes don't regress empty-
completion rates.

### Commit 2: NPC roster panel (E.2)

New right-rail panel between Party and Rolls:
<section class="panel npcs"> showing per-NPC card
(portrait + name + race-or-kind + brief).

Populated via new npc_introduced WS message broadcast on
spawn_npc side-effect. Message is self-contained: carries
portrait URL, name, race/kind, brief description inline
(NOT just an NPC ID for the client to fetch separately).
Client accumulates introductions in the roster.

Inline NPC portraits in narration disappear entirely — the
rail surfaces them. Scene art stays inline as the only
narration-flow visual.

Same parchment-card visual as Party panel. Mobile: rail
collapses below as it already does.

Tests: rendering test for panel, dispatch test for
npc_introduced WS message.

After commit: restart, sanity check NPC introductions
populate the rail and don't leak inline.

## Step 4: Module commits

### Commit 3: ModuleContent schema + handlers + tests

- Pydantic v2 ModuleContent in app/llm/modules.py
- mark_beat and reveal_secret handlers
- Register handlers, flip implemented=True in TOOLS
- _render_module_section wired into _render_system

Unit tests for handlers + prompt rendering. No integration
play yet — module_state won't be populated until commit 4
lands the loader.

### Commit 4: Module loader + tests

POST /api/campaigns/from-module + symbolic-id-map building
+ image dedup-or-enqueue + module-aware prompt.

npcs_in_scene snapshot inclusion populated from loaded
module's symbol map (NPCs whose canonical portrait has
been image_ready'd in this session — accumulated client-
side from npc_introduced messages).

Integration test: load a small fixture module, verify
campaign has locations + NPCs + image jobs enqueued +
module_state populated.

Dev sanity check: load a 3-location fixture, eyeball that
portraits start streaming back.

### Commit 5: Module extraction + tests

POST /api/sessions/{id}/extract-module + LLM prompt builder
+ JSON validation + retry-on-ValidationError up to 3 times.
reasoning_mode="full".

Integration test mocks LLM with hand-shaped valid response
plus a malformed-then-valid response.

No dev play required yet.

### Commit 6: data/bfrpg/modules/morgansfort.json + bootstrap script

Shape:
- Morgansfort itself: 6-8 locations (gate, common hall,
  smithy, chapel, market, mayor's office, barracks, jail)
- Greenhill Caves (BFRPG-canonical goblin caves): 5-7
  locations (entrance, sentry post, common burrow, shaman's
  chamber, treasure pit, secret tomb-room)
- Wraith Tomb / Vance's hold: 4-5 locations (forest
  approach, gatehouse, audience hall, sanctum, tomb beneath)

~30 NPCs: 4-5 keep notables (Castellan Thorvald, smith Jeb,
mother Serra cleric, captain Audrik, trader Magda); 6-8
villagers; 10 goblins/shamans/champion; 5-7 Vance's
followers; Lord Vance himself (fallen paladin, undead).

5 plot beats:
1. arrival_briefing — Castellan hires the party
2. goblin_tip — Jeb/Serra/Magda lets slip something about
   Vance's name
3. caves_secret — discover goblin caves are an old elven
   tomb / Vance's pawns
4. vance_revealed — Vance's fall from paladinhood becomes
   known
5. vance_confronted — final encounter at the wraith tomb

3 endings: clean (Vance defeated, redeemed) / grey (Vance
defeated, leaves corruption trail) / tragic (Vance escapes,
returns).

Tone: gritty per spec. BFRPG monsters from data/bfrpg/
monsters.yaml. Equipment from data/bfrpg/equipment.yaml.
No invented mechanics.

Write as JSON directly using ModuleContent schema. Do NOT
prose-then-translate. Validate against the schema before
commit. Consider landing in two passes if the content
volume is unwieldy: locations + NPCs first, then beats +
secrets + endings + world_facts. Your call.

Bootstrap script: uv run python -m app.scripts.load_module
morgansfort that validates and registers Morgansfort as a
system-owned module on first server startup.

### Commit 7: Validation playthrough

Play one full Morgansfort session (~60-90 min on Ben's
end). Verify at least one beat fires (DM calls mark_beat),
at least one secret reveals, scene art generates, NPCs
surface in the rail.

Open deploy/PLAYTHROUGH_PHASE_8.md with findings. Fix
blockers in this commit cluster; defer non-blockers to
Follow-ups.

### Commit 8: Round-trip validation

Extract a module from the played session, verify it
validates, reload into a fresh campaign, play 15 min,
confirm extraction produces coherent JSON.

Document findings in PLAYTHROUGH_PHASE_8.md.

### Commit 9: Documentation and close-out

- AGENTS.md current build phase → "Phase 8 complete"
- New Critical Invariants:
  - #19: module symbolic-id discipline — modules reference
    entities by symbol, never UUID
  - #20: module load is idempotent on prompt_hash for
    images and on (module_id, campaign_id) for module_state
  - #21: module beats are LLM-judged: schema specifies
    trigger_hint, not mechanical conditions
- README upgrade: how to load a module, how to extract,
  how to author
- Spec rev to v0.8 capturing what Phase 8 actually shipped

Don't do public-release prep (screenshots, demo deck,
license review). Phase 9 may or may not happen; close
Phase 8 cleanly.

## Constraints

- Module schema is JSON, single-file, human-editable. Not
  SQLite, not pickle, not protobuf.
- Loading a module is idempotent across campaigns. Same
  module → two distinct trees in two campaigns.
- Image dedup on load via prompt_hash. Reloading the same
  module twice doesn't double image count on disk.
- Privacy/spoiler discipline. Secrets and dm_notes never
  reach player API serialisers.
- Beat tracking is LLM-judged. trigger_hint is natural-
  language guidance only.
- Reuse existing tool dispatch infrastructure. New tools
  route through the Phase 6.9 _classify_tool_call gate.
- Morgansfort exercises BFRPG specifically. Don't invent
  module mechanics that BFRPG doesn't support.
- Defer the rich editor UI past Phase 8 per spec §10
  stretch. JSON in/out via PATCH /api/modules/{id} +
  GET /api/modules/{id} is enough for v1.

## Stopping criteria

Stop and ask if:
- The modules table or campaigns.module_id stub doesn't
  exist or has a shape different from this brief assumes.
- Architectural fix #2 (NPC roster) requires UI changes
  larger than expected.
- Beat tracking's LLM-judged approach is unreliable in
  early testing.
- Morgansfort authoring takes substantially longer than
  estimated.

Otherwise: confirm you can implement the plan as-described,
then proceed. Checkpoint after each commit so Ben can
validate before the next lands.

## Validation criteria for Phase 8 done

- Morgansfort plays end-to-end with at least one beat
  fired (DM correctly calls mark_beat)
- Extraction from a played session produces valid JSON
  that re-loads cleanly
- The "DM doesn't lead the story" feeling that surfaced
  in earlier playthroughs is structurally resolved —
  module-loaded sessions feel like a real adventure is
  being run, not improvised on the fly

This brief comes from a prior session's read-back that
explored alternatives and settled on the design above.
Don't re-derive; implement. Flag genuine blockers but
trust the architecture decisions absent counter-evidence.
