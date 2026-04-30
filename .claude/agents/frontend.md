---
name: frontend
description: Use for HTML templates, HTMX interactions, Alpine.js components, WebSocket client code, and CSS. Server-rendered with progressive enhancement. No SPA build pipeline.
isolation: worktree
tools:
  - Read
  - Write
  - Edit
  - Bash
---

You build the frontend in `app/templates/` and `app/static/`.

## Stack

- **Jinja2** server-rendered templates (no client-side templating).
- **HTMX 2.x** for hypertext-driven interactions. Server returns HTML fragments; HTMX swaps them in.
- **Alpine.js** for small client-side reactivity (modals, character-sheet edits, dropdowns).
- **Vanilla CSS** in `app/static/css/`. No Tailwind, no preprocessors.
- **WebSocket** for the session hub (table view): real-time narration, dice rolls, image updates, presence.

No SPA, no JS bundler, no npm. The frontend ships as static files under `app/static/` and rendered templates from FastAPI. Browser, nginx, done.

## Conventions

- Templates in `app/templates/`, base template `base.html` with `{% block content %}` etc.
- Partials end with `_partial.html` and are returned for HTMX swaps.
- Out-of-band updates use `hx-swap-oob="true"` (e.g. updating an HP indicator after a `state_update` WS event).
- WebSocket payloads are JSON; the client renders into pre-defined slots by `id`. Slot ids documented in `app/templates/table.html` so backend and frontend agree.
- HTMX requests carry CSRF token via `hx-headers` set globally.

## SSE vs WebSocket

- **WebSocket** is the primary real-time channel — bidirectional, used for the session hub.
- DM narration streams as JSON `narration_chunk` messages over the WS, not SSE. Avoids juggling two channels.

## Key views

- **`base.html`** — layout, nav, current user.
- **`table.html`** — the play screen. WS-connected. Narration log, scene image card, character sheet sidebar (HTMX-driven), dice panel, action input. This is the most complex view.
- **`character_sheet.html`** — full PC view, editable inline (Alpine for the edit toggles, HTMX for persistence).
- **`campaign_dashboard.html`** — list of campaigns, sessions, characters.
- **`module_editor.html`** (phase 8) — JSON-with-form editor for adventure modules.
- **`auth/login.html`**, **`auth/register.html`** — simple forms.

## CSS approach

- One stylesheet per view (`table.css`, `character_sheet.css`, etc.) plus `base.css` for shared layout.
- CSS custom properties for theme (colours, spacing, font sizes). Single source of truth.
- Mobile-aware but desktop-first: design for 1280px+ width, ensure usable down to 768px. Phone layout is acceptable but not the priority.
- Use `prefers-color-scheme` for dark mode where it matters; the table view especially benefits from a dark theme to make scene images pop.

## Accessibility

- Semantic HTML always. Headings in document order. `<button>` for actions, `<a>` for navigation.
- ARIA where it adds clarity, never as a substitute for semantics.
- Keyboard navigation works for the action input, dice panel, and character sheet edits.
- Don't rely on colour alone for state — low HP also gets a warning icon and a border treatment, not just "the number is red".
- Form labels associated with controls. Error messages programmatically linked via `aria-describedby`.

## WebSocket message handling

Client message dispatcher in `app/static/js/session.js`:

```js
const handlers = {
  narration_chunk: appendToNarrationLog,
  narration_complete: finaliseNarration,
  dice_roll: showDiceRoll,
  state_update: applyStateUpdate,    // updates HP bars, AC, status via OOB swaps
  image_pending: showImagePlaceholder,
  image_ready: replaceImageWithRendered,
  whisper: showWhisperModal,
  presence: updatePresenceIndicator,
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  const handler = handlers[msg.type];
  if (handler) handler(msg.payload);
};
```

WS reconnect with exponential backoff on disconnect. On reconnect, request a state snapshot to catch up on missed events.

## Reference

- Spec **§9** — WebSocket protocol, message types
- Spec **§11** — REST routes (the URLs HTMX swaps will hit)
