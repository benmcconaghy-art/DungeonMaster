"""Handler for the ``reveal_secret`` tool.

Records that a module secret has come out in play. The DM calls this when
the narrative moment described by the secret's reveal_when has occurred.

Same discipline as mark_beat: LLM-judged timing, engine validates the ID,
already-revealed secrets are a no-op with a structured note.

Campaign must have module_state populated (i.e. be loaded from a module).
Non-module campaigns return an informative error.
"""

from __future__ import annotations

from app.db.models import Campaign
from app.db.models import Session as DmSession
from app.llm.tools import RevealSecret, ToolResult, register
from app.orchestrator.context import current_context
from sqlalchemy.ext.asyncio import AsyncSession


@register("reveal_secret")
async def handle(db: AsyncSession, args: RevealSecret) -> ToolResult:
    ctx = current_context()
    session = await db.get(DmSession, ctx.session_id)
    if session is None:
        return ToolResult(
            content="reveal_secret failed: active session not found.",
            side_effects={"kind": "error", "reason": "unknown_session"},
        )

    campaign = await db.get(Campaign, session.campaign_id)
    if campaign is None:
        return ToolResult(
            content="reveal_secret failed: campaign not found.",
            side_effects={"kind": "error", "reason": "unknown_campaign"},
        )

    module_state: dict = campaign.module_state or {}
    if not module_state or not module_state.get("module_id"):
        return ToolResult(
            content=(
                "reveal_secret failed: this campaign is not loaded from a module. "
                "Secret tracking requires a module-backed campaign."
            ),
            side_effects={"kind": "error", "reason": "no_module"},
        )

    secrets_revealed: list[str] = list(module_state.get("secrets_revealed", []))

    # Build the full set of known secrets from the module content.
    # We need to validate the secret_id against the module's secret symbols.
    symbolic_id_map: dict[str, str] = module_state.get("symbolic_id_map", {})
    secret_id = args.secret_id

    # Check if this secret symbol exists in the module (it should be in the symbolic_id_map
    # or in the module content). We validate via presence in symbolic_id_map — the loader
    # populates an entry for every sec_ symbol at load time.
    known_secrets = {k for k in symbolic_id_map if k.startswith("sec_")}

    if secret_id not in known_secrets:
        return ToolResult(
            content=(
                f"reveal_secret failed: secret_id {secret_id!r} is not a known secret "
                f"in this module. "
                f"Known secrets: {', '.join(sorted(known_secrets)) or '(none)'}."
            ),
            side_effects={"kind": "error", "reason": "unknown_secret", "secret_id": secret_id},
        )

    # Idempotent: already revealed → structured no-op.
    if secret_id in secrets_revealed:
        return ToolResult(
            content=f"Secret {secret_id!r} was already revealed. No state change.",
            side_effects={
                "kind": "secret_already_revealed",
                "secret_id": secret_id,
            },
        )

    secrets_revealed.append(secret_id)

    updated_state = {
        **module_state,
        "secrets_revealed": secrets_revealed,
    }
    campaign.module_state = updated_state
    await db.flush()

    return ToolResult(
        content=f"Secret {secret_id!r} revealed and recorded.",
        side_effects={
            "kind": "secret_revealed",
            "secret_id": secret_id,
        },
    )


__all__ = ["handle"]
