"""SQLAlchemy ORM models for the spec §5 schema.

Conventions (per spec §5 + AGENTS.md "Database"):

- Primary keys are UUIDv7 strings, generated in Python via the
  ``uuid_extensions.uuid7`` callable, stored as ``TEXT(36)``.
- ``created_at`` / ``updated_at`` are ISO-8601 strings stored as ``TEXT``;
  the server default ``strftime('%Y-%m-%dT%H:%M:%fZ','now')`` writes the
  canonical form. Updates are timestamped at the model boundary, not
  trigger-driven, so the ORM stays the source of truth.
- JSON payloads are stored via SQLAlchemy's ``JSON`` type (which maps to
  ``TEXT`` on SQLite); a ``CHECK(json_valid(col))`` table-level
  constraint enforces parseability at the database level.
- Booleans are stored as ``INTEGER`` 0/1 — SQLAlchemy's default mapping.
- Foreign keys are declared and enforced at the connection level (the
  ``foreign_keys = ON`` pragma in ``app/db/session.py``).
- Embeddings are ``BLOB``; the application is responsible for keeping
  them L2-normalised before insert (AGENTS.md invariant #5).

Cycle-breaking note: ``campaigns.module_id`` and ``modules.source_session_id``
are advisory-only references; declaring them as real foreign keys would
create a three-table cycle (campaigns → modules → sessions → campaigns)
which SQLite cannot express via ALTER TABLE ADD CONSTRAINT. The columns
are plain ``String(36)`` and the application enforces the relationship
when wiring a module into a campaign.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON, Boolean, Integer, Text
from uuid_extensions import uuid7

from app.db.base import Base


def _new_uuid() -> str:
    """UUIDv7 hex with dashes (36 chars). Time-ordered → cheap inserts."""

    return str(uuid7())


# Server-side default expression for ISO-8601 timestamps. SQLite's
# %Y-%m-%dT%H:%M:%fZ format gives millisecond precision and the trailing
# Z that ``datetime.fromisoformat`` accepts.
_NOW = text("(strftime('%Y-%m-%dT%H:%M:%fZ','now'))")

# Static-literal server defaults — values match the DDL ``DEFAULT`` clauses
# in spec §5. These need to be ``text(...)`` expressions, not bare Python
# values, so SQLAlchemy emits them as DDL defaults rather than Python-side
# defaults that bypass raw-SQL inserts.
_FALSE = text("0")
_TRUE = text("1")
_EMPTY_OBJ = text("'{}'")
_EMPTY_ARR = text("'[]'")


# ---------------------------------------------------------------------------
# Users / accounts
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    username: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str | None] = mapped_column(Text, nullable=True)
    pwd_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=_FALSE)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (
        # COLLATE NOCASE on usernames and emails so case variants are the
        # same identity (per spec §5).
        UniqueConstraint("username", name="uq_users_username"),
        UniqueConstraint("email", name="uq_users_email"),
    )


# ---------------------------------------------------------------------------
# Campaigns + membership
# ---------------------------------------------------------------------------


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    owner_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    ruleset: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'bfrpg'"))
    house_rules: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=_EMPTY_OBJ
    )
    world_state: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=_EMPTY_OBJ
    )
    long_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Advisory FK to modules.id — see module-loading flow in spec §10. Not
    # declared as a real ForeignKey to avoid the campaigns/modules/sessions
    # cycle.
    module_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    module_state: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=_EMPTY_OBJ
    )
    image_style: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Per-campaign negative prompt for FLUX requests (spec §8). Prepended
    # to the negative_prompt slot on every /generate and /edit call.
    # Typical content: "modern objects, photographic, watermark, text
    # artefacts, extra fingers". Nullable — campaigns can be created
    # without one and inherit FLUX's empty default.
    image_negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (
        CheckConstraint("json_valid(house_rules)", name="ck_campaigns_house_rules_json"),
        CheckConstraint("json_valid(world_state)", name="ck_campaigns_world_state_json"),
        CheckConstraint("json_valid(module_state)", name="ck_campaigns_module_state_json"),
    )


class CampaignMember(Base):
    __tablename__ = "campaign_members"

    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    role: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint("role IN ('owner','player')", name="ck_campaign_members_role"),
    )


class CampaignInvite(Base):
    """Audit-and-revocation surface for campaign invites (Phase 7).

    Phase 6 invites were stateless ``URLSafeTimedSerializer`` tokens — no
    DB row, no audit, no revocation. Phase 7 promotes them to row-backed
    single-use codes: the signed token now carries ``invite_id`` and the
    redeem path looks up the row to confirm it exists, isn't revoked,
    isn't expired, and hasn't already been used.

    Single-use: once redeemed, ``used_by``/``used_at`` are populated and
    further redemption attempts (even by the same user) return 400. To
    invite a second player, the owner mints a second code. Multi-use
    semantics could be revisited in a later phase if the friction
    actually bites in play.
    """

    __tablename__ = "campaign_invites"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_by: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)
    # ISO-8601 absolute expiry. Spec/Phase 6 default is 7 days from mint.
    # Stored explicitly rather than re-derived from ``created_at`` so an
    # admin can later mint a custom-TTL code without schema changes.
    expires_at: Mapped[str] = mapped_column(Text, nullable=False)
    revoked_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Single-use audit fields. Both NULL until first redemption.
    used_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    used_at: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("idx_campaign_invites_campaign_id", "campaign_id"),)


# ---------------------------------------------------------------------------
# Generated images
# ---------------------------------------------------------------------------


class GeneratedImage(Base):
    __tablename__ = "generated_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_image_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("generated_images.id"), nullable=True
    )
    edit_instruction: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (Index("idx_generated_images_session_id", "session_id"),)


# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    race: Mapped[str] = mapped_column(Text, nullable=False)
    class_name: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    xp: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    hp_current: Mapped[int] = mapped_column(Integer, nullable=False)
    hp_max: Mapped[int] = mapped_column(Integer, nullable=False)
    ac: Mapped[int] = mapped_column(Integer, nullable=False)
    str_score: Mapped[int] = mapped_column(Integer, nullable=False)
    int_score: Mapped[int] = mapped_column(Integer, nullable=False)
    wis_score: Mapped[int] = mapped_column(Integer, nullable=False)
    dex_score: Mapped[int] = mapped_column(Integer, nullable=False)
    con_score: Mapped[int] = mapped_column(Integer, nullable=False)
    cha_score: Mapped[int] = mapped_column(Integer, nullable=False)
    gold: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    alignment: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'alive'"))
    # Free-form status effects — transient conditions applied by the rules
    # engine: "poisoned", "paralyzed", "blessed", "dying", etc. Module-specific
    # effects ("cursed by the shrine", "marked by Vance") are also permitted.
    # Closed enums would prevent module-specific effects.
    # Phase 3 deferred these as "finer status flags"; Phase 6.12 delivers them.
    status_effects: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default=_EMPTY_ARR
    )
    pronouns: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    sheet: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=_EMPTY_OBJ
    )
    canonical_image_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("generated_images.id"), nullable=True
    )
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (
        CheckConstraint("json_valid(status_effects)", name="ck_characters_status_effects_json"),
        CheckConstraint("json_valid(sheet)", name="ck_characters_sheet_json"),
        Index("idx_characters_campaign", "campaign_id"),
    )


# ---------------------------------------------------------------------------
# Inventory + spells known
# ---------------------------------------------------------------------------


class InventoryItem(Base):
    __tablename__ = "inventory_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    character_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("characters.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    item_type: Mapped[str] = mapped_column(Text, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    equipped: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=_FALSE)
    properties: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=_EMPTY_OBJ
    )
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (
        CheckConstraint("json_valid(properties)", name="ck_inventory_items_properties_json"),
    )


class SpellKnown(Base):
    __tablename__ = "spells_known"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    character_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("characters.id", ondelete="CASCADE"),
        nullable=False,
    )
    spell_name: Mapped[str] = mapped_column(Text, nullable=False)
    spell_level: Mapped[int] = mapped_column(Integer, nullable=False)
    prepared: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=_FALSE)


# ---------------------------------------------------------------------------
# Sessions + messages
# ---------------------------------------------------------------------------


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    started_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)
    ended_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_location_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    state: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=_EMPTY_OBJ
    )

    __table_args__ = (CheckConstraint("json_valid(state)", name="ck_sessions_state_json"),)


class SessionMessage(Base):
    __tablename__ = "session_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    sender_kind: Mapped[str] = mapped_column(Text, nullable=False)
    sender_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    audience: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default=_EMPTY_ARR
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    image_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    dice_rolls: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    tool_calls: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (
        CheckConstraint("json_valid(audience)", name="ck_session_messages_audience_json"),
        Index("idx_messages_session_time", "session_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# NPCs + locations + encounters
# ---------------------------------------------------------------------------


class Npc(Base):
    __tablename__ = "npcs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    stats: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=_EMPTY_OBJ
    )
    location_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    alive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=_TRUE)
    canonical_image_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("generated_images.id"), nullable=True
    )
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (CheckConstraint("json_valid(stats)", name="ck_npcs_stats_json"),)


class Location(Base):
    __tablename__ = "locations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("locations.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    location_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict, server_default=_EMPTY_OBJ
    )

    __table_args__ = (CheckConstraint("json_valid(metadata)", name="ck_locations_metadata_json"),)


class Encounter(Base):
    __tablename__ = "encounters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'active'"))
    monsters: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default=_EMPTY_ARR
    )
    initiative: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default=_EMPTY_ARR
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    current_turn: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (
        CheckConstraint("json_valid(monsters)", name="ck_encounters_monsters_json"),
        CheckConstraint("json_valid(initiative)", name="ck_encounters_initiative_json"),
    )


# ---------------------------------------------------------------------------
# World facts (long-term memory) + dice audit log + modules
# ---------------------------------------------------------------------------


class WorldFact(Base):
    __tablename__ = "world_facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    fact: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    tags: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default=_EMPTY_ARR
    )
    importance: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("5"))
    source_session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (
        CheckConstraint("json_valid(tags)", name="ck_world_facts_tags_json"),
        Index("idx_world_facts_campaign", "campaign_id"),
    )


class DiceRoll(Base):
    __tablename__ = "dice_rolls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    actor_kind: Mapped[str] = mapped_column(Text, nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    expression: Mapped[str] = mapped_column(Text, nullable=False)
    individual: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    total: Mapped[int] = mapped_column(Integer, nullable=False)
    purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (
        CheckConstraint("json_valid(individual)", name="ck_dice_rolls_individual_json"),
    )


class Module(Base):
    __tablename__ = "modules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    author_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    min_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tone: Mapped[str | None] = mapped_column(Text, nullable=True)
    estimated_sessions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    image_manifest: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default=_EMPTY_ARR
    )
    # Advisory FK to sessions.id — see cycle-breaking note in module
    # docstring.
    source_session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    public: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=_FALSE)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    created_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False, server_default=_NOW)

    __table_args__ = (
        CheckConstraint("json_valid(content)", name="ck_modules_content_json"),
        CheckConstraint("json_valid(image_manifest)", name="ck_modules_image_manifest_json"),
    )


__all__ = [
    "Campaign",
    "CampaignMember",
    "Character",
    "DiceRoll",
    "Encounter",
    "GeneratedImage",
    "InventoryItem",
    "Location",
    "Module",
    "Npc",
    "Session",
    "SessionMessage",
    "SpellKnown",
    "User",
    "WorldFact",
]
