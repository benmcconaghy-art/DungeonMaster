"""phase 7 add campaign_invites

Revision ID: a96d3a6e501d
Revises: 78fa9cf6ec1a
Create Date: 2026-05-01 12:00:49.951555

Phase 7 hardening. Promotes campaign invites from stateless
``URLSafeTimedSerializer`` blobs (Phase 6) to row-backed single-use
codes with audit trail and revocation. The signed token now carries
``invite_id``; the redeem path looks up the row to confirm it exists,
isn't revoked, isn't expired, and hasn't already been used.

Single-use semantics: once redeemed, ``used_by`` / ``used_at`` are
populated and further redemption attempts return 400 (handled in
``app.api.campaigns.join_via_invite``). To invite a second player,
the owner mints a second code.

Existing in-flight Phase 6 tokens (no ``invite_id``) are accepted via
a 7-day grace path in the redeem handler; after that, they 400. This
migration only creates the table — the grace lives in code so the
cutoff is explicit and reviewable.

The FK constraints follow project convention. ``ON DELETE CASCADE``
on the ``campaigns.id`` reference keeps the table self-cleaning when
a campaign is deleted; ``users.id`` references intentionally do NOT
cascade — deleting a user shouldn't silently remove the audit trail
of who minted/redeemed.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a96d3a6e501d"
down_revision: Union[str, Sequence[str], None] = "78fa9cf6ec1a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the ``campaign_invites`` table + its lookup index."""

    op.create_table(
        "campaign_invites",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("campaign_id", sa.String(length=36), nullable=False),
        sa.Column("created_by", sa.String(length=36), nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ','now'))"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.Text(), nullable=False),
        sa.Column("revoked_at", sa.Text(), nullable=True),
        sa.Column("used_by", sa.String(length=36), nullable=True),
        sa.Column("used_at", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["campaign_id"],
            ["campaigns.id"],
            name="fk_campaign_invites_campaign_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name="fk_campaign_invites_created_by",
        ),
        sa.ForeignKeyConstraint(
            ["used_by"],
            ["users.id"],
            name="fk_campaign_invites_used_by",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("campaign_invites", schema=None) as batch_op:
        batch_op.create_index("idx_campaign_invites_campaign_id", ["campaign_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("campaign_invites", schema=None) as batch_op:
        batch_op.drop_index("idx_campaign_invites_campaign_id")
    op.drop_table("campaign_invites")
