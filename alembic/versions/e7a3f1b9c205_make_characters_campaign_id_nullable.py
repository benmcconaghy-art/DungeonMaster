"""make characters.campaign_id nullable for roster support

Revision ID: e7a3f1b9c205
Revises: c3f1a2e8d904
Create Date: 2026-05-27 00:00:00.000000

Characters without a campaign_id are "roster" characters — saved
between campaigns. Characters with a campaign_id are actively enrolled
in that campaign. The ondelete is SET NULL so deleting a campaign
returns its characters to the roster rather than destroying them.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e7a3f1b9c205'
down_revision: Union[str, Sequence[str], None] = 'c3f1a2e8d904'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Make campaign_id nullable and change ondelete to SET NULL.

    Uses batch_alter_table (SQLite doesn't support ALTER COLUMN). The
    batch operation recreates the table, picking up the nullable=True
    and SET NULL FK from the updated model definition.
    """
    with op.batch_alter_table('characters', schema=None) as batch_op:
        batch_op.alter_column(
            'campaign_id',
            existing_type=sa.String(length=36),
            nullable=True,
        )


def downgrade() -> None:
    """Reverse: campaign_id NOT NULL with CASCADE.

    Roster characters (campaign_id IS NULL) are deleted first — they
    cannot satisfy the NOT NULL constraint.
    """
    op.execute("DELETE FROM characters WHERE campaign_id IS NULL")
    with op.batch_alter_table('characters', schema=None) as batch_op:
        batch_op.alter_column(
            'campaign_id',
            existing_type=sa.String(length=36),
            nullable=False,
        )
