"""add pronouns and description to characters

Revision ID: c3f1a2e8d904
Revises: b4e1bcdf4bff
Create Date: 2026-05-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3f1a2e8d904'
down_revision: Union[str, Sequence[str], None] = 'b4e1bcdf4bff'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable pronouns and description columns to characters."""
    op.add_column('characters', sa.Column('pronouns', sa.Text(), nullable=True))
    op.add_column('characters', sa.Column('description', sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove pronouns and description columns from characters."""
    op.drop_column('characters', 'description')
    op.drop_column('characters', 'pronouns')
