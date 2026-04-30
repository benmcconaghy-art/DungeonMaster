"""add campaigns.image_negative_prompt

Revision ID: 749179faffc7
Revises: ca912dd07aa8
Create Date: 2026-05-01 06:53:03.663310

Phase 5 image-generation prep. Adds a per-campaign negative-prompt
slot used by the FLUX worker — prepended to FLUX's ``negative_prompt``
field on every ``/generate`` and ``/edit`` request (spec §8 "Style
consistency").

The other Phase 5 schema additions called out in the spec
(``characters.canonical_image_id``, ``npcs.canonical_image_id``,
``generated_images.source_image_id``, ``generated_images.edit_instruction``)
already shipped in the consolidated Phase 1 schema migration.

Single ADD COLUMN, nullable: existing campaigns inherit ``NULL`` and
the worker treats ``NULL`` the same as the empty string.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "749179faffc7"
down_revision: Union[str, Sequence[str], None] = "ca912dd07aa8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("campaigns", schema=None) as batch_op:
        batch_op.add_column(sa.Column("image_negative_prompt", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("campaigns", schema=None) as batch_op:
        batch_op.drop_column("image_negative_prompt")
