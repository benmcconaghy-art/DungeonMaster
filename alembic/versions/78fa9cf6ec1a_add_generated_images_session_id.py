"""add generated_images.session_id

Revision ID: 78fa9cf6ec1a
Revises: 749179faffc7
Create Date: 2026-05-01 10:32:50.701017

Phase 6 prep. Adds an optional ``session_id`` reference to
``generated_images`` so the WS snapshot can replay recent image events
for a reconnecting client by session.

The image worker already carries ``session_id`` on its in-flight
``ImageJob`` payload (used for the live broadcast); this migration plus
the model + worker changes simply persist it on the row.

Design notes:

- ``ondelete='SET NULL'`` is intentional. Generated images outlive any
  one session: they belong to the campaign and may be referenced by
  ``characters.canonical_image_id`` / ``npcs.canonical_image_id``.
  Deleting a session must not cascade-delete its images.
- The FK is added inside ``batch_alter_table`` because SQLite's plain
  ``ALTER TABLE`` cannot add FK constraints — batch performs the
  standard 12-step recreate.
- Index follows the project convention ``idx_<table>_<columns>``;
  needed for the snapshot's ``WHERE session_id = ?`` lookup to stay
  cheap as the table grows.
- The FK constraint is given an explicit name so the downgrade can
  drop it deterministically; the index lookup name lives at module
  scope to keep upgrade and downgrade in sync.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "78fa9cf6ec1a"
down_revision: Union[str, Sequence[str], None] = "749179faffc7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_INDEX_NAME = "idx_generated_images_session_id"
_FK_NAME = "fk_generated_images_session_id_sessions"


def upgrade() -> None:
    with op.batch_alter_table("generated_images", schema=None) as batch_op:
        batch_op.add_column(sa.Column("session_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            _FK_NAME,
            "sessions",
            ["session_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(_INDEX_NAME, ["session_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("generated_images", schema=None) as batch_op:
        batch_op.drop_index(_INDEX_NAME)
        batch_op.drop_constraint(_FK_NAME, type_="foreignkey")
        batch_op.drop_column("session_id")
