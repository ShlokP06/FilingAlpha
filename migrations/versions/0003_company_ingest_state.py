"""company ingest state

Add per-firm ingestion lifecycle tracking so a batch interrupted by Yahoo/SEC
rate limits never leaves a firm looking "done" with incomplete data. A firm is
only ``complete`` once both filings and prices land; ``no_data`` is terminal
(delisted/illiquid), ``failed`` is retried on the next run.

The upgrade backfills existing rows honestly: any company that already has price
rows is marked ``complete``; everything else stays ``pending`` and will be
reprocessed by ``scripts/ingest_batch.py``. This auto-heals partial firms from
earlier runs (e.g. companies whose prices silently failed) without re-ingesting
the ones that are already fine.

Revision ID: 0003_company_ingest_state
Revises: 0002_event_study_backtest
Create Date: 2026-07-12

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_company_ingest_state"
down_revision: str | None = "0002_event_study_backtest"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "companies",
        sa.Column(
            "ingest_state",
            sa.String(length=16),
            server_default="pending",
            nullable=False,
        ),
    )
    op.add_column("companies", sa.Column("ingest_error", sa.String(length=512), nullable=True))
    op.add_column(
        "companies",
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=True,
        ),
    )
    op.create_index("ix_companies_ingest_state", "companies", ["ingest_state"])

    # Backfill: firms that already have price history are genuinely complete;
    # leave everyone else 'pending' so the batch ingester picks them up.
    op.execute(
        """
        UPDATE companies
        SET ingest_state = 'complete'
        WHERE id IN (SELECT DISTINCT company_id FROM prices)
        """
    )


def downgrade() -> None:
    op.drop_index("ix_companies_ingest_state", table_name="companies")
    op.drop_column("companies", "updated_at")
    op.drop_column("companies", "ingest_error")
    op.drop_column("companies", "ingest_state")
