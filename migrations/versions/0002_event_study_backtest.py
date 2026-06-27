"""event-study backtest metrics

Replace the portfolio-oriented backtest columns (``ls_sharpe``, ``hit_rate``,
``cum_return``) with the event-study spread metrics (``ls_spread``,
``spread_tstat``). Annual 10-K filings are too sparse for a per-rebalance-date
long-short portfolio, so the headline statistic becomes the top-minus-bottom
tercile forward-return spread with a Welch t-stat.

Revision ID: 0002_event_study_backtest
Revises: 0001_initial
Create Date: 2026-06-16

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_event_study_backtest"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("backtest_runs", sa.Column("ls_spread", sa.Float(), nullable=True))
    op.add_column("backtest_runs", sa.Column("spread_tstat", sa.Float(), nullable=True))
    op.drop_column("backtest_runs", "ls_sharpe")
    op.drop_column("backtest_runs", "hit_rate")
    op.drop_column("backtest_runs", "cum_return")


def downgrade() -> None:
    op.add_column("backtest_runs", sa.Column("cum_return", sa.Float(), nullable=True))
    op.add_column("backtest_runs", sa.Column("hit_rate", sa.Float(), nullable=True))
    op.add_column("backtest_runs", sa.Column("ls_sharpe", sa.Float(), nullable=True))
    op.drop_column("backtest_runs", "spread_tstat")
    op.drop_column("backtest_runs", "ls_spread")
