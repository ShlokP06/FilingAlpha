"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-15

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("cik", sa.String(16), nullable=False),
        sa.Column("name", sa.String(256)),
        sa.Column("sector", sa.String(128)),
    )
    op.create_index("ix_companies_ticker", "companies", ["ticker"], unique=True)
    op.create_index("ix_companies_cik", "companies", ["cik"])

    op.create_table(
        "filings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("company_id", sa.Integer, sa.ForeignKey("companies.id"), nullable=False),
        sa.Column("form_type", sa.String(16), nullable=False),
        sa.Column("filing_date", sa.Date, nullable=False),
        sa.Column("fiscal_period", sa.String(16)),
        sa.Column("period_end", sa.Date),
        sa.Column("text_path", sa.String(512)),
        sa.Column("item1a_text", sa.Text),
        sa.Column("mdna_text", sa.Text),
        sa.UniqueConstraint("company_id", "form_type", "filing_date", name="uq_filing"),
    )
    op.create_index("ix_filings_company_id", "filings", ["company_id"])
    op.create_index("ix_filings_filing_date", "filings", ["filing_date"])

    op.create_table(
        "prices",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("company_id", sa.Integer, sa.ForeignKey("companies.id"), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("adj_close", sa.Float, nullable=False),
        sa.UniqueConstraint("company_id", "date", name="uq_price"),
    )
    op.create_index("ix_prices_company_id", "prices", ["company_id"])
    op.create_index("ix_prices_date", "prices", ["date"])

    op.create_table(
        "signals",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("filing_id", sa.Integer, sa.ForeignKey("filings.id"), nullable=False),
        sa.Column("company_id", sa.Integer, sa.ForeignKey("companies.id"), nullable=False),
        sa.Column("filing_date", sa.Date, nullable=False),
        sa.Column("lm_negative", sa.Float),
        sa.Column("lm_uncertainty", sa.Float),
        sa.Column("lm_litigious", sa.Float),
        sa.Column("yoy_similarity", sa.Float),
        sa.Column("risk_factor_delta", sa.Float),
        sa.Column("fog_readability", sa.Float),
    )
    op.create_index("ix_signals_filing_id", "signals", ["filing_id"], unique=True)
    op.create_index("ix_signals_company_id", "signals", ["company_id"])
    op.create_index("ix_signals_filing_date", "signals", ["filing_date"])

    op.create_table(
        "forward_returns",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("filing_id", sa.Integer, sa.ForeignKey("filings.id"), nullable=False),
        sa.Column("horizon_days", sa.Integer, nullable=False),
        sa.Column("fwd_return", sa.Float, nullable=False),
        sa.UniqueConstraint("filing_id", "horizon_days", name="uq_fwd_return"),
    )
    op.create_index("ix_forward_returns_filing_id", "forward_returns", ["filing_id"])

    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("signal", sa.String(64), nullable=False),
        sa.Column("horizon_days", sa.Integer, nullable=False),
        sa.Column("config_json", sa.Text),
        sa.Column("ic", sa.Float),
        sa.Column("ic_tstat", sa.Float),
        sa.Column("ls_sharpe", sa.Float),
        sa.Column("hit_rate", sa.Float),
        sa.Column("cum_return", sa.Float),
    )

    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("model_type", sa.String(64), nullable=False),
        sa.Column("features_json", sa.Text),
        sa.Column("metrics_json", sa.Text),
    )


def downgrade() -> None:
    op.drop_table("model_runs")
    op.drop_table("backtest_runs")
    op.drop_table("forward_returns")
    op.drop_table("signals")
    op.drop_table("prices")
    op.drop_table("filings")
    op.drop_table("companies")
