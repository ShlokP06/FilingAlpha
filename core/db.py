"""Database engine and session factory.

A single synchronous engine serves both the pipeline (batch writes) and the
API (reads). Synchronous SQLAlchemy keeps the pipeline simple — ``edgartools``
and ``yfinance`` are synchronous anyway — and FastAPI runs sync routes in a
threadpool without issue.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """Yield a database session, closing it afterwards.

    Usable both as a FastAPI dependency and as a context-managed generator.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
