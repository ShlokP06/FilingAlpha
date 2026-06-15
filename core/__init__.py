"""Shared core package: ORM models, schemas, DB session, and settings.

Imported by both the ``api`` service and the ``pipeline`` worker so the data
contract lives in exactly one place.
"""

from core.config import settings
from core.db import SessionLocal, engine, get_session

__all__ = ["settings", "engine", "SessionLocal", "get_session"]
