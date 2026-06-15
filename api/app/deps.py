"""FastAPI dependency aliases for shared resources.

Routers import ``SessionDep`` instead of wiring ``Depends(get_session)``
on every function parameter — keeps signatures concise and testable.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from core.db import get_session

SessionDep = Annotated[Session, Depends(get_session)]

__all__ = ["SessionDep"]
