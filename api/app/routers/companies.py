"""Companies router — read all tracked companies."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from sqlalchemy import select

from api.app.deps import SessionDep
from core.models import Company
from core.schemas import CompanyOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=list[CompanyOut])
def list_companies(session: SessionDep) -> list[Company]:
    """Return every company tracked in the FilingAlpha universe.

    Returns:
        A list of company records with ticker, CIK, name, and sector.
    """
    logger.debug("Fetching all companies")
    rows = session.execute(select(Company)).scalars().all()
    return list(rows)
