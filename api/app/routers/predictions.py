"""Predictions router — walk-forward model run results."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from sqlalchemy import select

from api.app.deps import SessionDep
from core.models import ModelRun
from core.schemas import ModelRunOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=list[ModelRunOut])
def list_predictions(session: SessionDep) -> list[ModelRun]:
    """Return all walk-forward model runs, newest first.

    Args:
        session: Injected database session.

    Returns:
        A list of ``ModelRunOut`` objects sorted by creation date descending.
    """
    logger.debug("Fetching all model runs")

    rows = session.execute(select(ModelRun).order_by(ModelRun.created_at.desc())).scalars().all()

    return list(rows)
