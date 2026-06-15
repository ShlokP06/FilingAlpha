"""Expanding-window walk-forward model on filing signals.

Trains an sklearn classifier to predict the **sign** of the filing-lagged
forward return from the classical-NLP signals, evaluated strictly
out-of-sample. Observations are sorted by ``filing_date`` and split into
sequential folds; for each fold the model is trained only on data with a
filing date strictly **before** the fold (an expanding window), then used to
predict the fold. This prevents any future information from entering training
and is the model-side analogue of the filing-lag in :mod:`pipeline.returns`.

Out-of-sample accuracy and ROC-AUC are reported alongside feature importances,
and a :class:`ModelRun` is persisted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models import ForwardReturn, ModelRun, Signal

logger = logging.getLogger(__name__)

DEFAULT_FEATURES: list[str] = [
    "lm_negative",
    "lm_uncertainty",
    "lm_litigious",
    "yoy_similarity",
    "risk_factor_delta",
    "fog_readability",
]


@dataclass
class WalkForwardResult:
    """Out-of-sample walk-forward metrics."""

    model_type: str
    features: list[str]
    horizon_days: int
    n_folds: int
    n_oos: int
    oos_accuracy: float
    oos_auc: float
    feature_importances: dict[str, float] = field(default_factory=dict)
    fold_boundaries: list[dict[str, str]] = field(default_factory=list)


def load_feature_matrix(session: Session, features: list[str], horizon: int) -> pd.DataFrame:
    """Load a date-sorted feature matrix with the binary forward-return label.

    Args:
        session: Active database session.
        features: Signal column names to use as features.
        horizon: Forward horizon in trading days.

    Returns:
        DataFrame sorted by ``filing_date`` with the feature columns plus
        ``fwd_return`` and a binary ``label`` (1 if ``fwd_return > 0`` else 0),
        rows with any null feature or null return dropped.
    """
    cols = [getattr(Signal, f) for f in features]
    rows = session.execute(
        select(Signal.filing_date, *cols, ForwardReturn.fwd_return)
        .join(ForwardReturn, ForwardReturn.filing_id == Signal.filing_id)
        .where(ForwardReturn.horizon_days == horizon)
        .order_by(Signal.filing_date.asc())
    ).all()

    frame = pd.DataFrame(rows, columns=["filing_date", *features, "fwd_return"])
    frame = frame.dropna(subset=[*features, "fwd_return"]).reset_index(drop=True)
    frame = frame.sort_values("filing_date", kind="stable").reset_index(drop=True)
    frame["label"] = (frame["fwd_return"] > 0).astype(int)
    return frame


def _fold_indices(n: int, n_folds: int) -> list[tuple[int, int]]:
    """Compute ``(start, stop)`` test-fold index ranges over ``n`` rows.

    The first ``min_train`` rows seed the initial training window and are never
    used as test data; the remainder is partitioned into ``n_folds`` contiguous
    test blocks.

    Args:
        n: Total number of observations.
        n_folds: Number of test folds.

    Returns:
        List of ``(start, stop)`` index pairs (half-open) for each test fold.
    """
    min_train = max(n // (n_folds + 1), 1)
    remaining = n - min_train
    if remaining < n_folds:
        # Not enough rows for the requested folds; degrade to one fold.
        return [(min_train, n)] if remaining > 0 else []
    block = remaining // n_folds
    bounds: list[tuple[int, int]] = []
    start = min_train
    for i in range(n_folds):
        stop = n if i == n_folds - 1 else start + block
        bounds.append((start, stop))
        start = stop
    return bounds


def _snap_index(idx: int, dates: list) -> int:
    """Advance ``idx`` to the next date-change boundary.

    Many firms file on the same date, so a row-count fold boundary can fall in
    the middle of a same-date group — which would split one ``filing_date``
    across train and test and violate the strict temporal invariant. Snapping
    forward to where the date changes keeps each date wholly on one side.
    """
    n = len(dates)
    if idx <= 0 or idx >= n:
        return idx
    while idx < n and dates[idx] == dates[idx - 1]:
        idx += 1
    return idx


def run_walkforward(
    session: Session,
    features: list[str] | None = None,
    horizon: int = 21,
    n_folds: int = 5,
    persist: bool = True,
    random_state: int = 42,
) -> WalkForwardResult:
    """Run an expanding-window walk-forward classification and persist results.

    For each fold, a :class:`~sklearn.ensemble.GradientBoostingClassifier` is
    fit only on observations whose ``filing_date`` precedes the fold (expanding
    window), then predicts the fold. Predictions are accumulated across folds
    and scored out-of-sample. The strict temporal split guarantees
    ``max(train filing_date) < min(test filing_date)`` for every fold.

    Args:
        session: Active database session.
        features: Feature columns; defaults to all six signals.
        horizon: Forward horizon in trading days.
        n_folds: Number of expanding-window folds.
        persist: If ``True``, write a :class:`ModelRun` row.
        random_state: Seed for reproducibility.

    Returns:
        The :class:`WalkForwardResult` with OOS accuracy/AUC and averaged
        feature importances.
    """
    feats = list(features) if features is not None else list(DEFAULT_FEATURES)
    frame = load_feature_matrix(session, feats, horizon)
    n = len(frame)

    bounds = _fold_indices(n, n_folds) if n >= 2 else []
    oos_pred: list[int] = []
    oos_proba: list[float] = []
    oos_true: list[int] = []
    importances_acc = np.zeros(len(feats), dtype=float)
    folds_used = 0
    fold_boundaries: list[dict[str, str]] = []

    x_all = frame[feats].to_numpy(dtype=float)
    y_all = frame["label"].to_numpy(dtype=int)
    dates = frame["filing_date"].tolist()

    # Snap fold boundaries to date changes so a single filing_date is never
    # split across the train/test cut (preserves the no-lookahead invariant
    # when many firms share a filing date).
    snapped: list[tuple[int, int]] = []
    for start, stop in bounds:
        s, e = _snap_index(start, dates), _snap_index(stop, dates)
        if s > 0 and s < e:
            snapped.append((s, e))
    bounds = snapped

    for start, stop in bounds:
        x_train, y_train = x_all[:start], y_all[:start]
        x_test, y_test = x_all[start:stop], y_all[start:stop]
        if len(x_test) == 0 or len(x_train) == 0:
            continue
        # Need both classes in training for a meaningful classifier.
        if len(np.unique(y_train)) < 2:
            continue

        # Enforce the temporal invariant explicitly.
        assert max(dates[:start]) < min(
            dates[start:stop]
        ), "Walk-forward lookahead: train dates must precede test dates."

        model = GradientBoostingClassifier(random_state=random_state)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        oos_pred.extend(int(p) for p in preds)
        oos_true.extend(int(t) for t in y_test)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_test)
            # Probability of the positive class (label 1).
            pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
            oos_proba.extend(float(p) for p in proba[:, pos_idx])

        importances_acc += model.feature_importances_
        folds_used += 1
        fold_boundaries.append(
            {
                "train_max_date": str(max(dates[:start])),
                "test_min_date": str(min(dates[start:stop])),
                "test_max_date": str(max(dates[start:stop])),
            }
        )

    n_oos = len(oos_true)
    if n_oos > 0:
        accuracy = float(accuracy_score(oos_true, oos_pred))
    else:
        accuracy = 0.0

    if n_oos > 0 and len(set(oos_true)) > 1 and len(oos_proba) == n_oos:
        auc = float(roc_auc_score(oos_true, oos_proba))
    else:
        auc = float("nan")

    if folds_used > 0:
        mean_importances = importances_acc / folds_used
        feature_importances = {f: float(v) for f, v in zip(feats, mean_importances)}
    else:
        feature_importances = {f: 0.0 for f in feats}

    result = WalkForwardResult(
        model_type="GradientBoostingClassifier",
        features=feats,
        horizon_days=horizon,
        n_folds=folds_used,
        n_oos=n_oos,
        oos_accuracy=accuracy,
        oos_auc=auc,
        feature_importances=feature_importances,
        fold_boundaries=fold_boundaries,
    )

    if persist:
        session.add(
            ModelRun(
                model_type=result.model_type,
                features_json=json.dumps(feats),
                metrics_json=json.dumps(
                    {
                        "horizon_days": horizon,
                        "n_folds": folds_used,
                        "n_oos": n_oos,
                        "oos_accuracy": accuracy,
                        "oos_auc": None if np.isnan(auc) else auc,
                        "feature_importances": feature_importances,
                    }
                ),
            )
        )
        session.commit()

    logger.info(
        "Walk-forward %s h=%d: folds=%d n_oos=%d acc=%.3f auc=%s",
        result.model_type,
        horizon,
        folds_used,
        n_oos,
        accuracy,
        "nan" if np.isnan(auc) else f"{auc:.3f}",
    )
    return result


__all__ = ["run_walkforward", "load_feature_matrix", "WalkForwardResult", "DEFAULT_FEATURES"]
