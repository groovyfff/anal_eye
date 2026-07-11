"""Walk-forward validation for side specialists (chronological, no leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ae_brain.training.metrics import brier
from ae_brain.training.splits import walk_forward_folds
from ae_brain.training.specialist_metrics import (
    precision_at_top_frac,
    simulate_publishable_ev,
)
from ae_brain.training.specialist_train import train_specialist_pair


def run_walk_forward_specialists(
    F: np.ndarray,
    y_long: np.ndarray,
    y_short: np.ndarray,
    ev_long: np.ndarray,
    ev_short: np.ndarray,
    timestamps: np.ndarray,
    *,
    n_folds: int = 4,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    label_horizon: int = 72,
    model_kind: str = "lightgbm",
    calibration_method: str = "sigmoid",
    symbols: np.ndarray | None = None,
    regime_ids: np.ndarray | None = None,
    publish_threshold: float = 0.70,
) -> dict[str, Any]:
    """Expanding-window folds; calibration fit only on each fold's validation slice."""
    ts = pd.Series(pd.to_datetime(timestamps, utc=True))
    folds, holdout_idx = walk_forward_folds(
        ts,
        n_folds=n_folds,
        val_frac=val_frac,
        test_frac=test_frac,
        label_horizon=label_horizon,
    )
    fold_reports: list[dict[str, Any]] = []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        if train_idx.size < 200 or val_idx.size < 50:
            continue
        cut_train = int(train_idx.size)
        # Reindex arrays into contiguous train|val for train_specialist_pair API.
        order = np.concatenate([train_idx, val_idx])
        rep = train_specialist_pair(
            F[order],
            y_long[order],
            y_short[order],
            ev_long[order],
            ev_short[order],
            cut_train=cut_train,
            cut_val=len(order),
            model_kind=model_kind,
            calibration_method=calibration_method,
            symbols=symbols[order] if symbols is not None else None,
            regime_ids=regime_ids[order] if regime_ids is not None else None,
        )
        prod = rep["validation_production_metrics"]
        cal = rep["calibration"]
        ce = rep["confidence_ceiling"]
        fold_reports.append(
            {
                "fold": fold_i,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "val_start": str(ts.iloc[val_idx[0]]),
                "val_end": str(ts.iloc[val_idx[-1]]),
                f"publishable_LONG_ge_{publish_threshold:.2f}": int(
                    prod.get(f"publishable_LONG_ge_{publish_threshold:.2f}", 0)
                ),
                f"publishable_SHORT_ge_{publish_threshold:.2f}": int(
                    prod.get(f"publishable_SHORT_ge_{publish_threshold:.2f}", 0)
                ),
                "publishable_EV_LONG": float(prod.get("publishable_EV_LONG", 0.0)),
                "publishable_EV_SHORT": float(prod.get("publishable_EV_SHORT", 0.0)),
                "brier_LONG_raw": float(cal["LONG"].get("brier_raw", 0.0)),
                "brier_LONG_calibrated": float(cal["LONG"].get("brier_calibrated", 0.0)),
                "brier_SHORT_raw": float(cal["SHORT"].get("brier_raw", 0.0)),
                "brier_SHORT_calibrated": float(cal["SHORT"].get("brier_calibrated", 0.0)),
                "top_k_precision_LONG": ce["LONG"].get("calibrated_precision_at_top_k", {}),
                "top_k_precision_SHORT": ce["SHORT"].get("calibrated_precision_at_top_k", {}),
                "long_auc": rep["long_metrics"].get("val_auc"),
                "short_auc": rep["short_metrics"].get("val_auc"),
            }
        )

    stable = _assess_fold_stability(fold_reports, publish_threshold=publish_threshold)
    return {
        "n_folds": len(fold_reports),
        "folds": fold_reports,
        "holdout_n": int(holdout_idx.size),
        "stability": stable,
        "no_test_leakage": True,
    }


def _assess_fold_stability(folds: list[dict[str, Any]], *, publish_threshold: float) -> dict[str, Any]:
    if not folds:
        return {"stable": False, "reason": "no_folds"}
    l_key = f"publishable_LONG_ge_{publish_threshold:.2f}"
    s_key = f"publishable_SHORT_ge_{publish_threshold:.2f}"
    long_counts = [int(f.get(l_key, 0)) for f in folds]
    short_counts = [int(f.get(s_key, 0)) for f in folds]
    ev_long = [float(f.get("publishable_EV_LONG", 0.0)) for f in folds]
    ev_short = [float(f.get("publishable_EV_SHORT", 0.0)) for f in folds]
    long_positive = all(c > 0 for c in long_counts)
    short_positive = all(c > 0 for c in short_counts)
    ev_long_positive = all(e > 0 for e in ev_long)
    ev_short_positive = all(e > 0 for e in ev_short)
    return {
        "stable": bool(long_positive and short_positive and ev_long_positive and ev_short_positive),
        "long_publishable_all_folds": long_positive,
        "short_publishable_all_folds": short_positive,
        "ev_long_positive_all_folds": ev_long_positive,
        "ev_short_positive_all_folds": ev_short_positive,
        "long_publishable_counts": long_counts,
        "short_publishable_counts": short_counts,
    }
