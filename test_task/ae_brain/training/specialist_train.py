"""Train/calibrate side specialist pair with production metrics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ae_brain.layers.side_specialists import SideSpecialistModel
from ae_brain.training.calibration import ConfidenceCalibrator, SideCalibrators
from ae_brain.training.specialist_metrics import (
    calibration_ceiling_summary,
    confidence_ceiling_report,
    precision_at_top_frac,
    second_pass_threshold_report,
    side_balance_report,
    simulate_publishable_ev,
)


def _resolve_weight(
    requested: float | str | None,
    *,
    pos_count: int,
    neg_count: int,
    kind: str = "scale_pos_weight",
) -> float | None:
    """Resolve an ``auto``/float class-balance weight.

    ``auto`` computes the imbalance ratio (XGBoost scale_pos_weight convention):
    ``neg_count / pos_count``. A float is passed through. ``None`` disables
    weighting.
    """
    if requested is None:
        return None
    if isinstance(requested, str):
        if requested.lower() != "auto":
            raise ValueError(f"invalid weight specifier {requested!r}; expected 'auto' or a float")
        if pos_count <= 0:
            return None
        return float(neg_count) / float(pos_count)
    val = float(requested)
    return val if val > 0 else None


def _balanced_train_idx(
    y: np.ndarray,
    *,
    cut_train: int,
    max_per_class: int | None,
    seed: int = 13,
) -> np.ndarray:
    """Balanced chronological train subset.

    Selects positions from ``[0, cut_train)`` only (no future leakage). Undersamples
    the majority class down to the minority class count (capped by ``max_per_class``),
    keeping all minority-class rows. The validation/test slices are never touched.
    """
    rng = np.random.default_rng(seed)
    y_tr = np.asarray(y, dtype=int)[:cut_train]
    pos = np.flatnonzero(y_tr == 1)
    neg = np.flatnonzero(y_tr == 0)
    n_each = min(pos.size, neg.size)
    if max_per_class is not None:
        n_each = min(n_each, int(max_per_class))
    n_each = max(0, n_each)
    if n_each == 0:
        return np.arange(cut_train)
    pos_sel = pos if pos.size <= n_each else rng.choice(pos, size=n_each, replace=False)
    neg_sel = neg if neg.size <= n_each else rng.choice(neg, size=n_each, replace=False)
    return np.sort(np.concatenate([pos_sel, neg_sel]).astype(int))


def train_specialist_pair(
    F: np.ndarray,
    y_long: np.ndarray,
    y_short: np.ndarray,
    ev_long: np.ndarray,
    ev_short: np.ndarray,
    *,
    cut_train: int,
    cut_val: int,
    model_kind: str = "logreg",
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic",
    long_class_weight: float | None = None,
    short_class_weight: float | None = None,
    long_scale_pos_weight: float | None = None,
    short_scale_pos_weight: float | None = None,
    balance_train_samples: bool = False,
    max_side_train_samples_per_class: int | None = None,
    symbols: np.ndarray | None = None,
    regime_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    # Resolve auto weights from the chronological train slice (no leakage).
    y_long_tr = np.asarray(y_long, dtype=int)[:cut_train]
    y_short_tr = np.asarray(y_short, dtype=int)[:cut_train]
    long_spw = _resolve_weight(
        long_scale_pos_weight,
        pos_count=int(y_long_tr.sum()),
        neg_count=int((y_long_tr == 0).sum()),
    )
    short_spw = _resolve_weight(
        short_scale_pos_weight,
        pos_count=int(y_short_tr.sum()),
        neg_count=int((y_short_tr == 0).sum()),
    )

    # Balanced sampling is confined to [0, cut_train) and is per-side independent.
    long_train_idx = (
        _balanced_train_idx(
            y_long, cut_train=cut_train, max_per_class=max_side_train_samples_per_class
        )
        if balance_train_samples
        else None
    )
    short_train_idx = (
        _balanced_train_idx(
            y_short, cut_train=cut_train, max_per_class=max_side_train_samples_per_class
        )
        if balance_train_samples
        else None
    )

    long_model = SideSpecialistModel("LONG", model_kind=model_kind)
    short_model = SideSpecialistModel("SHORT", model_kind=model_kind)
    long_model.fit(
        F,
        y_long,
        train_end=cut_train,
        class_weight=long_class_weight,
        scale_pos_weight=long_spw,
        train_idx=long_train_idx,
    )
    short_model.fit(
        F,
        y_short,
        train_end=cut_train,
        class_weight=short_class_weight,
        scale_pos_weight=short_spw,
        train_idx=short_train_idx,
    )

    F_val = F[cut_train:cut_val]
    y_l_val = y_long[cut_train:cut_val]
    y_s_val = y_short[cut_train:cut_val]
    ev_l_val = ev_long[cut_train:cut_val]
    ev_s_val = ev_short[cut_train:cut_val]
    sym_val = symbols[cut_train:cut_val] if symbols is not None else None
    reg_val = regime_ids[cut_train:cut_val] if regime_ids is not None else None

    long_raw = np.array([long_model.predict_raw(F_val[i]) for i in range(len(F_val))], dtype=float)
    short_raw = np.array([short_model.predict_raw(F_val[i]) for i in range(len(F_val))], dtype=float)

    side_cals = SideCalibrators(calibration_method)
    cal_long = side_cals.long.fit(long_raw, y_l_val)
    cal_short = side_cals.short.fit(short_raw, y_s_val)
    long_cal = np.array([side_cals.calibrate("LONG", r) for r in long_raw], dtype=float)
    short_cal = np.array([side_cals.calibrate("SHORT", r) for r in short_raw], dtype=float)

    pub_sim = simulate_publishable_ev(long_cal, short_cal, ev_l_val, ev_s_val)

    balance_diag = side_balance_report(
        y_long_train=y_long_tr,
        y_short_train=y_short_tr,
        y_long_val=y_l_val,
        y_short_val=y_s_val,
        long_cal=long_cal,
        short_cal=short_cal,
        long_ev=ev_l_val,
        short_ev=ev_s_val,
        symbols_val=sym_val,
    )
    ceiling_summary = calibration_ceiling_summary(long_cal, short_cal)
    threshold_report = second_pass_threshold_report(
        long_cal, short_cal, ev_l_val, ev_s_val
    )

    return {
        "long_model": long_model,
        "short_model": short_model,
        "side_calibrators": side_cals,
        "long_metrics": long_model.metrics,
        "short_metrics": short_model.metrics,
        "calibration": {"LONG": cal_long.to_dict(), "SHORT": cal_short.to_dict()},
        "validation_production_metrics": {
            "precision_at_top_5pct_LONG_raw": precision_at_top_frac(y_l_val, long_raw, 0.05),
            "precision_at_top_5pct_SHORT_raw": precision_at_top_frac(y_s_val, short_raw, 0.05),
            "precision_at_top_5pct_LONG_cal": precision_at_top_frac(y_l_val, long_cal, 0.05),
            "precision_at_top_5pct_SHORT_cal": precision_at_top_frac(y_s_val, short_cal, 0.05),
            **pub_sim,
        },
        "confidence_ceiling": {
            "LONG": confidence_ceiling_report(
                side="LONG",
                y_true=y_l_val,
                raw=long_raw,
                calibrated=long_cal,
                ev_usd=ev_l_val,
                symbols=sym_val,
                regime_ids=reg_val,
            ),
            "SHORT": confidence_ceiling_report(
                side="SHORT",
                y_true=y_s_val,
                raw=short_raw,
                calibrated=short_cal,
                ev_usd=ev_s_val,
                symbols=sym_val,
                regime_ids=reg_val,
            ),
        },
        "side_balance": balance_diag,
        "calibration_ceiling_summary": ceiling_summary,
        "second_pass_threshold_report": threshold_report,
        "balancing": {
            "balance_train_samples": balance_train_samples,
            "max_side_train_samples_per_class": max_side_train_samples_per_class,
            "long_scale_pos_weight": long_spw,
            "short_scale_pos_weight": short_spw,
            "long_train_idx_size": int(long_train_idx.size) if long_train_idx is not None else None,
            "short_train_idx_size": int(short_train_idx.size) if short_train_idx is not None else None,
            "train_split_chronological": True,
            "validation_unchanged": True,
            "no_future_leakage": True,
        },
    }
