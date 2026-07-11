"""Comprehensive model evaluation, reporting, and promotion health checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ae_brain.layers.meta import CLASS_LONG, CLASS_SHORT, CLASS_SKIP
from ae_brain.training.labels import LABEL_LONG, LABEL_SHORT, LABEL_SKIP, label_distribution_report
from ae_brain.training.metrics import MetricsBundle, brier, summarize_trades
from ae_brain.training.splits import make_time_split

_CLASS_TO_NAME = {CLASS_SHORT: "SHORT", CLASS_SKIP: "SKIP", CLASS_LONG: "LONG"}
_LABEL_TO_NAME = {LABEL_SHORT: "SHORT", LABEL_SKIP: "SKIP", LABEL_LONG: "LONG"}

PUBLISHABLE_GE_70_FIELD_NAMES: tuple[str, ...] = (
    "publishable_long_count_ge_70",
    "publishable_short_count_ge_70",
    "publishable_long_ev_ge_70",
    "publishable_short_ev_ge_70",
    "publishable_total_ev_ge_70",
    "publishable_total_trade_count_ge_70",
)


@dataclass
class SignalBatch:
    decisions: np.ndarray
    expected_evs: np.ndarray
    confidence: np.ndarray
    symbols: np.ndarray
    timestamps: np.ndarray
    publishable: np.ndarray
    meta_p_short: np.ndarray | None = None
    meta_p_long: np.ndarray | None = None
    meta_p_skip: np.ndarray | None = None
    fused_scores: np.ndarray | None = None
    tabular_p_up: np.ndarray | None = None
    raw_long_confidence: np.ndarray | None = None
    raw_short_confidence: np.ndarray | None = None


def _decision_name(value: str | int) -> str:
    if isinstance(value, str):
        return value
    return _CLASS_TO_NAME.get(int(value), str(value))


def _counts_by_key(
    decisions: np.ndarray,
    keys: np.ndarray,
    *,
    publishable: np.ndarray | None = None,
    min_confidence: float | None = None,
    confidence: np.ndarray | None = None,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key in sorted(set(keys)):
        mask = keys == key
        if publishable is not None:
            mask = mask & publishable
        elif min_confidence is not None and confidence is not None:
            actionable = np.isin(decisions, ["LONG", "SHORT"])
            mask = mask & actionable & (confidence >= min_confidence)
        bucket = out.setdefault(str(key), {"LONG": 0, "SHORT": 0, "SKIP": 0})
        for d in ("LONG", "SHORT", "SKIP"):
            bucket[d] = int((decisions[mask] == d).sum())
    return out


def _distribution_stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def publishable_ge_70_metrics(
    batch: SignalBatch,
    *,
    publish_confidence: float = 0.70,
) -> dict[str, float | int]:
    """Side-level publishable metrics for trades passing confidence >= threshold."""
    long_pub = (batch.decisions == "LONG") & batch.publishable
    short_pub = (batch.decisions == "SHORT") & batch.publishable
    long_count = int(long_pub.sum())
    short_count = int(short_pub.sum())
    long_ev = float(batch.expected_evs[long_pub].sum()) if long_count else 0.0
    short_ev = float(batch.expected_evs[short_pub].sum()) if short_count else 0.0
    return {
        "publishable_long_count_ge_70": long_count,
        "publishable_short_count_ge_70": short_count,
        "publishable_long_ev_ge_70": long_ev,
        "publishable_short_ev_ge_70": short_ev,
        "publishable_total_ev_ge_70": long_ev + short_ev,
        "publishable_total_trade_count_ge_70": long_count + short_count,
        "publish_confidence_threshold": publish_confidence,
    }


def _metrics_for_mask(batch: SignalBatch, mask: np.ndarray) -> MetricsBundle:
    pnls = np.where(np.isin(batch.decisions[mask], ["LONG", "SHORT"]), batch.expected_evs[mask], 0.0)
    return summarize_trades(
        batch.decisions[mask],
        pnls,
        batch.expected_evs[mask],
        symbols=batch.symbols[mask],
        confidence=batch.confidence[mask],
    )


def build_test_metrics_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Canonical test_metrics.json body aligned with promote_model.py."""
    internal = report.get("backtest_internal_all_signals") or {}
    publish_key = next(
        (k for k in report if k.startswith("backtest_publishable_confidence_ge_")),
        "backtest_publishable_confidence_ge_0.70",
    )
    publishable_bt = report.get(publish_key) or {}
    pub_signals = report.get("publishable_signals_confidence_ge_0.70") or report.get(
        "telegram_publishable_signals_confidence_ge_70", {}
    )
    side_metrics = {k: report[k] for k in PUBLISHABLE_GE_70_FIELD_NAMES if k in report}
    return {
        **internal,
        "internal_model_signals": report.get("internal_model_signals", {}),
        "publishable_signals_ge_70": pub_signals,
        "backtest_publishable_ge_70": publishable_bt,
        **side_metrics,
        "promotable": report.get("promotable", False),
        "promotion_blockers": report.get("promotion_blockers", []),
    }


def build_evaluation_report(
    batch: SignalBatch,
    *,
    publish_confidence: float = 0.70,
    label_report: dict[str, Any] | None = None,
    meta_eval: dict[str, Any] | None = None,
    training_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a full evaluation report separating internal vs publishable behavior."""
    actionable = np.isin(batch.decisions, ["LONG", "SHORT"])
    publish_mask = batch.publishable & actionable

    internal = _metrics_for_mask(batch, np.ones(len(batch.decisions), dtype=bool))
    publishable = _metrics_for_mask(batch, publish_mask) if publish_mask.any() else MetricsBundle()

    conf_actionable = batch.confidence[actionable] if actionable.any() else np.array([])
    ev_actionable = batch.expected_evs[actionable] if actionable.any() else np.array([])

    confidence_by_decision = {
        d: _distribution_stats(batch.confidence[batch.decisions == d])
        for d in ("LONG", "SHORT", "SKIP")
    }
    ev_by_decision = {
        d: _distribution_stats(batch.expected_evs[batch.decisions == d])
        for d in ("LONG", "SHORT", "SKIP")
    }

    ts = pd.to_datetime(batch.timestamps, utc=True, errors="coerce")
    if isinstance(ts, pd.Series):
        months = ts.dt.to_period("M").astype(str).to_numpy()
    else:
        months = pd.DatetimeIndex(ts).to_period("M").astype(str)

    report = {
        "evaluation_scope": {
            "n_inferences": int(len(batch.decisions)),
            "publish_confidence_threshold": publish_confidence,
            "pnl_basis_internal": "sum(expected_value_usd) for all internal LONG/SHORT decisions",
            "pnl_basis_publishable": (
                f"sum(expected_value_usd) for LONG/SHORT passing publish gate (>={publish_confidence})"
            ),
        },
        "internal_model_signals": {
            "LONG": int((batch.decisions == "LONG").sum()),
            "SHORT": int((batch.decisions == "SHORT").sum()),
            "SKIP": int((batch.decisions == "SKIP").sum()),
        },
        f"publishable_signals_confidence_ge_{publish_confidence:.2f}": {
            "LONG": int(((batch.decisions == "LONG") & batch.publishable).sum()),
            "SHORT": int(((batch.decisions == "SHORT") & batch.publishable).sum()),
        },
        "suppressed_low_confidence_signals": int(actionable.sum() - publish_mask.sum()),
        "backtest_internal_all_signals": internal.to_dict(),
        f"backtest_publishable_confidence_ge_{publish_confidence:.2f}": publishable.to_dict(),
        "decisions_by_symbol": _counts_by_key(batch.decisions, batch.symbols),
        "publishable_decisions_by_symbol": _counts_by_key(
            batch.decisions,
            batch.symbols,
            publishable=batch.publishable,
        ),
        "decisions_by_month": _counts_by_key(batch.decisions, months.to_numpy()),
        "publishable_decisions_by_month": _counts_by_key(
            batch.decisions,
            months.to_numpy(),
            publishable=batch.publishable,
        ),
        "confidence_distribution_by_decision": confidence_by_decision,
        "ev_distribution_by_decision": ev_by_decision,
        "confidence_summary_actionable": _distribution_stats(conf_actionable),
        "ev_summary_actionable": _distribution_stats(ev_actionable),
        "calibration": {
            "brier_ev_positive": brier(
                (ev_actionable > 0).astype(int),
                np.clip(conf_actionable, 0.0, 1.0),
            )
            if conf_actionable.size >= 5
            else None,
            "max_actionable_confidence": float(conf_actionable.max()) if conf_actionable.size else None,
            "share_actionable_ge_70": float((conf_actionable >= publish_confidence).mean())
            if conf_actionable.size
            else 0.0,
        },
    }
    if label_report is not None:
        report["label_distribution"] = label_report
    if meta_eval is not None:
        report["meta_model_evaluation"] = meta_eval
    if training_metrics is not None:
        report["training_metrics"] = training_metrics

    report["side_diagnostics"] = build_side_diagnostics(batch, publish_confidence=publish_confidence)
    report.update(publishable_ge_70_metrics(batch, publish_confidence=publish_confidence))

    from ae_brain.training.promotion import evaluate_promotion

    test_metrics = build_test_metrics_payload(report)
    summary_stub = {
        "publishable_signals_ge_70": report.get(f"publishable_signals_confidence_ge_{publish_confidence:.2f}", {}),
        "publishable_backtest_ge_70": report.get(f"backtest_publishable_confidence_ge_{publish_confidence:.2f}", {}),
    }
    promo = evaluate_promotion(test_metrics, summary=summary_stub)
    report["promotable"] = promo.passed
    report["promotion_blockers"] = promo.reasons
    report["warnings"] = report.get("warnings", [])
    return report


def build_side_diagnostics(batch: SignalBatch, *, publish_confidence: float = 0.70) -> dict[str, Any]:
    """Per-side diagnostics: confidence, EV, publishable counts (LONG/SHORT)."""
    ts = pd.to_datetime(batch.timestamps, utc=True, errors="coerce")
    if isinstance(ts, pd.Series):
        months = ts.dt.to_period("M").astype(str).to_numpy()
    else:
        months = pd.DatetimeIndex(ts).to_period("M").astype(str)

    out: dict[str, Any] = {}
    for side in ("LONG", "SHORT"):
        mask = batch.decisions == side
        pub_mask = mask & batch.publishable
        evs = batch.expected_evs[mask]
        confs = batch.confidence[mask]
        pub_evs = batch.expected_evs[pub_mask]
        profitable_rate = float((evs > 0).mean()) if evs.size else 0.0
        raw_confs = batch.raw_long_confidence if side == "LONG" else batch.raw_short_confidence
        raw_side = raw_confs[mask] if raw_confs is not None else np.array([])
        fused_side = batch.fused_scores[mask] if batch.fused_scores is not None else np.array([])
        tab_up = batch.tabular_p_up[mask] if batch.tabular_p_up is not None else np.array([])
        out[side] = {
            "internal_count": int(mask.sum()),
            "raw_directional_score": _distribution_stats(fused_side),
            "tabular_p_up": _distribution_stats(tab_up),
            "raw_confidence": _distribution_stats(raw_side),
            "calibrated_confidence": _distribution_stats(confs),
            "ev_distribution": _distribution_stats(evs),
            "profitable_or_ev_positive_rate": profitable_rate,
            f"publishable_count_ge_{publish_confidence:.2f}": int(pub_mask.sum()),
            "publishable_ev": float(pub_evs.sum()) if pub_evs.size else 0.0,
            "per_symbol": _counts_by_key(
                np.where(mask, side, "SKIP"),
                batch.symbols,
                min_confidence=publish_confidence,
                confidence=batch.confidence,
            ),
            "per_month": _counts_by_key(
                np.where(mask, side, "SKIP"),
                months,
                min_confidence=publish_confidence,
                confidence=batch.confidence,
            ),
        }
    return out


def build_mode_investigation(batch: SignalBatch, *, publish_confidence: float = 0.70) -> dict[str, Any]:
    """Root-cause diagnostics for side bias: tabular p_up, meta trade probs, EV symmetry."""
    actionable = np.isin(batch.decisions, ["LONG", "SHORT"])
    inv: dict[str, Any] = {
        "tabular_p_up_all": _distribution_stats(batch.tabular_p_up) if batch.tabular_p_up is not None else {"n": 0},
        "fused_score_all": _distribution_stats(batch.fused_scores) if batch.fused_scores is not None else {"n": 0},
    }
    if batch.tabular_p_up is not None:
        inv["tabular_p_up_by_decision"] = {
            d: _distribution_stats(batch.tabular_p_up[batch.decisions == d]) for d in ("LONG", "SHORT", "SKIP")
        }
        inv["fused_negative_rate"] = float((batch.fused_scores < 0).mean()) if batch.fused_scores is not None else None
        inv["fused_positive_rate"] = float((batch.fused_scores > 0).mean()) if batch.fused_scores is not None else None
    if batch.meta_p_short is not None:
        inv["meta_p_trade"] = _distribution_stats(batch.meta_p_skip)
        inv["meta_p_short_given_trade"] = _distribution_stats(batch.meta_p_short[actionable]) if actionable.any() else {"n": 0}
        inv["meta_p_long_given_trade"] = _distribution_stats(batch.meta_p_long[actionable]) if actionable.any() else {"n": 0}
        inv["short_meta_ge_70_rate"] = float((batch.meta_p_short >= publish_confidence).mean())
        inv["long_meta_ge_70_rate"] = float((batch.meta_p_long >= publish_confidence).mean())
    if batch.raw_long_confidence is not None and batch.raw_short_confidence is not None:
        inv["raw_long_confidence_all"] = _distribution_stats(batch.raw_long_confidence)
        inv["raw_short_confidence_all"] = _distribution_stats(batch.raw_short_confidence)
        inv["raw_long_ge_70"] = int((batch.raw_long_confidence >= publish_confidence).sum())
        inv["raw_short_ge_70"] = int((batch.raw_short_confidence >= publish_confidence).sum())
    for side in ("LONG", "SHORT"):
        mask = batch.decisions == side
        inv[f"{side}_ev_positive_rate"] = float((batch.expected_evs[mask] > 0).mean()) if mask.any() else 0.0
        inv[f"{side}_publishable_count"] = int((mask & batch.publishable).sum())
    return inv


@dataclass
class RunHealth:
    promotable: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def assess_run_health(
    report: dict[str, Any],
    *,
    publish_confidence: float = 0.70,
    require_test_metrics: bool = True,
) -> RunHealth:
    """Assess promotion health using the same rules as promote_model.py."""
    from ae_brain.training.promotion import evaluate_promotion

    if require_test_metrics and not report.get("backtest_internal_all_signals"):
        return RunHealth(promotable=False, blockers=["missing_or_empty_test_metrics"], warnings=[])

    test_metrics = build_test_metrics_payload(report)
    summary_stub = {
        "publishable_signals_ge_70": report.get(f"publishable_signals_confidence_ge_{publish_confidence:.2f}")
        or report.get("publishable_signals_ge_70", {}),
        "publishable_backtest_ge_70": report.get(f"backtest_publishable_confidence_ge_{publish_confidence:.2f}")
        or report.get("publishable_backtest_ge_70", {}),
    }
    promo = evaluate_promotion(test_metrics, summary=summary_stub)
    warnings = list(report.get("warnings") or [])
    internal_metrics = report.get("backtest_internal_all_signals") or {}
    publish_key = f"publishable_signals_confidence_ge_{publish_confidence:.2f}"
    publishable = report.get(publish_key) or report.get("publishable_signals_ge_70") or {}
    if float(internal_metrics.get("net_pnl_usd", 0.0)) > 0 and int(publishable.get("LONG", 0)) + int(
        publishable.get("SHORT", 0)
    ) == 0:
        if "positive_internal_pnl_but_zero_publishable_signals" not in warnings:
            warnings.append("positive_internal_pnl_but_zero_publishable_signals")
    if internal_metrics.get("precision_at_conf_70") is None and int(publishable.get("LONG", 0)) + int(
        publishable.get("SHORT", 0)
    ) == 0:
        if "precision_at_conf_70_unavailable_no_high_confidence_trades" not in warnings:
            warnings.append("precision_at_conf_70_unavailable_no_high_confidence_trades")
    return RunHealth(promotable=promo.passed, blockers=promo.reasons, warnings=warnings)


def label_distribution_for_dataset(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> dict[str, Any]:
    labels = df[label_col].to_numpy()
    return label_distribution_report(
        labels,
        df[timestamp_col],
        df[symbol_col].to_numpy(),
    )


def split_label_counts(
    df: pd.DataFrame,
    *,
    label_col: str,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> dict[str, Any]:
    split = make_time_split(df[timestamp_col])
    out: dict[str, Any] = {}
    for name, idx in (("train", split.train_idx), ("val", split.val_idx), ("test", split.test_idx)):
        sub = df.iloc[idx]
        counts = label_distribution_for_dataset(sub, label_col=label_col, timestamp_col=timestamp_col, symbol_col=symbol_col)
        out[name] = counts["label_distribution_overall"]
    return out


def evaluate_meta_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    *,
    classes: list[int] | None = None,
) -> dict[str, Any]:
    classes = classes or [CLASS_SHORT, CLASS_SKIP, CLASS_LONG]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    pred_counts = {_CLASS_TO_NAME[c]: int((y_pred == c).sum()) for c in classes}
    true_counts = {_CLASS_TO_NAME[c]: int((y_true == c).sum()) for c in classes}
    directional_mask = np.isin(y_pred, [CLASS_SHORT, CLASS_LONG])
    if proba.ndim == 2 and proba.shape[1] >= 3:
        p_dir = np.max(proba[:, [0, 2]], axis=1)
    else:
        p_dir = proba
    return {
        "confusion_matrix": {
            "labels": [_CLASS_TO_NAME[c] for c in classes],
            "matrix": cm.tolist(),
        },
        "true_class_distribution": true_counts,
        "predicted_class_distribution": pred_counts,
        "directional_prediction_share": {
            "LONG": int((y_pred == CLASS_LONG).sum()),
            "SHORT": int((y_pred == CLASS_SHORT).sum()),
            "SKIP": int((y_pred == CLASS_SKIP).sum()),
        },
        "directional_confidence_stats": _distribution_stats(p_dir[directional_mask])
        if directional_mask.any()
        else {"n": 0},
    }


def build_summary_json(
    run_id: str,
    report: dict[str, Any],
    *,
    artifacts_path: str,
    training_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    internal = report.get("backtest_internal_all_signals") or {}
    publish_key = next(
        (k for k in report if k.startswith("backtest_publishable_confidence_ge_")),
        "backtest_publishable_confidence_ge_0.70",
    )
    publishable = report.get(publish_key) or {}
    side_fields = {k: report[k] for k in PUBLISHABLE_GE_70_FIELD_NAMES if k in report}
    return {
        "run_id": run_id,
        "artifacts_path": artifacts_path,
        "promotable": report.get("promotable", False),
        "promotion_blockers": report.get("promotion_blockers", []),
        "warnings": report.get("warnings", []),
        "internal_signals": report.get("internal_model_signals", {}),
        "publishable_signals_ge_70": report.get("publishable_signals_confidence_ge_0.70")
        or report.get("telegram_publishable_signals_confidence_ge_70", {}),
        **side_fields,
        "internal_backtest": {
            "net_pnl_usd": internal.get("net_pnl_usd", 0.0),
            "trade_count": internal.get("trade_count", 0),
            "long_count": internal.get("long_count", 0),
            "short_count": internal.get("short_count", 0),
        },
        "publishable_backtest_ge_70": {
            "net_pnl_usd": publishable.get("net_pnl_usd", 0.0),
            "trade_count": publishable.get("trade_count", 0),
            "expected_ev_usd": publishable.get("expected_ev_usd", publishable.get("net_pnl_usd", 0.0)),
        },
        "training_metrics": training_metrics or report.get("training_metrics"),
        "success": False,
    }
