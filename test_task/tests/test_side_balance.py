"""Tests for Top200 side-balance class balancing, leakage-free sampling, and promotion.

Covers:
* class balancing config is passed to side specialists.
* validation/test splits remain chronological.
* no leakage from balanced training sampling.
* not promotable if LONG share below minimum.
* promotable only when LONG and SHORT both pass minimum share/count.
* memory-safe skip-sequence path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ae_brain.layers.side_specialists import SideSpecialistModel
from ae_brain.training.promotion import PromotionRules, evaluate_promotion
from ae_brain.training.specialist_metrics import (
    calibration_ceiling_summary,
    second_pass_threshold_report,
    side_balance_report,
)
from ae_brain.training.specialist_train import (
    _balanced_train_idx,
    _resolve_weight,
    train_specialist_pair,
)


# --------------------------------------------------------------------------- #
# 1. Class balancing config is passed to side specialists
# --------------------------------------------------------------------------- #
def test_resolve_weight_auto_computes_imbalance_ratio() -> None:
    # 90 negatives, 10 positives -> scale_pos_weight = 9.0
    w = _resolve_weight("auto", pos_count=10, neg_count=90)
    assert w == pytest.approx(9.0)


def test_resolve_weight_float_passed_through() -> None:
    assert _resolve_weight(2.5, pos_count=10, neg_count=90) == pytest.approx(2.5)


def test_resolve_weight_none_disables() -> None:
    assert _resolve_weight(None, pos_count=10, neg_count=90) is None


def test_resolve_weight_auto_zero_positives_returns_none() -> None:
    assert _resolve_weight("auto", pos_count=0, neg_count=90) is None


def test_resolve_weight_rejects_bad_string() -> None:
    with pytest.raises(ValueError, match="invalid weight"):
        _resolve_weight("bogus", pos_count=10, neg_count=90)


def test_side_specialist_records_scale_pos_weight_in_metrics() -> None:
    rng = np.random.default_rng(7)
    F = rng.normal(size=(300, 7)).astype(np.float32)
    y = (rng.random(300) < 0.2).astype(int)  # imbalanced
    model = SideSpecialistModel("LONG", model_kind="logreg")
    metrics = model.fit(F, y, train_end=200, scale_pos_weight=4.0)
    assert metrics["scale_pos_weight"] == pytest.approx(4.0)
    assert metrics["class_weight"] is None


def test_side_specialist_records_class_weight_in_metrics() -> None:
    rng = np.random.default_rng(8)
    F = rng.normal(size=(300, 7)).astype(np.float32)
    y = (rng.random(300) < 0.2).astype(int)
    model = SideSpecialistModel("LONG", model_kind="logreg")
    metrics = model.fit(F, y, train_end=200, class_weight=3.0)
    assert metrics["class_weight"] == pytest.approx(3.0)


def test_train_specialist_pair_records_auto_scale_pos_weight() -> None:
    rng = np.random.default_rng(11)
    n = 400
    F = rng.normal(size=(n, 7)).astype(np.float32)
    cut_train, cut_val = 280, 360
    y_long = np.zeros(n, dtype=int)
    y_long[:cut_train] = (rng.random(cut_train) < 0.15).astype(int)
    y_short = np.zeros(n, dtype=int)
    y_short[:cut_train] = (rng.random(cut_train) < 0.30).astype(int)
    ev_long = rng.uniform(0, 10, n)
    ev_short = rng.uniform(0, 10, n)

    rep = train_specialist_pair(
        F, y_long, y_short, ev_long, ev_short,
        cut_train=cut_train, cut_val=cut_val,
        model_kind="logreg",
        long_scale_pos_weight="auto",
        short_scale_pos_weight="auto",
    )
    bal = rep["balancing"]
    # auto weights computed from imbalance: LONG is more imbalanced -> higher weight
    long_pos = int(y_long[:cut_train].sum())
    long_neg = cut_train - long_pos
    assert bal["long_scale_pos_weight"] == pytest.approx(long_neg / long_pos)
    assert bal["no_future_leakage"] is True
    assert rep["long_metrics"]["scale_pos_weight"] is not None


# --------------------------------------------------------------------------- #
# 2. Validation/test splits remain chronological
# --------------------------------------------------------------------------- #
def test_balanced_train_idx_stays_within_train_window() -> None:
    cut_train = 200
    y = np.zeros(300, dtype=int)
    y[:cut_train] = (np.arange(cut_train) % 3 == 0).astype(int)  # ~67 pos in train
    idx = _balanced_train_idx(y, cut_train=cut_train, max_per_class=None)
    # All selected indices must be < cut_train (no validation/test leakage)
    assert idx.max() < cut_train
    assert idx.min() >= 0


def test_balanced_train_idx_equalizes_classes() -> None:
    cut_train = 200
    y = np.zeros(300, dtype=int)
    y[:cut_train] = (np.arange(cut_train) < 150).astype(int)  # 150 pos, 50 neg
    idx = _balanced_train_idx(y, cut_train=cut_train, max_per_class=None)
    selected = y[idx]
    n_pos = int((selected == 1).sum())
    n_neg = int((selected == 0).sum())
    assert n_pos == n_neg  # equalized to minority (50)


def test_balanced_train_idx_respects_max_per_class_cap() -> None:
    cut_train = 200
    y = np.zeros(300, dtype=int)
    y[:cut_train] = (np.arange(cut_train) < 150).astype(int)
    idx = _balanced_train_idx(y, cut_train=cut_train, max_per_class=30)
    assert int((y[idx] == 1).sum()) == 30
    assert int((y[idx] == 0).sum()) == 30


def test_balanced_train_idx_does_not_touch_validation() -> None:
    cut_train = 200
    y = np.zeros(400, dtype=int)
    y[200:400] = 1  # all "positives" are in validation+test region
    idx = _balanced_train_idx(y, cut_train=cut_train, max_per_class=None)
    # The balanced train subset must never include any index >= cut_train.
    assert not (idx >= cut_train).any()


# --------------------------------------------------------------------------- #
# 3. No leakage from balanced training sampling
# --------------------------------------------------------------------------- #
def test_train_specialist_pair_balancing_does_not_leak_into_validation() -> None:
    rng = np.random.default_rng(3)
    n = 500
    F = rng.normal(size=(n, 6)).astype(np.float32)
    cut_train, cut_val = 350, 425
    # Make the positive class in train depend on F[:,0], and the validation labels
    # depend on a different signal, so we can confirm training only used [0, cut_train).
    y_long = np.zeros(n, dtype=int)
    y_long[:cut_train] = (F[:cut_train, 0] > 0).astype(int)
    y_long[cut_train:] = (F[cut_train:, 1] > 0).astype(int)
    y_short = 1 - y_long
    ev_long = rng.uniform(1, 5, n)
    ev_short = rng.uniform(1, 5, n)

    rep = train_specialist_pair(
        F, y_long, y_short, ev_long, ev_short,
        cut_train=cut_train, cut_val=cut_val,
        model_kind="logreg",
        balance_train_samples=True,
        max_side_train_samples_per_class=None,
    )
    bal = rep["balancing"]
    assert bal["balance_train_samples"] is True
    assert bal["train_split_chronological"] is True
    assert bal["validation_unchanged"] is True
    assert bal["no_future_leakage"] is True
    # The LONG train subset used only [0, cut_train) rows.
    assert bal["long_train_idx_size"] is not None
    assert bal["long_train_idx_size"] <= cut_train


# --------------------------------------------------------------------------- #
# 4. Not promotable if LONG share below minimum
# --------------------------------------------------------------------------- #
def _base_promotable_metrics(pub_long: int, pub_short: int) -> dict:
    total_ev = max(pub_long + pub_short, 1)
    return {
        "expected_ev_usd": 1000.0,
        "net_pnl_usd": 1000.0,
        "max_drawdown": 0.1,
        "trade_count": pub_long + pub_short + 10,
        "long_count": pub_long + 5,
        "short_count": pub_short + 5,
        "precision_at_conf_70": 0.6,
        "per_symbol": {
            "BTCUSDT": {"net_pnl_usd": 500.0, "trades": 10},
            "ETHUSDT": {"net_pnl_usd": 500.0, "trades": 10},
        },
        "ev_by_confidence_bucket": {"0.70-0.80": 100.0},
        "internal_model_signals": {"LONG": pub_long, "SHORT": pub_short, "SKIP": 0},
        "publishable_signals_ge_70": {"LONG": pub_long, "SHORT": pub_short},
        "backtest_publishable_ge_70": {"trade_count": pub_long + pub_short, "net_pnl_usd": 1000.0},
        "publishable_long_count_ge_70": pub_long,
        "publishable_short_count_ge_70": pub_short,
        "publishable_long_ev_ge_70": float(pub_long) * 10,
        "publishable_short_ev_ge_70": float(pub_short) * 10,
        "publishable_total_ev_ge_70": float(total_ev) * 10,
        "publishable_total_trade_count_ge_70": pub_long + pub_short,
    }


def test_not_promotable_if_long_share_below_minimum() -> None:
    # 8 LONG vs 97 SHORT -> long_share = 0.076 < 0.10 (the original failing case)
    metrics = _base_promotable_metrics(pub_long=8, pub_short=97)
    result = evaluate_promotion(metrics, rules=PromotionRules(min_trades=20))
    assert result.passed is False
    assert any("publishable_long_share_too_low" in r for r in result.reasons)
    # The report exposes the exact minimum required share and current shares.
    sb = result.metrics["side_balance"]
    assert sb["publishable_long_share"] == pytest.approx(8 / 105, rel=1e-3)
    assert sb["min_required_long_share"] == 0.10


def test_promotable_when_both_sides_pass_minimum_share() -> None:
    metrics = _base_promotable_metrics(pub_long=50, pub_short=50)
    result = evaluate_promotion(metrics, rules=PromotionRules(min_trades=20))
    assert result.passed is True
    assert not any("share_too_low" in r for r in result.reasons)


def test_promotable_fails_if_short_share_below_minimum() -> None:
    metrics = _base_promotable_metrics(pub_long=97, pub_short=8)
    result = evaluate_promotion(metrics, rules=PromotionRules(min_trades=20))
    assert result.passed is False
    assert any("publishable_short_share_too_low" in r for r in result.reasons)


def test_promotion_min_share_is_configurable() -> None:
    # With a 0.20 minimum LONG share, a 50/50 split still passes.
    metrics = _base_promotable_metrics(pub_long=50, pub_short=50)
    rules = PromotionRules(min_trades=20, min_publishable_long_share=0.20)
    result = evaluate_promotion(metrics, rules=rules)
    assert result.passed is True


# --------------------------------------------------------------------------- #
# 5. Diagnostics: side balance + calibration ceiling + threshold pass
# --------------------------------------------------------------------------- #
def test_side_balance_report_counts_labels_by_split() -> None:
    report = side_balance_report(
        y_long_train=np.array([1, 0, 1, 0, 0]),
        y_short_train=np.array([0, 1, 0, 1, 0]),
        y_long_val=np.array([1, 0, 0]),
        y_short_val=np.array([0, 1, 1]),
    )
    assert report["label_counts"]["train"]["LONG_profitable"] == 2
    assert report["label_counts"]["train"]["SHORT_profitable"] == 2
    assert report["label_counts"]["validation"]["LONG_profitable"] == 1
    assert report["label_counts"]["validation"]["SHORT_profitable"] == 2


def test_calibration_ceiling_summary_flags_low_long_ceiling() -> None:
    long_cal = np.array([0.4, 0.5, 0.6, 0.45])  # never reaches 0.70
    short_cal = np.array([0.8, 0.9, 0.75, 0.85])
    summary = calibration_ceiling_summary(long_cal, short_cal)
    assert summary["LONG"]["ceiling_too_low"] is True
    assert summary["SHORT"]["ceiling_too_low"] is False
    assert summary["LONG"]["crossing_publish_threshold"] == 0
    assert summary["SHORT"]["crossing_publish_threshold"] == 4
    assert summary["long_ceiling_diagnosis"] == "long_calibrated_max_below_publish_threshold"


def test_second_pass_threshold_report_is_diagnostic_only() -> None:
    long_cal = np.array([0.6, 0.72, 0.8])
    short_cal = np.array([0.7, 0.85, 0.9])
    long_ev = np.array([1.0, 2.0, 3.0])
    short_ev = np.array([2.0, 3.0, 4.0])
    report = second_pass_threshold_report(long_cal, short_cal, long_ev, short_ev)
    assert report["diagnostic_only"] is True
    assert report["promotion_threshold_fixed_at"] == 0.70
    assert "0.65" in report["per_threshold"]
    assert "0.70" in report["per_threshold"]
    assert "0.75" in report["per_threshold"]
    # long_share computed at each threshold
    t70 = report["per_threshold"]["0.70"]
    assert "long_share" in t70
    assert "short_share" in t70


# --------------------------------------------------------------------------- #
# 6. Memory-safe skip-sequence path
# --------------------------------------------------------------------------- #
def test_seq_series_returns_neutral_when_module_is_none() -> None:
    from ae_brain.training.meta_series import seq_series
    import pandas as pd

    frame = pd.DataFrame({"close": np.arange(100.0), "open": np.arange(100.0)})
    p_cont, trend = seq_series(None, None, None, frame, window=48, device="cpu")
    assert p_cont.shape == (100,)
    assert trend.shape == (100,)
    assert np.allclose(p_cont, 0.5)
    assert np.allclose(trend, 0.0)


def test_train_specialist_pair_works_with_no_sequence_metadata() -> None:
    # The specialist pair trainer should not depend on a sequence module; it
    # consumes already-assembled feature rows. This mirrors the skip-sequence path
    # where seq_module is None but specialist features are still assembled.
    rng = np.random.default_rng(5)
    n = 300
    F = rng.normal(size=(n, 7)).astype(np.float32)
    cut_train, cut_val = 200, 260
    y_long = (rng.random(n) < 0.25).astype(int)
    y_short = (rng.random(n) < 0.25).astype(int)
    ev_long = rng.uniform(0, 5, n)
    ev_short = rng.uniform(0, 5, n)
    rep = train_specialist_pair(
        F, y_long, y_short, ev_long, ev_short,
        cut_train=cut_train, cut_val=cut_val,
        model_kind="logreg",
    )
    assert rep["long_model"].is_ready()
    assert rep["short_model"].is_ready()
    assert rep["side_balance"] is not None
    assert rep["calibration_ceiling_summary"] is not None
    assert rep["second_pass_threshold_report"] is not None
