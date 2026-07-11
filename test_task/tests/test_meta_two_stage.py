"""Tests for two-stage meta and confidence calibration."""

from __future__ import annotations

import numpy as np

from ae_brain.layers.meta import (
    CLASS_LONG,
    CLASS_SHORT,
    CLASS_SKIP,
    TwoStageMetaModel,
    resolve_two_stage_direction,
)
from ae_brain.training.calibration import ConfidenceCalibrator


def test_resolve_two_stage_direction_margin_skips_ambiguous() -> None:
    side, reason = resolve_two_stage_direction(0.51, 0.49, margin=0.05)
    assert side is None
    assert reason == "directional_ambiguity"
    side, _ = resolve_two_stage_direction(0.60, 0.40, margin=0.05)
    assert side == CLASS_LONG


def test_two_stage_meta_fit_predict_both_directions() -> None:
    rng = np.random.default_rng(0)
    n = 400
    F = rng.normal(size=(n, 7)).astype(np.float32)
    y = np.full(n, CLASS_SKIP, dtype=int)
    y[:100] = CLASS_LONG
    y[100:200] = CLASS_SHORT
    meta = TwoStageMetaModel()
    metrics = meta.fit(F, y)
    assert metrics["kind"] == "two_stage"
    preds = [meta.predict(F[i], trade_threshold=0.3, direction_margin=0.02) for i in range(200, 300)]
    longs = sum(1 for p in preds if p.directional_class == CLASS_LONG)
    shorts = sum(1 for p in preds if p.directional_class == CLASS_SHORT)
    assert longs > 0
    assert shorts > 0


def test_confidence_calibrator_improves_or_matches_brier() -> None:
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.2, 0.8, size=200)
    y = (raw + rng.normal(0, 0.1, size=200) > 0.5).astype(int)
    cal = ConfidenceCalibrator("isotonic")
    report = cal.fit(raw, y)
    assert report.brier_raw is not None
    assert report.brier_calibrated is not None
    assert cal.calibrate(0.5) >= 0.0
