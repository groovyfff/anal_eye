"""Label generation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ae_brain.training.labels import LABEL_LONG, LABEL_SHORT, LABEL_SKIP, LabelConfig, ev_aware_directional_labels
from ae_brain.training.synthetic import generate_synthetic_candles


def test_labels_include_long_short_skip() -> None:
    candles = generate_synthetic_candles(n=400, seed=42)
    atr = (candles["close"].diff().abs().fillna(1.0) * 3).to_numpy(float)
    labels = ev_aware_directional_labels(candles, atr, cfg=LabelConfig(horizon=12, min_net_reward_usd=0.0))
    assert LABEL_LONG in labels
    assert LABEL_SHORT in labels
    assert LABEL_SKIP in labels
    assert int((labels == LABEL_LONG).sum()) > 0 or int((labels == LABEL_SHORT).sum()) > 0


def test_labels_symmetric_classes_present() -> None:
    candles = generate_synthetic_candles(n=800, seed=7)
    atr = np.full(len(candles), float(candles["close"].mean() * 0.01))
    labels = ev_aware_directional_labels(candles, atr, cfg=LabelConfig(horizon=24, min_net_reward_usd=0.0))
    long_n = int((labels == LABEL_LONG).sum())
    short_n = int((labels == LABEL_SHORT).sum())
    assert long_n > 0, "LONG labels too rare — investigate barrier/cost settings"
    assert short_n > 0, "SHORT labels too rare — investigate barrier/cost settings"
