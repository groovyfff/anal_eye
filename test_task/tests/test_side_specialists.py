"""Tests for independent side specialist models."""

from __future__ import annotations

import numpy as np

from ae_brain.layers.side_specialists import SideSpecialistModel


def test_side_specialist_binary_fit_predict() -> None:
    rng = np.random.default_rng(42)
    F = rng.normal(size=(200, 7)).astype(np.float32)
    y_long = (F[:, 0] > 0).astype(int)
    model = SideSpecialistModel("LONG")
    metrics = model.fit(F, y_long, train_end=140)
    assert model.is_ready()
    assert metrics["n_train"] == 140
    raw = model.predict_raw(F[150])
    assert 0.0 <= raw <= 1.0
