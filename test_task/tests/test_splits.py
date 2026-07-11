"""Walk-forward split tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ae_brain.training.splits import chronological_mask, make_time_split, purge_embargo_indices


def test_chronological_split_preserves_order() -> None:
    ts = pd.date_range("2020-01-01", periods=1000, freq="h", tz="UTC")
    train, val, test = chronological_mask(ts, train_end="2020-01-15", val_end="2020-01-25")
    assert train.sum() > 0 and val.sum() > 0 and test.sum() > 0


def test_purge_embargo_removes_overlapping_train_rows() -> None:
    train_idx = np.arange(0, 500)
    val_idx = np.arange(480, 520)
    purged = purge_embargo_indices(train_idx, val_idx, label_horizon=24, embargo=24)
    assert purged.max() < 480 - 48


def test_make_time_split_indices_disjoint() -> None:
    ts = pd.date_range("2021-01-01", periods=5000, freq="h", tz="UTC")
    split = make_time_split(ts, train_end="2023-01-01", val_end="2024-01-01", label_horizon=24)
    assert not set(split.train_idx).intersection(split.val_idx)
    assert not set(split.val_idx).intersection(split.test_idx)
