"""Walk-forward chronological splits with purging/embargo for overlapping labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TimeSplit:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def chronological_mask(
    timestamps: pd.Series,
    *,
    train_end: str,
    val_end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by UTC date boundaries: train < train_end <= val < val_end <= test."""
    ts = pd.to_datetime(timestamps, utc=True)
    t_train = pd.Timestamp(train_end, tz="UTC")
    t_val = pd.Timestamp(val_end, tz="UTC")
    train = np.asarray(ts < t_train)
    val = np.asarray((ts >= t_train) & (ts < t_val))
    test = np.asarray(ts >= t_val)
    return train, val, test


def purge_embargo_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    label_horizon: int,
    embargo: int | None = None,
) -> np.ndarray:
    """Remove train rows whose label window overlaps the validation start."""
    gap = label_horizon + (embargo if embargo is not None else label_horizon)
    if val_idx.size == 0 or train_idx.size == 0:
        return train_idx
    val_start = int(np.min(val_idx))
    cutoff = val_start - gap
    return train_idx[train_idx < cutoff]


def walk_forward_folds(
    timestamps: pd.Series,
    *,
    n_folds: int = 4,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    label_horizon: int = 24,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Return list of (train_idx, val_idx) expanding windows (chronological)."""
    n = len(timestamps)
    ts = pd.to_datetime(timestamps, utc=True).sort_values()
    order = np.argsort(ts.to_numpy())
    test_n = max(1, int(n * test_frac))
    holdout = order[-test_n:]
    work = order[:-test_n]
    fold_size = max(1, int(len(work) * val_frac))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        val_end = min(len(work), (k + 1) * fold_size + fold_size)
        val_start = max(0, val_end - fold_size)
        val_idx = work[val_start:val_end]
        train_idx = work[:val_start]
        train_idx = purge_embargo_indices(train_idx, val_idx, label_horizon=label_horizon)
        if train_idx.size > 100 and val_idx.size > 10:
            folds.append((train_idx, val_idx))
    return folds, holdout


def make_time_split(
    timestamps: pd.Series,
    *,
    train_end: str = "2024-01-01",
    val_end: str = "2025-01-01",
    label_horizon: int = 24,
) -> TimeSplit:
    train, val, test = chronological_mask(timestamps, train_end=train_end, val_end=val_end)
    train_idx = np.flatnonzero(train)
    val_idx = np.flatnonzero(val)
    test_idx = np.flatnonzero(test)
    train_idx = purge_embargo_indices(train_idx, val_idx, label_horizon=label_horizon)
    return TimeSplit("default", train_idx, val_idx, test_idx)
