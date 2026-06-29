"""Dataset construction: labeling + windowing.

* **Tabular** - triple-barrier labels: for each candle, does price reach the
  +ATR*tp_mult barrier before the -ATR*sl_mult barrier within a horizon? This
  makes the tabular target *exactly* the ``prob_tp`` quantity the EV gate needs.
* **Sequence** - sliding windows + a continuation/reversal label (does the next
  k-bar return agree with the trailing trend?) plus a trend-sign regression
  target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ae_brain.features.engineering import FeatureEngineer
from ae_brain.features.schema import FEATURE_NAMES
from ae_brain.layers.sequence import SEQ_CHANNELS


def relative_vol_scale(
    atr_pct: np.ndarray, window: int = 200, lo: float = 0.5, hi: float = 2.0
) -> np.ndarray:
    """Per-bar multiplier that adapts the *barrier multiple* to relative vol.

    The triple barrier is already ATR-scaled (so absolute TP/SL distance grows
    with volatility). This goes one step further and adapts the *multiple*
    itself: ``atr_pct / rolling_median(atr_pct)`` -> >1 when the current bar is
    more volatile than its recent norm (wider targets), <1 when calmer (tighter
    targets), clipped to ``[lo, hi]``.
    """
    s = pd.Series(np.asarray(atr_pct, dtype=float))
    med = s.rolling(window, min_periods=max(5, window // 4)).median()
    med = med.bfill().fillna(s.median() if s.median() > 0 else 1.0)
    scale = (s / med.replace(0.0, np.nan)).fillna(1.0)
    return np.clip(scale.to_numpy(), lo, hi)


def triple_barrier_labels(
    candles: pd.DataFrame,
    atr: np.ndarray,
    *,
    tp_mult: float = 2.5,
    sl_mult: float = 1.5,
    horizon: int = 24,
    vol_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Return {1: TP-first, 0: SL-first or timeout} per candle (long view).

    Barriers are ATR-scaled per bar; ``vol_scale`` (see :func:`relative_vol_scale`)
    additionally adapts the multiple by the bar's *relative* volatility.
    """
    close = candles["close"].to_numpy(float)
    high = candles["high"].to_numpy(float)
    low = candles["low"].to_numpy(float)
    n = len(close)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n - 1):
        entry = close[i]
        vs = float(vol_scale[i]) if vol_scale is not None else 1.0
        up = entry + atr[i] * tp_mult * vs
        dn = entry - atr[i] * sl_mult * vs
        end = min(i + horizon, n)
        hit = 0
        for j in range(i + 1, end):
            if high[j] >= up:
                hit = 1
                break
            if low[j] <= dn:
                hit = 0
                break
        labels[i] = hit
    return labels


def directional_barrier_labels(
    candles: pd.DataFrame,
    atr: np.ndarray,
    *,
    tp_mult: float = 2.5,
    sl_mult: float = 1.5,
    horizon: int = 24,
    vol_scale: np.ndarray | None = None,
) -> np.ndarray:
    """3-class first-passage label per candle for the meta-model.

    Returns ``{0: SHORT, 1: SKIP, 2: LONG}``:

    * **LONG (2)**  - the +TP barrier is hit before the long stop (and, if a
      short setup also resolves, the long take-profit happens no later).
    * **SHORT (0)** - the -TP barrier is hit before the short stop (and earlier
      than any long take-profit).
    * **SKIP (1)**  - neither side reaches its take-profit within the horizon.
    """
    close = candles["close"].to_numpy(float)
    high = candles["high"].to_numpy(float)
    low = candles["low"].to_numpy(float)
    n = len(close)
    labels = np.ones(n, dtype=np.int64)  # default SKIP
    for i in range(n - 1):
        entry = close[i]
        vs = float(vol_scale[i]) if vol_scale is not None else 1.0
        u_tp = entry + atr[i] * tp_mult * vs   # long take-profit
        d_sl = entry - atr[i] * sl_mult * vs   # long stop
        d_tp = entry - atr[i] * tp_mult * vs   # short take-profit
        u_sl = entry + atr[i] * sl_mult * vs   # short stop
        end = min(i + horizon, n)
        long_t: int | None = None
        short_t: int | None = None
        for j in range(i + 1, end):
            hj, lj = high[j], low[j]
            if long_t is None:
                if hj >= u_tp:
                    long_t = j
                elif lj <= d_sl:
                    long_t = -1  # stopped out -> invalid
            if short_t is None:
                if lj <= d_tp:
                    short_t = j
                elif hj >= u_sl:
                    short_t = -1
            if long_t is not None and short_t is not None:
                break
        long_win = long_t is not None and long_t > 0
        short_win = short_t is not None and short_t > 0
        if long_win and short_win:
            labels[i] = 2 if long_t <= short_t else 0
        elif long_win:
            labels[i] = 2
        elif short_win:
            labels[i] = 0
        else:
            labels[i] = 1
    return labels


def build_tabular_dataset(
    candles: pd.DataFrame,
    *,
    tp_mult: float = 2.5,
    sl_mult: float = 1.5,
    horizon: int = 24,
    z_window: int = 100,
    regime_model: object | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names) for the tabular layer."""
    eng = FeatureEngineer(z_window=z_window, regime_model=regime_model)
    feats = eng.compute_frame(candles)
    atr = feats["atr_14"].to_numpy(float)
    atr = np.where(atr <= 0, candles["close"].to_numpy(float) * 0.005, atr)
    vol_scale = relative_vol_scale(feats["atr_pct"].to_numpy(float))
    y = triple_barrier_labels(
        candles, atr, tp_mult=tp_mult, sl_mult=sl_mult, horizon=horizon, vol_scale=vol_scale
    )

    X = feats[list(FEATURE_NAMES)].to_numpy(np.float32)
    # Drop the tail where the horizon would run off the end.
    valid = slice(z_window, len(X) - horizon)
    return X[valid], y[valid], list(FEATURE_NAMES)


def build_sequence_dataset(
    candles: pd.DataFrame,
    *,
    window: int = 48,
    horizon: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return (X_windows, y_continuation, y_trendsign, (mean, std)).

    X_windows: (N, window, C) standardized channel windows.
    y_continuation: 1 if next-`horizon` return agrees with trailing trend.
    y_trendsign: trailing trend sign in [-1, 1].
    """
    df = candles
    chans = np.column_stack(
        [df[c].to_numpy(float) if c in df else np.zeros(len(df)) for c in SEQ_CHANNELS]
    )
    close = df["close"].to_numpy(float)

    mean = chans.mean(axis=0)
    std = chans.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    norm = (chans - mean) / std

    xs, yc, ys = [], [], []
    for i in range(window, len(df) - horizon):
        xs.append(norm[i - window : i])
        trail = (close[i - 1] - close[i - window]) / close[i - window]
        fwd = (close[i + horizon] - close[i]) / close[i]
        trend_sign = np.sign(trail)
        # continuation if forward move agrees with trailing trend
        yc.append(1 if np.sign(fwd) == trend_sign and trend_sign != 0 else 0)
        ys.append(float(np.clip(trail * 50, -1, 1)))  # squashed trend strength

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(yc, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        (mean.astype(np.float32), std.astype(np.float32)),
    )
