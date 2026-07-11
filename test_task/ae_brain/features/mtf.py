"""15-minute features aggregated onto 1h rows for multi-timeframe specialist inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MTF_15M_FEATURE_NAMES: tuple[str, ...] = (
    "mtf_15m_ret_4",
    "mtf_15m_ret_8",
    "mtf_15m_ret_16",
    "mtf_15m_volatility",
    "mtf_15m_rsi",
    "mtf_15m_volume_impulse",
    "mtf_15m_trend_strength",
    "mtf_15m_high_breakout_dist",
    "mtf_15m_low_breakout_dist",
)


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = pd.Series(np.where(delta > 0, delta, 0.0)).ewm(alpha=1 / period).mean()
    loss = pd.Series(np.where(delta < 0, -delta, 0.0)).ewm(alpha=1 / period).mean()
    rs = np.divide(gain.to_numpy(), loss.to_numpy(), out=np.zeros_like(gain), where=loss.to_numpy() != 0)
    return 100 - 100 / (1 + rs)


def compute_15m_features_for_1h(
    candles_1h: pd.DataFrame,
    candles_15m: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Align 15m sub-bars to each 1h timestamp (last closed 1h bar uses prior 15m window)."""
    n = len(candles_1h)
    out = {name: np.zeros(n, dtype=float) for name in MTF_15M_FEATURE_NAMES}
    if candles_15m.empty or n == 0:
        return out

    ts_1h = pd.to_datetime(candles_1h["ts"] if "ts" in candles_1h.columns else candles_1h["timestamp"], utc=True)
    ts_15 = pd.to_datetime(candles_15m["ts"] if "ts" in candles_15m.columns else candles_15m["timestamp"], utc=True)
    c15 = candles_15m.copy()
    c15["ts"] = ts_15
    close15 = c15["close"].to_numpy(float)
    high15 = c15["high"].to_numpy(float)
    low15 = c15["low"].to_numpy(float)
    vol15 = c15["volume"].to_numpy(float) if "volume" in c15.columns else np.ones(len(c15))
    log_ret15 = np.log(np.maximum(close15 / np.roll(close15, 1), 1e-12))
    log_ret15[0] = 0.0
    rsi15 = _rsi(close15)

    for i, t_end in enumerate(ts_1h):
        sub = c15[c15["ts"] <= t_end].tail(16)
        if len(sub) < 4:
            continue
        sc = sub["close"].to_numpy(float)
        sh = sub["high"].to_numpy(float)
        sl = sub["low"].to_numpy(float)
        sv = sub["volume"].to_numpy(float) if "volume" in sub.columns else np.ones(len(sub))
        lr = np.log(np.maximum(sc / np.roll(sc, 1), 1e-12))
        lr[0] = 0.0

        def _ret(k: int) -> float:
            if len(sc) <= k:
                return 0.0
            return float(np.log(max(sc[-1] / sc[-k - 1], 1e-12)))

        out["mtf_15m_ret_4"][i] = _ret(4) if len(sc) > 4 else _ret(len(sc) - 1)
        out["mtf_15m_ret_8"][i] = _ret(8) if len(sc) > 8 else _ret(len(sc) - 1)
        out["mtf_15m_ret_16"][i] = _ret(16) if len(sc) > 16 else _ret(len(sc) - 1)
        out["mtf_15m_volatility"][i] = float(np.std(lr[-min(16, len(lr)):]))
        out["mtf_15m_rsi"][i] = float(rsi15[sub.index[-1] - c15.index[0]]) if len(rsi15) else 50.0
        vol_z = (sv[-1] - np.mean(sv)) / max(np.std(sv), 1e-9)
        out["mtf_15m_volume_impulse"][i] = float(vol_z * out["mtf_15m_ret_4"][i])
        ema_fast = pd.Series(sc).ewm(span=4).mean().iloc[-1]
        ema_slow = pd.Series(sc).ewm(span=12).mean().iloc[-1]
        out["mtf_15m_trend_strength"][i] = float((ema_fast - ema_slow) / max(sc[-1], 1e-9))
        roll_high = float(np.max(sh))
        roll_low = float(np.min(sl))
        atr_proxy = max(float(np.mean(sh - sl)), sc[-1] * 1e-4)
        out["mtf_15m_high_breakout_dist"][i] = (sc[-1] - roll_high) / atr_proxy
        out["mtf_15m_low_breakout_dist"][i] = (roll_low - sc[-1]) / atr_proxy

    return out


def load_15m_candles(data_dir: Path, symbol: str) -> pd.DataFrame:
    path = data_dir / f"{symbol}_15m.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "ts" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    return df
