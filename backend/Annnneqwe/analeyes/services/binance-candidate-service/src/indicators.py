from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.candle_buffer import Candle


def _series(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def ema_last(values: list[float], period: int) -> float | None:
    if len(values) < period or period < 1:
        return None
    arr = _series(values)
    alpha = 2.0 / (period + 1.0)
    ema = arr[0]
    for price in arr[1:]:
        ema = alpha * price + (1.0 - alpha) * ema
    return float(ema)


def rsi_last(values: list[float], period: int = 14) -> float | None:
    if len(values) < period + 1:
        return None
    arr = _series(values)
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def macd_last(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float | None, float | None, float | None]:
    if len(values) < slow + signal:
        return None, None, None
    arr = _series(values)

    def _ema_series(data: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(data)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
        return out

    ema_fast = _ema_series(arr, fast)
    ema_slow = _ema_series(arr, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema_series(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return float(macd_line[-1]), float(macd_signal[-1]), float(macd_hist[-1])


def atr_last(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return None
    return float(np.mean(trs[-period:]))


def adx_last(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period * 2 + 1:
        return None
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    trs: list[float] = []
    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = float(np.mean(trs[-period:]))
    if atr == 0:
        return None
    plus_di = 100.0 * float(np.mean(plus_dm[-period:])) / atr
    minus_di = 100.0 * float(np.mean(minus_dm[-period:])) / atr
    denom = plus_di + minus_di
    if denom == 0:
        return None
    dx = abs(plus_di - minus_di) / denom * 100.0
    return float(dx)


def pct_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return float((current - previous) / previous * 100.0)


def compute_features(candles: list[Candle]) -> dict[str, Any]:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]
    current_price = closes[-1]

    ema_short = ema_last(closes, 12)
    ema_long = ema_last(closes, 26)
    ema_50 = ema_last(closes, 50)
    ema_200 = ema_last(closes, 200) if len(closes) >= 200 else ema_last(closes, min(len(closes), 100))

    macd, macd_signal, macd_hist = macd_last(closes)
    rsi = rsi_last(closes, 14)
    atr = atr_last(highs, lows, closes, 14)
    adx = adx_last(highs, lows, closes, 14)

    volume_change: float | None = None
    if len(volumes) >= 2 and volumes[-2] > 0:
        volume_change = float((volumes[-1] - volumes[-2]) / volumes[-2])

    price_change_1h: float | None = pct_change(closes[-1], closes[-2]) if len(closes) >= 2 else None
    price_change_24h: float | None = pct_change(closes[-1], closes[-25]) if len(closes) >= 25 else None

    features: dict[str, Any] = {
        "current_price": current_price,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "adx": adx,
        "atr": atr,
        "ema_short": ema_short,
        "ema_long": ema_long,
        "ema_50": ema_50,
        "ema_200": ema_200,
        "volume_change": volume_change,
        "price_change_1h": price_change_1h,
        "price_change_24h": price_change_24h,
    }
    return features


def infer_market_state(features: dict[str, Any], current_price: float) -> str:
    adx = features.get("adx")
    atr = features.get("atr")
    ema_short = features.get("ema_short")
    ema_long = features.get("ema_long")

    if adx is None or atr is None:
        return "unknown"

    atr_pct = atr / current_price if current_price > 0 else 0.0
    if atr_pct >= 0.02:
        return "volatile"

    if adx >= 25 and ema_short is not None and ema_long is not None and ema_short != ema_long:
        return "trend"

    if adx < 20:
        return "range"

    return "unknown"


def compute_composite_score(features: dict[str, Any]) -> float:
    """Neutral technical strength score in [0, 1] — not a trade decision."""
    parts: list[float] = []

    rsi = features.get("rsi")
    if rsi is not None:
        parts.append(min(abs(float(rsi) - 50.0) / 50.0, 1.0))

    macd_hist = features.get("macd_hist")
    if macd_hist is not None:
        parts.append(min(abs(float(macd_hist)) / 2.0, 1.0))

    adx = features.get("adx")
    if adx is not None:
        parts.append(min(max(float(adx), 0.0) / 50.0, 1.0))

    volume_change = features.get("volume_change")
    if volume_change is not None:
        parts.append(min(abs(float(volume_change)), 1.0))

    if not parts:
        return 0.0
    return round(float(sum(parts) / len(parts)), 4)
