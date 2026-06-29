"""Feature engineering: raw OHLCV + microstructure -> canonical feature vector.

The :class:`FeatureEngineer` is deliberately pure/CPU-bound and stateless so it
can be safely dispatched to a ``ProcessPoolExecutor`` from the async inference
loop. It returns features in the exact order defined by ``FEATURE_NAMES``.

TA-Lib is imported lazily; if it is unavailable we fall back to vectorized
numpy/pandas implementations so unit tests can run without the C library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ae_brain.features.schema import FEATURE_NAMES, REGIME_ONEHOT_NAMES

try:  # TA-Lib is the production path (pinned 0.4.28, requires numpy<2).
    import talib  # type: ignore

    _HAS_TALIB = True
except Exception:  # pragma: no cover - fallback path
    talib = None  # type: ignore
    _HAS_TALIB = False


def _zscore(s: pd.Series, window: int = 100) -> pd.Series:
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=float)), where=np.asarray(b) != 0)


def _coerce_series(df: pd.DataFrame, name: str, default) -> pd.Series:
    """Return a numeric Series for ``name``, mapping null/missing -> ``default``.

    This is the single choke-point for null-handling across asset classes. For
    traditional assets ('stock'/'metal'/'forex') derivative microstructure
    columns (funding_rate, open_interest, taker_buy_volume, liquidations, ...)
    arrive either absent or full of JSON ``null`` -> here they are coerced to a
    neutral numeric default (scalar) or a per-row array default (e.g. 0.5*volume
    for taker-buy), preventing ``ValueError``/dtype issues downstream.
    """
    n = len(df)
    if name in df.columns:
        s = pd.to_numeric(df[name], errors="coerce")
    else:
        s = pd.Series(np.full(n, np.nan), index=df.index)
    if np.isscalar(default):
        return s.fillna(float(default))
    default_s = pd.Series(np.asarray(default, dtype=float), index=df.index)
    return s.fillna(default_s)


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    if _HAS_TALIB:
        return talib.RSI(close, timeperiod=period)
    delta = np.diff(close, prepend=close[0])
    gain = pd.Series(np.where(delta > 0, delta, 0.0)).ewm(alpha=1 / period).mean()
    loss = pd.Series(np.where(delta < 0, -delta, 0.0)).ewm(alpha=1 / period).mean()
    rs = _safe_div(gain.to_numpy(), loss.to_numpy())
    return 100 - 100 / (1 + rs)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    if _HAS_TALIB:
        return talib.ATR(high, low, close, timeperiod=period)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    return pd.Series(tr).ewm(alpha=1 / period).mean().to_numpy()


def _hurst(ts: np.ndarray) -> float:
    """Rescaled-range Hurst exponent on a short window (NaN-safe)."""
    ts = ts[~np.isnan(ts)]
    if ts.size < 20:
        return 0.5
    lags = range(2, min(20, ts.size // 2))
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    tau = np.asarray(tau)
    if np.any(tau <= 0):
        return 0.5
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0])


@dataclass(slots=True)
class FeatureEngineer:
    """Compute the canonical feature vector from a candle window.

    Parameters
    ----------
    z_window:
        Lookback (in candles) for rolling z-scores / normalization.
    regime_model:
        Optional fitted :class:`~ae_brain.features.regime.RegimeModel`. When
        present, ``compute_frame`` fills the ``regime_*`` one-hot columns from
        the per-row regime classification; when ``None`` they stay zero (a
        neutral, contract-preserving default).
    """

    z_window: int = 100
    regime_model: Any = None

    def compute_frame(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for every row of ``candles`` (vectorized).

        ``candles`` must contain at least: open, high, low, close, volume.
        Optional microstructure columns (filled with neutral defaults if absent):
        taker_buy_volume, open_interest, funding_rate, basis, bid_size, ask_size,
        spread, long_liq_notional, short_liq_notional, trade_count, ts.
        """
        df = candles.copy()
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df:
                raise ValueError(f"candles missing required column '{col}'")

        # Required OHLCV: coerce to numeric and forward/back-fill so a sporadic
        # null in a price/volume field cannot blow up the whole window.
        ohlcv = {
            k: pd.to_numeric(df[k], errors="coerce").ffill().bfill().fillna(0.0)
            for k in ("open", "high", "low", "close", "volume")
        }
        o, h, l, c, v = (ohlcv[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
        close = pd.Series(c, index=df.index)
        log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Optional microstructure with neutral fallbacks (null-safe across asset
        # classes; absent OR null -> neutral default). See _coerce_series.
        taker_buy = _coerce_series(df, "taker_buy_volume", v * 0.5).to_numpy(float)
        oi = _coerce_series(df, "open_interest", np.nan)
        funding = _coerce_series(df, "funding_rate", 0.0)
        basis = _coerce_series(df, "basis", 0.0)
        bid_size = _coerce_series(df, "bid_size", 1.0).to_numpy(float)
        ask_size = _coerce_series(df, "ask_size", 1.0).to_numpy(float)
        spread = _coerce_series(df, "spread", 0.0).to_numpy(float)
        long_liq = _coerce_series(df, "long_liq_notional", 0.0).to_numpy(float)
        short_liq = _coerce_series(df, "short_liq_notional", 0.0).to_numpy(float)
        trade_count = _coerce_series(df, "trade_count", 1.0)

        out: dict[str, Any] = {}

        # --- Volatility --------------------------------------------------
        atr = _atr(h, l, c, 14)
        out["atr_14"] = atr
        out["atr_pct"] = _safe_div(atr, c)
        rvol = log_ret.rolling(30, min_periods=10).std(ddof=0)
        out["realized_vol_30"] = rvol
        out["vol_z"] = _zscore(rvol.fillna(0.0), self.z_window)
        if _HAS_TALIB:
            up, mid, lo = talib.BBANDS(c, timeperiod=20)
        else:
            mid = close.rolling(20, min_periods=5).mean().to_numpy()
            sd = close.rolling(20, min_periods=5).std(ddof=0).to_numpy()
            up, lo = mid + 2 * sd, mid - 2 * sd
        out["bb_width"] = _safe_div(up - lo, mid)
        out["bb_pctb"] = _safe_div(c - lo, up - lo)
        out["parkinson_vol"] = np.sqrt(_safe_div((np.log(_safe_div(h, l))) ** 2, 4 * np.log(2)))
        out["vol_of_vol"] = rvol.rolling(30, min_periods=10).std(ddof=0)

        # --- Momentum ----------------------------------------------------
        rsi = _rsi(c, 14)
        out["rsi_14"] = rsi
        out["rsi_slope"] = pd.Series(rsi, index=df.index).diff()
        if _HAS_TALIB:
            macd, macd_sig, macd_hist = talib.MACD(c)
            out["adx_14"] = talib.ADX(h, l, c, 14)
            out["plus_di"] = talib.PLUS_DI(h, l, c, 14)
            out["minus_di"] = talib.MINUS_DI(h, l, c, 14)
            out["cci_20"] = talib.CCI(h, l, c, 20)
            out["willr_14"] = talib.WILLR(h, l, c, 14)
            out["stoch_k"], out["stoch_d"] = talib.STOCH(h, l, c)
            out["mfi_14"] = talib.MFI(h, l, c, v, 14)
            out["obv_slope"] = pd.Series(talib.OBV(c, v), index=df.index).diff()
            ema_fast = talib.EMA(c, 12)
            ema_slow = talib.EMA(c, 26)
        else:
            ema_fast = close.ewm(span=12).mean().to_numpy()
            ema_slow = close.ewm(span=26).mean().to_numpy()
            macd = ema_fast - ema_slow
            macd_sig = pd.Series(macd).ewm(span=9).mean().to_numpy()
            macd_hist = macd - macd_sig
            out["adx_14"] = pd.Series(np.abs(log_ret)).rolling(14).mean().to_numpy() * 100
            out["plus_di"] = np.full(len(df), 25.0)
            out["minus_di"] = np.full(len(df), 25.0)
            out["cci_20"] = _zscore(close, 20).to_numpy()
            out["willr_14"] = -50 + 50 * _zscore(close, 14).fillna(0.0).to_numpy()
            out["stoch_k"] = (_safe_div(c - l, h - l) * 100)
            out["stoch_d"] = pd.Series(out["stoch_k"]).rolling(3).mean().to_numpy()
            out["mfi_14"] = pd.Series(rsi).fillna(50.0).to_numpy()
            out["obv_slope"] = (np.sign(log_ret) * v).cumsum().diff().to_numpy()
        out["macd"] = macd
        out["macd_signal"] = macd_sig
        out["macd_hist"] = macd_hist
        out["ema_fast_slow_ratio"] = _safe_div(ema_fast, ema_slow) - 1
        vwap = _safe_div((c * v).cumsum(), v.cumsum())
        out["price_vs_vwap"] = _safe_div(c, vwap) - 1
        out["roc_10"] = close.pct_change(10)

        # --- Order flow --------------------------------------------------
        signed_vol = (2 * _safe_div(taker_buy, v) - 1) * v
        cvd = pd.Series(signed_vol, index=df.index).cumsum()
        out["cvd"] = cvd
        out["cvd_z"] = _zscore(cvd, self.z_window)
        out["cvd_slope"] = cvd.diff()
        ofi = (np.diff(bid_size, prepend=bid_size[0]) - np.diff(ask_size, prepend=ask_size[0]))
        out["ofi_delta"] = ofi
        out["ofi_z"] = _zscore(pd.Series(ofi, index=df.index), self.z_window)
        out["taker_buy_ratio"] = _safe_div(taker_buy, v)
        out["aggressor_imbalance"] = 2 * _safe_div(taker_buy, v) - 1
        out["trade_count_z"] = _zscore(pd.Series(trade_count.to_numpy(float), index=df.index), self.z_window)
        out["large_trade_ratio"] = _safe_div(np.abs(signed_vol), v).clip(0, 1) if False else _safe_div(taker_buy, v)
        out["book_imbalance"] = _safe_div(bid_size - ask_size, bid_size + ask_size)
        out["spread_bps"] = _safe_div(spread, c) * 1e4

        # --- Derivatives -------------------------------------------------
        out["oi_z"] = _zscore(oi.astype(float).ffill().fillna(0.0), self.z_window)
        out["oi_change_pct"] = oi.astype(float).pct_change().fillna(0.0)
        out["oi_price_divergence"] = np.sign(oi.astype(float).diff().fillna(0.0)) * np.sign(close.diff().fillna(0.0)) * -1
        out["funding_rate"] = funding.astype(float)
        out["funding_z"] = _zscore(funding.astype(float), self.z_window)
        out["basis_pct"] = basis.astype(float)
        out["liq_long_notional"] = np.log1p(long_liq)
        out["liq_short_notional"] = np.log1p(short_liq)
        out["liq_imbalance"] = _safe_div(long_liq - short_liq, long_liq + short_liq + 1.0)

        # --- Volume ------------------------------------------------------
        vol_s = pd.Series(v, index=df.index)
        out["volume_z"] = _zscore(vol_s, self.z_window)
        out["rel_volume"] = _safe_div(v, vol_s.rolling(50, min_periods=10).median().to_numpy())
        out["amihud_illiq"] = _safe_div(np.abs(log_ret.to_numpy()), v + 1.0)
        out["vwap_dist_z"] = _zscore(pd.Series(_safe_div(c, vwap) - 1, index=df.index), self.z_window)

        # --- Returns / shape ---------------------------------------------
        out["ret_1"] = log_ret
        out["ret_5"] = np.log(close / close.shift(5))
        out["ret_15"] = np.log(close / close.shift(15))
        out["ret_skew_30"] = log_ret.rolling(30, min_periods=10).skew()
        out["ret_kurt_30"] = log_ret.rolling(30, min_periods=10).kurt()
        rng = (h - l)
        out["body_to_range"] = _safe_div(np.abs(c - o), rng)
        out["upper_wick_ratio"] = _safe_div(h - np.maximum(o, c), rng)
        out["lower_wick_ratio"] = _safe_div(np.minimum(o, c) - l, rng)
        out["gap_pct"] = _safe_div(o - np.roll(c, 1), np.roll(c, 1))

        # --- Regime / time -----------------------------------------------
        out["hurst_exponent"] = close.rolling(64, min_periods=20).apply(
            lambda w: _hurst(np.asarray(w)), raw=False
        )
        out["autocorr_lag1"] = log_ret.rolling(30, min_periods=10).apply(
            lambda w: pd.Series(w).autocorr(lag=1), raw=False
        )
        out["regime_vol_bucket"] = pd.qcut(
            rvol.rank(method="first"), 4, labels=False, duplicates="drop"
        ).astype(float)
        if "ts" in df:
            hours = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.hour.fillna(0).to_numpy()
        else:
            hours = np.zeros(len(df))
        out["session_sin"] = np.sin(2 * np.pi * hours / 24)
        out["session_cos"] = np.cos(2 * np.pi * hours / 24)

        feat = pd.DataFrame({k: np.asarray(v_, dtype=float).reshape(-1)[: len(df)] for k, v_ in out.items()}, index=df.index)
        feat = feat.reindex(columns=FEATURE_NAMES)
        feat = feat.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        # --- unsupervised regime one-hot (computed from cleaned base feats) --
        # Done last so the regime model sees the same values train and serve.
        # No model attached -> the regime_* columns stay at their neutral 0.
        if self.regime_model is not None and self.regime_model.is_ready():
            oh = self.regime_model.predict_one_hot(feat)
            for j, name in enumerate(REGIME_ONEHOT_NAMES):
                feat[name] = oh[:, j]
        return feat

    def compute_latest(self, candles: pd.DataFrame) -> np.ndarray:
        """Return the most recent row's feature vector as a 1-D float32 array."""
        frame = self.compute_frame(candles)
        return frame.iloc[-1].to_numpy(dtype=np.float32)
