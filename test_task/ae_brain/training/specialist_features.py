"""Extended feature vector for side specialists (meta stack + tabular + context)."""

from __future__ import annotations

import numpy as np

from ae_brain.features.schema import FEATURE_NAMES
from ae_brain.layers.meta import META_INPUT_NAMES, build_meta_features

# Tabular features that help side-specific EV / microstructure signal.
SPECIALIST_TABULAR_NAMES: tuple[str, ...] = (
    "funding_rate",
    "funding_z",
    "oi_z",
    "oi_change_pct",
    "basis_pct",
    "adx_14",
    "rsi_14",
    "macd_hist",
    "ema_fast_slow_ratio",
    "ret_15",
    "vol_z",
    "realized_vol_30",
    "volume_z",
    "rel_volume",
    "cvd_z",
    "taker_buy_ratio",
    "hurst_exponent",
    "vwap_dist_z",
    "liq_imbalance",
    "dist_roll_high_atr",
    "dist_roll_low_atr",
    "volume_impulse",
    "btc_ret_15",
    "btc_vol_z",
    "btc_regime_trend",
)

SPECIALIST_FEATURE_NAMES: tuple[str, ...] = tuple(META_INPUT_NAMES) + SPECIALIST_TABULAR_NAMES + (
    "symbol_liquidity_bucket",
)


def _feat_idx(name: str) -> int:
    return FEATURE_NAMES.index(name)


def _roll_dist(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray, win: int, side: str) -> np.ndarray:
    import pandas as pd

    s = pd.Series(close)
    if side == "high":
        ref = s.rolling(win, min_periods=1).max().to_numpy()
        dist = (close - ref) / np.maximum(atr, close * 1e-6)
    else:
        ref = s.rolling(win, min_periods=1).min().to_numpy()
        dist = (ref - close) / np.maximum(atr, close * 1e-6)
    return np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)


def augment_symbol_frame(feats: np.ndarray, prices: np.ndarray, atr: np.ndarray, high: np.ndarray, low: np.ndarray) -> dict[str, np.ndarray]:
    """Precompute extra specialist columns aligned with feature rows."""
    vol_z = feats[:, _feat_idx("vol_z")]
    volume_z = feats[:, _feat_idx("volume_z")]
    rel_vol = feats[:, _feat_idx("rel_volume")]
    return {
        "dist_roll_high_atr": _roll_dist(high, low, prices, atr, 100, "high"),
        "dist_roll_low_atr": _roll_dist(high, low, prices, atr, 100, "low"),
        "volume_impulse": vol_z * rel_vol,
    }


def liquidity_bucket(symbol: str) -> float:
    """Coarse liquidity tier encoding for specialist models."""
    major = {"BTCUSDT", "ETHUSDT"}
    mid = {"SOLUSDT", "BNBUSDT", "XRPUSDT"}
    if symbol in major:
        return 1.0
    if symbol in mid:
        return 0.5
    return 0.0


def build_specialist_features_inference(
    *,
    tab_p_up: float,
    seq_p_cont: float,
    seq_trend: float,
    rl_expo: float,
    regime_oh: np.ndarray | list[float],
    tabular_row: np.ndarray,
    symbol: str,
    extra: dict[str, float] | None = None,
    btc_ctx: dict[str, float] | None = None,
    layer_mask: dict[str, bool] | None = None,
) -> np.ndarray:
    """Single-row specialist vector for live inference (matches training schema)."""
    meta = build_meta_features(
        tab_p_up, seq_p_cont, seq_trend, rl_expo, regime_oh, layer_mask=layer_mask
    )
    extra = extra or {}
    btc_ctx = btc_ctx or {}
    tab_parts = []
    for name in SPECIALIST_TABULAR_NAMES:
        if name in extra:
            tab_parts.append(float(extra[name]))
        elif name.startswith("btc_"):
            tab_parts.append(float(btc_ctx.get(name, 0.0)))
        else:
            tab_parts.append(float(tabular_row[_feat_idx(name)]))
    return np.asarray(list(meta) + tab_parts + [liquidity_bucket(symbol)], dtype=np.float32)


def build_specialist_feature_row(
    *,
    tab_p_up: float,
    seq_p_cont: float,
    seq_trend: float,
    rl_expo: float,
    regime_oh: np.ndarray,
    tabular_row: np.ndarray,
    extra: dict[str, np.ndarray],
    i: int,
    symbol: str,
    btc_ctx: dict[str, np.ndarray] | None,
    layer_mask: dict[str, bool] | None = None,
) -> np.ndarray:
    meta = build_meta_features(
        tab_p_up, seq_p_cont, seq_trend, rl_expo, regime_oh, layer_mask=layer_mask
    )
    tab_parts = []
    for name in SPECIALIST_TABULAR_NAMES:
        if name in extra:
            tab_parts.append(float(extra[name][i]))
        elif name.startswith("btc_") and btc_ctx is not None:
            tab_parts.append(float(btc_ctx[name][i]))
        else:
            tab_parts.append(float(tabular_row[_feat_idx(name)]))
    parts = list(meta) + tab_parts + [liquidity_bucket(symbol)]
    return np.asarray(parts, dtype=np.float32)
