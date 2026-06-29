"""Canonical feature schema (~60 numerical features).

This module is the single source of truth for the tabular feature contract.
Both the trainer and the live inference path import ``FEATURE_NAMES`` so that
column ordering can never drift between train and serve (a classic source of
silent model degradation).

Each feature carries a short, human-readable description and a ``group`` so we
can reason about feature families (volatility, flow, momentum, ...).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    group: str
    description: str


# ---------------------------------------------------------------------------
# The schema is intentionally explicit (no auto-generation) so reviewers can
# audit exactly what the model sees. ~60 features across 7 families.
# ---------------------------------------------------------------------------
FEATURE_SCHEMA: tuple[FeatureSpec, ...] = (
    # --- Volatility / dispersion -------------------------------------------
    FeatureSpec("vol_z", "volatility", "Z-score of realized volatility vs rolling mean"),
    FeatureSpec("atr_14", "volatility", "Average True Range, 14 candles"),
    FeatureSpec("atr_pct", "volatility", "ATR as a fraction of close price"),
    FeatureSpec("bb_width", "volatility", "Bollinger band width (upper-lower)/mid"),
    FeatureSpec("bb_pctb", "volatility", "%B position of close within Bollinger bands"),
    FeatureSpec("realized_vol_30", "volatility", "30-candle realized volatility (std of log ret)"),
    FeatureSpec("parkinson_vol", "volatility", "Parkinson high-low range volatility estimator"),
    FeatureSpec("vol_of_vol", "volatility", "Std of rolling volatility (vol-of-vol)"),
    # --- Momentum / trend ---------------------------------------------------
    FeatureSpec("rsi_14", "momentum", "Relative Strength Index, 14"),
    FeatureSpec("rsi_slope", "momentum", "First difference of RSI (momentum of momentum)"),
    FeatureSpec("macd", "momentum", "MACD line (12,26)"),
    FeatureSpec("macd_signal", "momentum", "MACD signal line (9 EMA of MACD)"),
    FeatureSpec("macd_hist", "momentum", "MACD histogram (line - signal)"),
    FeatureSpec("adx_14", "momentum", "Average Directional Index trend strength"),
    FeatureSpec("plus_di", "momentum", "+DI directional indicator"),
    FeatureSpec("minus_di", "momentum", "-DI directional indicator"),
    FeatureSpec("ema_fast_slow_ratio", "momentum", "EMA(12)/EMA(26) - 1 trend tilt"),
    FeatureSpec("price_vs_vwap", "momentum", "Close / rolling VWAP - 1"),
    FeatureSpec("roc_10", "momentum", "Rate of change over 10 candles"),
    FeatureSpec("stoch_k", "momentum", "Stochastic oscillator %K"),
    FeatureSpec("stoch_d", "momentum", "Stochastic oscillator %D"),
    FeatureSpec("cci_20", "momentum", "Commodity Channel Index, 20"),
    FeatureSpec("willr_14", "momentum", "Williams %R, 14"),
    # --- Order flow / microstructure ---------------------------------------
    FeatureSpec("cvd", "flow", "Cumulative Volume Delta (signed taker volume)"),
    FeatureSpec("cvd_z", "flow", "Z-score of CVD vs rolling window"),
    FeatureSpec("cvd_slope", "flow", "Slope of CVD (flow acceleration)"),
    FeatureSpec("ofi_delta", "flow", "Order-Flow Imbalance delta (bid-ask pressure)"),
    FeatureSpec("ofi_z", "flow", "Z-score of order-flow imbalance"),
    FeatureSpec("taker_buy_ratio", "flow", "Taker buy volume / total volume"),
    FeatureSpec("aggressor_imbalance", "flow", "(buy-sell)/(buy+sell) aggressor volume"),
    FeatureSpec("trade_count_z", "flow", "Z-score of trade count (activity spike)"),
    FeatureSpec("large_trade_ratio", "flow", "Share of notional from large prints"),
    FeatureSpec("book_imbalance", "flow", "Top-of-book size imbalance (bid-ask)/(bid+ask)"),
    FeatureSpec("spread_bps", "flow", "Quoted spread in basis points"),
    # --- Open interest / derivatives ---------------------------------------
    FeatureSpec("oi_z", "derivatives", "Z-score of open interest"),
    FeatureSpec("oi_change_pct", "derivatives", "Pct change of open interest"),
    FeatureSpec("oi_price_divergence", "derivatives", "Sign(OI change) vs sign(price change)"),
    FeatureSpec("funding_rate", "derivatives", "Current perpetual funding rate"),
    FeatureSpec("funding_z", "derivatives", "Z-score of funding rate"),
    FeatureSpec("basis_pct", "derivatives", "Perp-spot basis as a fraction"),
    FeatureSpec("liq_long_notional", "derivatives", "Recent long liquidation notional (log)"),
    FeatureSpec("liq_short_notional", "derivatives", "Recent short liquidation notional (log)"),
    FeatureSpec("liq_imbalance", "derivatives", "(long_liq-short_liq) normalized"),
    # --- Volume / liquidity -------------------------------------------------
    FeatureSpec("volume_z", "volume", "Z-score of volume vs rolling mean"),
    FeatureSpec("rel_volume", "volume", "Volume / rolling median volume"),
    FeatureSpec("obv_slope", "volume", "Slope of On-Balance Volume"),
    FeatureSpec("mfi_14", "volume", "Money Flow Index, 14"),
    FeatureSpec("amihud_illiq", "volume", "Amihud illiquidity (|ret|/volume)"),
    FeatureSpec("vwap_dist_z", "volume", "Z-score of distance from VWAP"),
    # --- Return / shape -----------------------------------------------------
    FeatureSpec("ret_1", "returns", "1-candle log return"),
    FeatureSpec("ret_5", "returns", "5-candle log return"),
    FeatureSpec("ret_15", "returns", "15-candle log return"),
    FeatureSpec("ret_skew_30", "returns", "Skewness of 30-candle returns"),
    FeatureSpec("ret_kurt_30", "returns", "Kurtosis of 30-candle returns"),
    FeatureSpec("body_to_range", "returns", "Candle body / high-low range"),
    FeatureSpec("upper_wick_ratio", "returns", "Upper wick / range"),
    FeatureSpec("lower_wick_ratio", "returns", "Lower wick / range"),
    FeatureSpec("gap_pct", "returns", "Open vs previous close gap pct"),
    # --- Regime / time ------------------------------------------------------
    FeatureSpec("hurst_exponent", "regime", "Rolling Hurst exponent (trend vs mean-rev)"),
    FeatureSpec("autocorr_lag1", "regime", "Lag-1 autocorrelation of returns"),
    FeatureSpec("regime_vol_bucket", "regime", "Discretized volatility regime (0..3)"),
    FeatureSpec("session_sin", "regime", "Sine encoding of UTC hour (intraday seasonality)"),
    FeatureSpec("session_cos", "regime", "Cosine encoding of UTC hour"),
    # --- Unsupervised market regime (one-hot from RegimeModel/GMM) ----------
    # Computed by FeatureEngineer when a fitted RegimeModel is attached; absent
    # model -> all-zero (neutral) so the contract/dimension never changes.
    FeatureSpec("regime_trend", "regime", "GMM regime one-hot: low-vol directional (trend)"),
    FeatureSpec("regime_chop", "regime", "GMM regime one-hot: low-vol mean-reverting (chop)"),
    FeatureSpec("regime_highvol", "regime", "GMM regime one-hot: high realized volatility"),
)

# Names of the regime one-hot columns, in canonical order (regime 0,1,2).
REGIME_ONEHOT_NAMES: tuple[str, ...] = ("regime_trend", "regime_chop", "regime_highvol")

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SCHEMA)

# Group -> ordered feature names (handy for ablation + monitoring dashboards).
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {}
for _spec in FEATURE_SCHEMA:
    FEATURE_GROUPS.setdefault(_spec.group, ())
    FEATURE_GROUPS[_spec.group] = (*FEATURE_GROUPS[_spec.group], _spec.name)


def n_features() -> int:
    """Return the number of features in the canonical schema."""
    return len(FEATURE_NAMES)


def assert_feature_order(names: list[str]) -> None:
    """Raise if a provided column ordering does not match the canonical schema."""
    if tuple(names) != FEATURE_NAMES:
        missing = set(FEATURE_NAMES) - set(names)
        extra = set(names) - set(FEATURE_NAMES)
        raise ValueError(
            "Feature contract violation. "
            f"missing={sorted(missing)} extra={sorted(extra)} "
            f"(expected {len(FEATURE_NAMES)} ordered features)."
        )
