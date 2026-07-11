"""Canonical multi-exchange candle schema for training pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ae_brain.symbols import extract_base_asset, normalize_symbol
from ae_brain.training.dataframe_utils import (
    first_present_column,
    missing_optional_columns,
    numeric_series,
    numeric_series_fallback,
)

CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "exchange",
    "symbol",
    "base_asset",
    "quote_asset",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "mark_open",
    "mark_high",
    "mark_low",
    "mark_close",
    "index_open",
    "index_high",
    "index_low",
    "index_close",
    "funding_rate",
    "open_interest",
    "spread_estimate",
    "slippage_estimate",
    "fee_rate",
)

OHLC_REQUIRED = ("open", "high", "low", "close")

OPTIONAL_CANONICAL_FIELDS: tuple[str, ...] = (
    "quote_volume",
    "trades_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "mark_open",
    "mark_high",
    "mark_low",
    "mark_close",
    "index_open",
    "index_high",
    "index_low",
    "index_close",
    "funding_rate",
    "open_interest",
    "spread_estimate",
    "slippage_estimate",
)


@dataclass(frozen=True, slots=True)
class CanonicalSchema:
    columns: tuple[str, ...] = CANONICAL_COLUMNS


def _quote_asset(symbol: str) -> str:
    sym = normalize_symbol(symbol)
    base = extract_base_asset(sym)
    if sym.endswith("USDT") and base != sym:
        return "USDT"
    if sym.endswith("USDC") and base != sym:
        return "USDC"
    return "USDT"


def to_canonical_frame(
    df: pd.DataFrame,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    fee_rate: float = 0.0004,
) -> pd.DataFrame:
    """Map a raw/normalized frame into the canonical schema (UTC timestamps)."""
    sym = normalize_symbol(symbol)
    out = pd.DataFrame(index=df.index)
    ts = first_present_column(df, "timestamp", "ts", "open_time")
    out["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce") if ts is not None else pd.NaT
    out["exchange"] = exchange
    out["symbol"] = sym
    out["base_asset"] = extract_base_asset(sym)
    out["quote_asset"] = _quote_asset(sym)
    out["timeframe"] = timeframe
    for col in OHLC_REQUIRED:
        out[col] = numeric_series(df, col, default=float("nan"))
    out["volume"] = numeric_series(df, "volume", default=0.0)
    out["quote_volume"] = numeric_series_fallback(
        df,
        "quote_volume",
        fallback="quote_asset_volume",
        default=0.0,
    )
    if out["quote_volume"].eq(0.0).all() and out["volume"].notna().any() and out["close"].notna().any():
        out["quote_volume"] = (out["volume"] * out["close"]).fillna(0.0)
    out["trades_count"] = numeric_series_fallback(df, "trades_count", fallback="trade_count", default=0.0).astype(
        int
    )
    out["taker_buy_base_volume"] = numeric_series_fallback(
        df, "taker_buy_base_volume", fallback="taker_buy_volume", default=0.0
    )
    out["taker_buy_quote_volume"] = numeric_series(df, "taker_buy_quote_volume", default=0.0)
    close_series = out["close"]
    for prefix, src_prefix in (("mark", "mark"), ("index", "index")):
        for leg in ("open", "high", "low", "close"):
            key = f"{prefix}_{leg}"
            src = f"{src_prefix}_{leg}"
            out[key] = numeric_series_fallback(df, src, fallback=close_series, default=float("nan"))
    out["funding_rate"] = numeric_series(df, "funding_rate", default=0.0)
    out["open_interest"] = numeric_series(df, "open_interest", default=0.0)
    out["spread_estimate"] = numeric_series(df, "spread_estimate", default=0.0)
    out["slippage_estimate"] = numeric_series(df, "slippage_estimate", default=0.0)
    out["fee_rate"] = fee_rate
    return out[list(CANONICAL_COLUMNS)]


def validate_canonical(df: pd.DataFrame) -> list[str]:
    """Return validation errors (empty list = ok)."""
    errors: list[str] = []
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing_columns={missing}")
        return errors
    if df["timestamp"].isna().any():
        errors.append("null_timestamps")
    if df.duplicated(subset=["exchange", "symbol", "timeframe", "timestamp"]).any():
        errors.append("duplicate_candles")
    for col in OHLC_REQUIRED:
        if df[col].isna().any():
            errors.append(f"null_{col}")
    if not df["timestamp"].is_monotonic_increasing:
        errors.append("timestamps_not_sorted")
    return errors


def optional_fields_missing(df: pd.DataFrame) -> list[str]:
    """Optional canonical fields absent or all-null in source frame."""
    return missing_optional_columns(df, OPTIONAL_CANONICAL_FIELDS)


def candles_from_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Strip to the OHLCV(+micro) columns FeatureEngineer expects."""
    out = df.copy()
    out["ts"] = out["timestamp"]
    return out
