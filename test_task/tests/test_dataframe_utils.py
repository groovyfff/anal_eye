"""Tests for safe dataframe column helpers and canonical conversion."""

from __future__ import annotations

import pandas as pd

from ae_brain.training.canonical import optional_fields_missing, to_canonical_frame, validate_canonical
from ae_brain.training.dataframe_utils import (
    assign_numeric_column,
    ensure_utc_timestamp_column,
    missing_optional_columns,
    numeric_series,
    numeric_series_fallback,
)


def test_numeric_series_missing_column_returns_series() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0]})
    s = numeric_series(df, "open_interest", default=0.0)
    assert isinstance(s, pd.Series)
    assert len(s) == 2
    assert s.tolist() == [0.0, 0.0]


def test_numeric_series_existing_column_coerces_and_fills() -> None:
    df = pd.DataFrame({"open_interest": ["1.5", None]})
    s = numeric_series(df, "open_interest", default=0.0)
    assert s.tolist() == [1.5, 0.0]


def test_assign_numeric_column_on_missing_does_not_crash() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")})
    assign_numeric_column(df, "open_interest", default=0.0)
    assert "open_interest" in df.columns
    assert df["open_interest"].tolist() == [0.0, 0.0, 0.0]


def test_merged_get_scalar_fillna_bug_regression() -> None:
    """Regression: DataFrame.get(col, 0.0) returns float when col missing."""
    merged = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "close": [100.0, 101.0],
        }
    )
    assign_numeric_column(merged, "open_interest", default=0.0)
    assert merged["open_interest"].fillna(0.0).tolist() == [0.0, 0.0]


def test_to_canonical_frame_minimal_ohlcv_only() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 11, 12, 13, 14],
        }
    )
    canon = to_canonical_frame(df, exchange="binance", symbol="BTCUSDT", timeframe="1h")
    assert validate_canonical(canon) == []
    assert canon["open_interest"].tolist() == [0.0] * 5
    assert canon["funding_rate"].tolist() == [0.0] * 5
    missing = optional_fields_missing(df)
    assert "open_interest" in missing
    assert "funding_rate" in missing


def test_to_canonical_without_optional_columns_no_attribute_error() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [100, 200, 300],
        }
    )
    canon = to_canonical_frame(df, exchange="binance", symbol="ETHUSDT", timeframe="1h")
    assert len(canon) == 3
    assert canon["trades_count"].tolist() == [0, 0, 0]


def test_numeric_series_fallback_uses_fallback_column() -> None:
    df = pd.DataFrame({"trade_count": [5, 10]})
    s = numeric_series_fallback(df, "trades_count", fallback="trade_count", default=0.0)
    assert s.tolist() == [5.0, 10.0]


def test_missing_optional_columns_detects_absent_and_all_null() -> None:
    df = pd.DataFrame({"funding_rate": [None, None], "open_interest": [1.0, 2.0]})
    missing = missing_optional_columns(df, ["funding_rate", "open_interest", "mark_open"])
    assert "funding_rate" in missing
    assert "mark_open" in missing
    assert "open_interest" not in missing


def test_ensure_utc_timestamp_column_mixed_iso_strings() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2021-01-01 00:00:00.000000+00:00",
                "2021-01-02 00:00:00+00:00",
                "2024-06-01T00:00:00+00:00",
            ]
        }
    )
    out = ensure_utc_timestamp_column(df)
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert len(out) == 3
