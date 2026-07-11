"""Safe pandas column extraction for data download/conversion pipelines."""

from __future__ import annotations

import pandas as pd


def numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric Series for ``column``, or a constant Series if missing."""
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def numeric_series_fallback(
    df: pd.DataFrame,
    column: str,
    *,
    fallback: str | pd.Series | None = None,
    default: float = 0.0,
) -> pd.Series:
    """Like :func:`numeric_series` but uses ``fallback`` column/Series when absent."""
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    if isinstance(fallback, str) and fallback in df.columns:
        return pd.to_numeric(df[fallback], errors="coerce").fillna(default)
    if isinstance(fallback, pd.Series):
        return pd.to_numeric(fallback, errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def first_present_column(df: pd.DataFrame, *columns: str) -> pd.Series | None:
    """Return the first column that exists, or ``None``."""
    for col in columns:
        if col in df.columns:
            return df[col]
    return None


def missing_optional_columns(df: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> list[str]:
    """Columns that are absent or entirely null (optional fields not populated)."""
    missing: list[str] = []
    for col in columns:
        if col not in df.columns:
            missing.append(col)
        elif df[col].isna().all():
            missing.append(col)
    return missing


def assign_numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.DataFrame:
    """In-place safe assign of a numeric column using :func:`numeric_series`."""
    df[column] = numeric_series(df, column, default=default)
    return df


def ensure_utc_timestamp_column(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """Coerce a column to timezone-aware UTC datetimes (safe before sort/merge_asof)."""
    if column not in df.columns:
        return df
    out = df.copy()
    series = out[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dt, "tz", None) is None:
            out[column] = series.dt.tz_localize("UTC")
        else:
            out[column] = series.dt.tz_convert("UTC")
    else:
        out[column] = pd.to_datetime(series, utc=True, format="ISO8601")
    return out
