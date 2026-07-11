"""Tests for download_market_data date parsing and timestamp normalization."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "download_market_data", ROOT / "scripts" / "download_market_data.py"
)
assert _SPEC and _SPEC.loader
dm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = dm
_SPEC.loader.exec_module(dm)


def test_parse_cli_time_range_start_string_end_now() -> None:
    tr = dm.parse_cli_time_range("2021-01-01", "now")
    assert tr.start_utc.tz is not None
    assert tr.end_utc.tz is not None
    assert tr.start_ms < tr.end_ms
    assert isinstance(tr.start_ms, int)
    assert isinstance(tr.end_ms, int)


def test_parse_cli_time_range_explicit_end_string() -> None:
    tr = dm.parse_cli_time_range("2021-01-01", "2026-07-01")
    assert tr.start_utc == pd.Timestamp("2021-01-01", tz="UTC")
    assert tr.end_utc == pd.Timestamp("2026-07-01", tz="UTC")
    assert tr.start_ms < tr.end_ms


def test_merge_asof_no_timestamp_string_comparison() -> None:
    left = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"],
            "close": [100.0, 101.0],
        }
    )
    right = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "funding_rate": [0.0001],
        }
    )
    merged = dm.merge_asof_backward(left, right)
    assert pd.api.types.is_datetime64_any_dtype(merged["timestamp"])
    assert len(merged) == 2


def test_concat_existing_csv_with_fetched_frames_no_type_error(tmp_path: Path) -> None:
    out_dir = tmp_path / "binance"
    sym_dir = out_dir / "BTCUSDT" / "1h"
    sym_dir.mkdir(parents=True)
    out_path = sym_dir / "klines.csv"
    existing_ms = 1717200000000
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(existing_ms, unit="ms", tz="UTC").isoformat()],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
            "trades_count": [1],
        }
    ).to_csv(out_path, index=False)

    new_ms = existing_ms + 3_600_000
    kline_row = [
        new_ms,
        "2",
        "3",
        "1",
        "2.5",
        "20",
        new_ms + 3_599_999,
        "40",
        5,
        "10",
        "20",
        0,
    ]
    calls: dict[str, int] = {}

    def fake_fapi(path: str, params: dict | None = None) -> object:
        calls[path] = calls.get(path, 0) + 1
        if calls[path] > 1:
            return []
        if path == "/fapi/v1/klines":
            return [kline_row]
        if path == "/fapi/v1/fundingRate":
            return [{"fundingTime": new_ms, "fundingRate": "0.0001"}]
        if path == "/fapi/v1/markPriceKlines":
            return [[new_ms, "2", "3", "1", "2.5", "0", 0, "0", 0, "0", "0", 0]]
        if path == "/fapi/v1/indexPriceKlines":
            return [[new_ms, "2", "3", "1", "2.5", "0", 0, "0", 0, "0", "0", 0]]
        if path == "/futures/data/openInterestHist":
            return [{"timestamp": new_ms, "sumOpenInterest": "123.45"}]
        raise AssertionError(path)

    time_range = dm.TimeRange(
        start_ms=existing_ms,
        end_ms=new_ms + 3_600_000,
        start_utc=pd.Timestamp(existing_ms, unit="ms", tz="UTC"),
        end_utc=pd.Timestamp(new_ms + 3_600_000, unit="ms", tz="UTC"),
    )

    with patch.object(dm, "_fapi_json", side_effect=fake_fapi), patch.object(dm.time, "sleep"):
        result = dm.download_symbol(
            "BTCUSDT",
            "1h",
            out_dir,
            time_range=time_range,
            include_funding=True,
            include_mark=True,
            include_index=True,
            include_oi=True,
        )

    assert result == out_path
    saved = pd.read_csv(out_path)
    saved["timestamp"] = pd.to_datetime(saved["timestamp"], utc=True)
    assert len(saved) >= 2
    assert saved["timestamp"].is_monotonic_increasing


def test_parse_cli_time_range_rejects_inverted_window() -> None:
    with pytest.raises(ValueError, match="start must be before end"):
        dm.parse_cli_time_range("2026-01-01", "2021-01-01")
