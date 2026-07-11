#!/usr/bin/env python3
"""Download multi-symbol market data from Binance Futures (and optional mirrors).

Usage::

    python scripts/download_market_data.py \\
        --exchange binance \\
        --symbols-from-config \\
        --timeframes 5m,15m,1h \\
        --start 2021-01-01 \\
        --end now \\
        --include-funding \\
        --include-mark-price \\
        --include-index-price \\
        --include-open-interest

Data layout::

    test_task/data/raw/binance/{symbol}/{timeframe}/klines.csv
    test_task/data/raw/binance/{symbol}/funding.csv
    ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE, parse_symbol_list
from ae_brain.training.dataframe_utils import (
    assign_numeric_column,
    ensure_utc_timestamp_column,
    missing_optional_columns,
    numeric_series,
)

log = logging.getLogger("ae_brain.download")

FAPI_BASES = (
    "https://fapi.binance.com",
    "https://fapi.binancefuture.com",
)
LIMIT = 1500
SLEEP_SEC = 0.25


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Normalized CLI time window — milliseconds for API, UTC timestamps for logs."""

    start_ms: int
    end_ms: int
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp


def parse_utc_timestamp(value: str | None) -> pd.Timestamp:
    """Parse CLI date or ``now`` to a UTC-aware :class:`pd.Timestamp`."""
    if value is None or not str(value).strip():
        raise ValueError("timestamp value must not be empty")
    if str(value).strip().lower() == "now":
        return pd.Timestamp.now(tz="UTC")
    ts = pd.to_datetime(value, utc=True)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def timestamp_to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def parse_cli_time_range(start: str, end: str) -> TimeRange:
    """Parse ``--start`` / ``--end`` once; internal API uses integer milliseconds."""
    start_utc = parse_utc_timestamp(start)
    end_utc = parse_utc_timestamp(end)
    if start_utc >= end_utc:
        raise ValueError(f"start must be before end: {start_utc.isoformat()} >= {end_utc.isoformat()}")
    return TimeRange(
        start_ms=timestamp_to_ms(start_utc),
        end_ms=timestamp_to_ms(end_utc),
        start_utc=start_utc,
        end_utc=end_utc,
    )


def _read_csv_timestamps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_utc_timestamp_column(df, "timestamp")


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ae-brain-downloader/1.0"})
    last_err: Exception | None = None
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            time.sleep(SLEEP_SEC * 2)
    raise RuntimeError(f"GET failed after retries: {url} ({last_err})")


def _fapi_json(path: str, params: dict | None = None) -> object:
    qs = urllib.parse.urlencode(params or {})
    last_err: Exception | None = None
    for base in FAPI_BASES:
        url = f"{base}{path}?{qs}" if qs else f"{base}{path}"
        try:
            return json.loads(_http_get(url).decode())
        except Exception as exc:
            last_err = exc
            time.sleep(SLEEP_SEC)
    raise RuntimeError(f"Binance FAPI failed for {path}: {last_err}")


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk = _fapi_json(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": LIMIT,
            },
        )
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        last_open = int(chunk[-1][0])
        if last_open <= cursor:
            break
        cursor = last_open + 1
        time.sleep(SLEEP_SEC)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume"):
        df[col] = numeric_series(df, col, default=float("nan"))
    df["trades_count"] = numeric_series(df, "trades_count", default=0.0).astype(int)
    return ensure_utc_timestamp_column(
        df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    )


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk = _fapi_json(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        cursor = int(chunk[-1]["fundingTime"]) + 1
        time.sleep(SLEEP_SEC)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = numeric_series(df, "fundingRate", default=float("nan"))
    return ensure_utc_timestamp_column(
        df[["timestamp", "funding_rate"]].drop_duplicates(subset=["timestamp"])
    )


def fetch_mark_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk = _fapi_json(
            "/fapi/v1/markPriceKlines",
            {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": LIMIT},
        )
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        cursor = int(chunk[-1][0]) + 1
        time.sleep(SLEEP_SEC)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df[0], unit="ms", utc=True)
    df["mark_open"] = pd.to_numeric(df[1], errors="coerce")
    df["mark_high"] = pd.to_numeric(df[2], errors="coerce")
    df["mark_low"] = pd.to_numeric(df[3], errors="coerce")
    df["mark_close"] = pd.to_numeric(df[4], errors="coerce")
    return ensure_utc_timestamp_column(
        df[["timestamp", "mark_open", "mark_high", "mark_low", "mark_close"]].drop_duplicates(subset=["timestamp"])
    )


def fetch_index_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    pair = symbol.replace("USDT", "USD") if symbol.endswith("USDT") else symbol
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        try:
            chunk = _fapi_json(
                "/fapi/v1/indexPriceKlines",
                {"pair": pair, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": LIMIT},
            )
        except RuntimeError:
            return pd.DataFrame()
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        cursor = int(chunk[-1][0]) + 1
        time.sleep(SLEEP_SEC)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df[0], unit="ms", utc=True)
    df["index_open"] = pd.to_numeric(df[1], errors="coerce")
    df["index_high"] = pd.to_numeric(df[2], errors="coerce")
    df["index_low"] = pd.to_numeric(df[3], errors="coerce")
    df["index_close"] = pd.to_numeric(df[4], errors="coerce")
    return ensure_utc_timestamp_column(
        df[["timestamp", "index_open", "index_high", "index_low", "index_close"]].drop_duplicates(
            subset=["timestamp"]
        )
    )


def fetch_open_interest(symbol: str, period: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        try:
            chunk = _fapi_json(
                "/futures/data/openInterestHist",
                {"symbol": symbol, "period": period, "startTime": cursor, "endTime": end_ms, "limit": 500},
            )
        except RuntimeError:
            return pd.DataFrame()
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        cursor = int(chunk[-1]["timestamp"]) + 1
        time.sleep(SLEEP_SEC)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open_interest"] = numeric_series(df, "sumOpenInterest", default=float("nan"))
    return ensure_utc_timestamp_column(
        df[["timestamp", "open_interest"]].drop_duplicates(subset=["timestamp"])
    )


def merge_asof_backward(left: pd.DataFrame, right: pd.DataFrame, on: str = "timestamp") -> pd.DataFrame:
    if right.empty:
        return ensure_utc_timestamp_column(left, on)
    left = ensure_utc_timestamp_column(left, on).sort_values(on)
    right = ensure_utc_timestamp_column(right, on).sort_values(on)
    return pd.merge_asof(left, right, on=on, direction="backward")


def download_symbol(
    symbol: str,
    timeframe: str,
    out_dir: Path,
    *,
    time_range: TimeRange,
    include_funding: bool,
    include_mark: bool,
    include_index: bool,
    include_oi: bool,
) -> Path:
    sym_dir = out_dir / symbol / timeframe
    sym_dir.mkdir(parents=True, exist_ok=True)
    out_path = sym_dir / "klines.csv"
    start_ms = time_range.start_ms
    end_ms = time_range.end_ms
    if out_path.exists():
        existing = _read_csv_timestamps(out_path)
        if len(existing) > 0:
            last_ts = timestamp_to_ms(existing["timestamp"].iloc[-1])
            start_ms = max(start_ms, last_ts + 1)
    log.info(
        "download.begin symbol=%s timeframe=%s start_utc=%s end_utc=%s effective_start_ms=%s end_ms=%s",
        symbol,
        timeframe,
        time_range.start_utc.isoformat(),
        time_range.end_utc.isoformat(),
        start_ms,
        end_ms,
    )
    if start_ms >= end_ms:
        log.info(
            "download.up_to_date symbol=%s timeframe=%s path=%s",
            symbol,
            timeframe,
            out_path,
        )
        return out_path
    klines = fetch_klines(symbol, timeframe, start_ms, end_ms)
    if klines.empty:
        raise RuntimeError(f"No klines returned for {symbol} {timeframe} — check symbol or network")
    merged = klines.copy()
    requested_optional: list[str] = []

    if include_funding:
        requested_optional.append("funding_rate")
        fund = fetch_funding(symbol, start_ms, end_ms)
        fund_path = out_dir / symbol / "funding.csv"
        if fund_path.exists() and not fund.empty:
            fund = pd.concat([_read_csv_timestamps(fund_path), fund]).drop_duplicates(subset=["timestamp"])
        elif fund_path.exists():
            fund = _read_csv_timestamps(fund_path)
        fund = ensure_utc_timestamp_column(fund, "timestamp")
        if not fund.empty:
            fund.to_csv(fund_path, index=False)
        merged = merge_asof_backward(merged, fund)
        assign_numeric_column(merged, "funding_rate", default=0.0)

    if include_mark:
        requested_optional.extend(["mark_open", "mark_high", "mark_low", "mark_close"])
        mark = fetch_mark_klines(symbol, timeframe, start_ms, end_ms)
        merged = merge_asof_backward(merged, mark)

    if include_index:
        requested_optional.extend(["index_open", "index_high", "index_low", "index_close"])
        idx = fetch_index_klines(symbol, timeframe, start_ms, end_ms)
        merged = merge_asof_backward(merged, idx)

    if include_oi:
        requested_optional.append("open_interest")
        oi = fetch_open_interest(symbol, timeframe, start_ms, end_ms)
        merged = merge_asof_backward(merged, oi)
        assign_numeric_column(merged, "open_interest", default=0.0)

    if out_path.exists():
        merged = pd.concat([_read_csv_timestamps(out_path), ensure_utc_timestamp_column(merged)])
        merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    optional_missing = missing_optional_columns(merged, requested_optional) if requested_optional else []

    merged.to_csv(out_path, index=False)
    log.info(
        "download.saved symbol=%s timeframe=%s rows=%s optional_missing=%s path=%s",
        symbol,
        timeframe,
        len(merged),
        optional_missing or "none",
        out_path,
    )
    print(
        f"saved symbol={symbol} timeframe={timeframe} rows={len(merged)} "
        f"optional_missing={optional_missing or 'none'} -> {out_path}"
    )
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Download multi-asset Binance Futures data")
    parser.add_argument("--exchange", default="binance", choices=["binance"])
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-from-config", action="store_true")
    parser.add_argument("--timeframes", default="1h")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="now")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw" / "binance")
    parser.add_argument("--include-funding", action="store_true")
    parser.add_argument("--include-mark-price", action="store_true")
    parser.add_argument("--include-index-price", action="store_true")
    parser.add_argument("--include-open-interest", action="store_true")
    args = parser.parse_args()

    if args.symbols_from_config:
        symbols = list(DEFAULT_SYMBOL_UNIVERSE)
    else:
        symbols = parse_symbol_list(args.symbols) if args.symbols else list(DEFAULT_SYMBOL_UNIVERSE)
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    try:
        time_range = parse_cli_time_range(args.start, args.end)
    except ValueError as exc:
        log.error("download.invalid_time_range %s", exc)
        print(f"Invalid time range: {exc}", file=sys.stderr)
        sys.exit(2)
    log.info(
        "download.range start_utc=%s end_utc=%s start_ms=%s end_ms=%s",
        time_range.start_utc.isoformat(),
        time_range.end_utc.isoformat(),
        time_range.start_ms,
        time_range.end_ms,
    )
    failures: list[str] = []

    for sym in symbols:
        for tf in timeframes:
            try:
                download_symbol(
                    sym,
                    tf,
                    args.output,
                    time_range=time_range,
                    include_funding=args.include_funding,
                    include_mark=args.include_mark_price,
                    include_index=args.include_index_price,
                    include_oi=args.include_open_interest,
                )
            except KeyboardInterrupt:
                log.warning("download.interrupted symbol=%s timeframe=%s", sym, tf)
                print("\nDownload interrupted by user.", file=sys.stderr)
                sys.exit(130)
            except Exception as exc:
                msg = f"{sym}/{tf}: {exc}"
                failures.append(msg)
                log.error("download.failed %s", msg)
                print(f"FAILED {msg}", file=sys.stderr)

    if failures:
        print("\nSome downloads failed. Retry with stable network:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
