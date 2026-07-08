#!/usr/bin/env python3
"""Discover top-200 Binance USDT-M perpetual symbols by 24h quote volume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.universe_top200 import (
    TOP200_SIZE,
    build_top200_universe,
    fetch_binance_futures_universe,
    symbols_csv,
    write_universe_files,
)

# Reuse downloader HTTP helper for consistent FAPI access.
from scripts.download_market_data import _fapi_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover top-200 USDT-M perpetual symbols")
    parser.add_argument("--size", type=int, default=TOP200_SIZE)
    parser.add_argument(
        "--txt-out",
        type=Path,
        default=ROOT / "config" / "universe_top200_usdtm.txt",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "config" / "universe_top200_usdtm.json",
    )
    args = parser.parse_args()

    exchange_info, tickers = fetch_binance_futures_universe(_fapi_json)
    record = build_top200_universe(exchange_info, tickers, target_size=args.size)
    write_universe_files(args.txt_out, args.json_out, record)

    print(f"Wrote {len(record.symbols)} symbols to {args.txt_out}")
    print(f"Wrote metadata to {args.json_out}")
    print(f"Forced includes: {','.join(record.forced_includes)}")
    print(f"SYMBOLS={symbols_csv(record.symbols)}")
    print(json.dumps({"count": len(record.symbols), "symbols": list(record.symbols)}, indent=2))


if __name__ == "__main__":
    main()
