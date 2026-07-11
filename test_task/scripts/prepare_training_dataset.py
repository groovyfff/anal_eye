#!/usr/bin/env python3
"""Merge raw downloads into a single canonical multi-asset parquet dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.symbols import DEFAULT_SYMBOL_UNIVERSE, parse_symbol_list
from ae_brain.training.canonical import to_canonical_frame, validate_canonical


def load_raw_klines(raw_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    path = raw_dir / symbol / timeframe / "klines.csv"
    if not path.exists():
        alt = raw_dir.parent / "production" / f"{symbol}_{timeframe}.csv"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns and "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"], utc=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-from-config", action="store_true")
    parser.add_argument("--timeframes", default="1h")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "binance")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "datasets" / "multi_asset.parquet")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "reports")
    args = parser.parse_args()

    symbols = list(DEFAULT_SYMBOL_UNIVERSE) if args.symbols_from_config else parse_symbol_list(args.symbols)
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for sym in symbols:
        for tf in timeframes:
            try:
                raw = load_raw_klines(args.input, sym, tf)
                canon = to_canonical_frame(raw, exchange="binance", symbol=sym, timeframe=tf)
                errs = validate_canonical(canon)
                if errs:
                    errors.append(f"{sym}/{tf}: {errs}")
                frames.append(canon)
            except FileNotFoundError as exc:
                errors.append(str(exc))

    if not frames:
        print("No data loaded. Run download_market_data.py first.", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timeframe", "timestamp"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"wrote {len(out)} rows -> {args.output}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": len(out),
        "symbols": sorted(out["symbol"].unique().tolist()),
        "timeframes": sorted(out["timeframe"].unique().tolist()),
        "errors": errors,
    }
    (args.report_dir / "prepare_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    if errors:
        print("warnings:", errors, file=sys.stderr)


if __name__ == "__main__":
    main()
