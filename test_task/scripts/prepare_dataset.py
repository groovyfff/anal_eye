"""Build a training dataset CSV for the A.E. Brain ensemble.

Data sourcing (in priority order):

1. **Real Binance klines** - fetched from the public REST API (the same source
   ``data/binance_data_exporter.py`` uses), paginated forward from ``--days``
   ago up to now. Converted into the column schema the trainers expect
   (``ts, open, high, low, close, volume, taker_buy_volume, trade_count``).
2. **Synthetic fallback** - if the network is unavailable or too few candles
   come back, fall back to ``generate_synthetic_candles`` (the project's
   built-in plumbing generator) so the pipeline still produces loadable weights
   offline.

Microstructure columns that the REST klines don't carry (open_interest,
funding_rate, basis, liquidations, ...) are intentionally omitted; the
``FeatureEngineer._coerce_series`` null-handling maps them to neutral defaults.

Usage::

    # single symbol (back-compat)
    python scripts/prepare_dataset.py --out data/candles.csv --interval 1h --days 320

    # multi-symbol deep history (production): one CSV per symbol under --outdir
    python scripts/prepare_dataset.py \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1h --days 1095 \
        --outdir data/production
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BINANCE_BASES = ("https://api3.binance.com", "https://api.binance.com", "https://data-api.binance.vision")
KLINES_ENDPOINT = "/api/v3/klines"

# Binance kline array indices.
K_OPEN_TIME, K_OPEN, K_HIGH, K_LOW, K_CLOSE, K_VOLUME = 0, 1, 2, 3, 4, 5
K_NUM_TRADES, K_TAKER_BUY_BASE = 8, 9


def _http_get_json(url: str, timeout: float = 15.0):
    req = urllib.request.Request(url, headers={"User-Agent": "ae-brain-dataset/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_klines(symbol: str, interval: str, days: int, cap: int) -> list[list]:
    """Paginate klines forward from ``days`` ago until now (bounded by ``cap``)."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    last_err: Exception | None = None
    for base in BINANCE_BASES:
        rows: list[list] = []
        cursor = start_ms
        try:
            while len(rows) < cap:
                params = urllib.parse.urlencode(
                    {"symbol": symbol, "interval": interval, "limit": 1000, "startTime": cursor}
                )
                batch = _http_get_json(f"{base}{KLINES_ENDPOINT}?{params}")
                if not batch:
                    break
                rows.extend(batch)
                next_cursor = int(batch[-1][K_OPEN_TIME]) + 1
                if next_cursor <= cursor or next_cursor >= now_ms:
                    break
                cursor = next_cursor
            if rows:
                print(f"[dataset] fetched {len(rows)} real {interval} klines for {symbol} from {base}")
                return rows[:cap]
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_err = exc
            print(f"[dataset] {base} failed: {exc}", file=sys.stderr)
    if last_err is not None:
        print(f"[dataset] all Binance mirrors failed ({last_err})", file=sys.stderr)
    return []


def _klines_to_frame(rows: list[list]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime([int(r[K_OPEN_TIME]) for r in rows], unit="ms", utc=True),
            "open": [float(r[K_OPEN]) for r in rows],
            "high": [float(r[K_HIGH]) for r in rows],
            "low": [float(r[K_LOW]) for r in rows],
            "close": [float(r[K_CLOSE]) for r in rows],
            "volume": [float(r[K_VOLUME]) for r in rows],
            "taker_buy_volume": [float(r[K_TAKER_BUY_BASE]) for r in rows],
            "trade_count": [float(r[K_NUM_TRADES]) for r in rows],
        }
    )


def build_dataset(out: Path, *, symbol: str, interval: str, days: int, cap: int, min_rows: int) -> pd.DataFrame:
    rows = _fetch_klines(symbol, interval, days, cap)
    if len(rows) >= min_rows:
        frame = _klines_to_frame(rows)
        source = "binance-rest"
    else:
        from ae_brain.training.synthetic import generate_synthetic_candles

        print(
            f"[dataset] only {len(rows)} real candles (< {min_rows}); "
            "falling back to synthetic generator",
            file=sys.stderr,
        )
        # Decorrelate per-symbol synthetic fallbacks (deterministic per symbol).
        seed = 7 + (abs(hash(symbol)) % 9973)
        frame = generate_synthetic_candles(n=max(cap, min_rows), seed=seed)
        source = "synthetic"

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[dataset] wrote {len(frame)} candles ({source}) -> {out}")
    return frame


def build_multi(
    symbols: list[str],
    outdir: Path,
    *,
    interval: str,
    days: int,
    cap: int,
    min_rows: int,
) -> dict[str, Path]:
    """Fetch a deep history for each symbol and write one CSV per symbol.

    Returns a ``{symbol: csv_path}`` map. Per-symbol CSVs keep the trainer's
    canonical column schema so the production orchestrator can build leak-free,
    per-symbol datasets (no cross-symbol windows / triple-barrier seams).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for symbol in symbols:
        out = outdir / f"{symbol}_{interval}.csv"
        # A distinct synthetic seed per symbol keeps offline fallbacks decorrelated.
        try:
            build_dataset(
                out, symbol=symbol, interval=interval, days=days, cap=cap, min_rows=min_rows
            )
        except Exception as exc:  # noqa: BLE001 - keep going across symbols
            print(f"[dataset] {symbol} failed: {exc}", file=sys.stderr)
            continue
        written[symbol] = out
    print(f"[dataset] multi-symbol complete: {len(written)}/{len(symbols)} -> {outdir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare A.E. Brain training dataset")
    parser.add_argument("--out", type=Path, default=Path("data/candles.csv"))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols for multi-symbol mode (e.g. BTCUSDT,ETHUSDT,SOLUSDT). "
        "When set, --outdir receives one CSV per symbol and --out/--symbol are ignored.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("data/production"))
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=320)
    parser.add_argument("--cap", type=int, default=8000, help="max candles to keep per symbol")
    parser.add_argument("--min-rows", type=int, default=500, help="min real rows before synthetic fallback")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if symbols:
        build_multi(
            symbols,
            args.outdir,
            interval=args.interval,
            days=args.days,
            cap=args.cap,
            min_rows=args.min_rows,
        )
        return

    build_dataset(
        args.out,
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        cap=args.cap,
        min_rows=args.min_rows,
    )


if __name__ == "__main__":
    main()
