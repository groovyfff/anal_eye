"""Top-200 Binance USDT-M perpetual universe discovery and file helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LEGACY_SIX_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
)

TOP200_SIZE = 200


@dataclass(frozen=True, slots=True)
class UniverseRecord:
    symbols: tuple[str, ...]
    forced_includes: tuple[str, ...]
    ranked_by_quote_volume: list[dict[str, Any]]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.symbols),
            "symbols": list(self.symbols),
            "forced_includes": list(self.forced_includes),
            "ranked_by_quote_volume": self.ranked_by_quote_volume,
        }


def normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def filter_trading_perpetual_usdt(
    exchange_info: dict[str, Any],
    *,
    quote_asset: str = "USDT",
    contract_type: str = "PERPETUAL",
    status: str = "TRADING",
) -> set[str]:
    """Return symbols that pass Binance Futures exchangeInfo filters."""
    allowed: set[str] = set()
    for item in exchange_info.get("symbols") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("quoteAsset", "")).upper() != quote_asset.upper():
            continue
        if str(item.get("contractType", "")).upper() != contract_type.upper():
            continue
        if str(item.get("status", "")).upper() != status.upper():
            continue
        sym = normalize_symbol(item.get("symbol"))
        if sym:
            allowed.add(sym)
    return allowed


def _ticker_quote_volume(ticker: dict[str, Any]) -> float:
    try:
        return float(ticker.get("quoteVolume") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_top200_universe(
    exchange_info: dict[str, Any],
    tickers_24h: list[dict[str, Any]],
    *,
    target_size: int = TOP200_SIZE,
    force_include: Iterable[str] = LEGACY_SIX_SYMBOLS,
) -> UniverseRecord:
    """Build ranked top-N universe with forced legacy symbols included."""
    allowed = filter_trading_perpetual_usdt(exchange_info)
    forced = tuple(normalize_symbol(s) for s in force_include)

    ranked_rows: list[dict[str, Any]] = []
    for row in tickers_24h:
        if not isinstance(row, dict):
            continue
        sym = normalize_symbol(row.get("symbol"))
        if sym not in allowed:
            continue
        ranked_rows.append(
            {
                "symbol": sym,
                "quoteVolume": _ticker_quote_volume(row),
                "lastPrice": row.get("lastPrice"),
                "priceChangePercent": row.get("priceChangePercent"),
            }
        )
    ranked_rows.sort(key=lambda r: r["quoteVolume"], reverse=True)

    selected: list[str] = []
    selected_set: set[str] = set()

    for sym in forced:
        if sym in allowed and sym not in selected_set:
            selected.append(sym)
            selected_set.add(sym)

    for row in ranked_rows:
        sym = row["symbol"]
        if sym in selected_set:
            continue
        selected.append(sym)
        selected_set.add(sym)
        if len(selected) >= target_size:
            break

    if len(selected) < target_size:
        raise ValueError(f"only {len(selected)} eligible symbols found; need {target_size}")

    return UniverseRecord(
        symbols=tuple(selected[:target_size]),
        forced_includes=forced,
        ranked_by_quote_volume=ranked_rows,
    )


def load_universe_txt(path: Path) -> list[str]:
    """Load symbols from txt (one per line or comma-separated)."""
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if "," in raw.splitlines()[0]:
        return [normalize_symbol(part) for part in raw.replace("\n", ",").split(",") if part.strip()]
    symbols: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        symbols.append(normalize_symbol(line))
    return symbols


def write_universe_files(
    txt_path: Path,
    json_path: Path,
    record: UniverseRecord,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(record.symbols) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(record.to_json_dict(), indent=2), encoding="utf-8")


def symbols_csv(symbols: Iterable[str]) -> str:
    return ",".join(normalize_symbol(s) for s in symbols)


def fetch_binance_futures_universe(http_get_json) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fetch exchangeInfo + 24hr tickers using injected JSON GET callable."""
    exchange_info = http_get_json("/fapi/v1/exchangeInfo")
    tickers = http_get_json("/fapi/v1/ticker/24hr")
    if not isinstance(exchange_info, dict):
        raise ValueError("invalid exchangeInfo payload")
    if not isinstance(tickers, list):
        raise ValueError("invalid ticker/24hr payload")
    return exchange_info, tickers
