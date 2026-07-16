"""Binance Futures symbol discovery and environment-based resolution."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REST_BASE_URL = "https://fapi.binance.com"
MIRROR_REST_BASE_URL = "https://fapi.binancefuture.com"

# Leveraged / synthetic token bases (e.g. BTCUP, ETHDOWN) before the quote asset.
_EXCLUDED_BASE_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def quote_asset_from_env() -> str:
    return (os.environ.get("QUOTE_ASSET") or "USDT").strip().upper()


def symbol_limit_from_env() -> int:
    return max(1, _env_int("SYMBOL_LIMIT", 200))


def manual_symbols_from_env() -> list[str] | None:
    """Return an explicit symbol list from SYMBOLS or legacy BINANCE_SYMBOLS."""
    for key in ("SYMBOLS", "BINANCE_SYMBOLS"):
        raw = os.environ.get(key)
        if raw is None or not raw.strip():
            continue
        symbols = [part.strip().upper() for part in raw.split(",") if part.strip()]
        if symbols:
            return symbols
    return None


def is_excluded_symbol(symbol: str, quote_asset: str) -> bool:
    """Filter leveraged / synthetic contracts (BTCUPUSDT, ETHDOWNUSDT, ...)."""
    sym = str(symbol or "").strip().upper()
    quote = quote_asset.strip().upper()
    if not sym.endswith(quote):
        return True
    base = sym[: -len(quote)]
    if not base:
        return True
    return any(base.endswith(suffix) for suffix in _EXCLUDED_BASE_SUFFIXES)


def _fetch_json(url: str, *, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "analeyes-binance-symbols/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_futures_json(path: str, rest_base_url: str) -> Any:
    last_exc: Exception | None = None
    for base in (rest_base_url.rstrip("/"), MIRROR_REST_BASE_URL):
        url = f"{base}{path}"
        try:
            return _fetch_json(url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Binance REST request failed url=%s reason=%s", url, exc)
            last_exc = exc
    raise RuntimeError(f"Binance REST failed for {path}: {last_exc}") from last_exc


def _tradable_perpetual_symbols(raw_exchange_info: Any, quote_asset: str) -> set[str]:
    if not isinstance(raw_exchange_info, dict):
        raise ValueError("exchangeInfo response is not an object")
    allowed: set[str] = set()
    quote = quote_asset.upper()
    for row in raw_exchange_info.get("symbols") or []:
        if not isinstance(row, dict):
            continue
        if row.get("status") != "TRADING":
            continue
        if row.get("contractType") != "PERPETUAL":
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol.endswith(quote):
            continue
        if is_excluded_symbol(symbol, quote):
            continue
        allowed.add(symbol)
    return allowed


def discover_top_futures_symbols(
    *,
    rest_base_url: str = DEFAULT_REST_BASE_URL,
    quote_asset: str = "USDT",
    limit: int = 200,
) -> list[str]:
    """Discover active USDT perpetuals sorted by 24h quote volume (descending)."""
    quote = quote_asset.strip().upper()
    lim = max(1, int(limit))

    exchange_info = _fetch_futures_json("/fapi/v1/exchangeInfo", rest_base_url)
    tradable = _tradable_perpetual_symbols(exchange_info, quote)

    tickers = _fetch_futures_json("/fapi/v1/ticker/24hr", rest_base_url)
    if isinstance(tickers, dict):
        tickers = [tickers]

    ranked: list[tuple[float, str]] = []
    for row in tickers:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol not in tradable:
            continue
        try:
            quote_volume = float(row.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            quote_volume = 0.0
        ranked.append((quote_volume, symbol))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [symbol for _, symbol in ranked[:lim]]
    if not selected:
        raise RuntimeError(f"No tradable {quote} perpetual symbols discovered")
    return selected


def resolve_binance_symbols(
    *,
    rest_base_url: str | None = None,
    quote_asset: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Resolve symbols from SYMBOLS / allowlist env, or auto-discover top-N by volume."""
    from shared.symbol_universe import resolve_production_symbols

    manual = manual_symbols_from_env()
    if manual:
        symbols = resolve_production_symbols(manual)
        logger.info(
            "Using manual Binance symbol list count=%s symbols=%s",
            len(symbols),
            ",".join(symbols[:5]),
        )
        return symbols

    allowed = resolve_production_symbols()
    if allowed:
        logger.info(
            "Using production allowlist for Binance symbols count=%s symbols=%s",
            len(allowed),
            ",".join(allowed),
        )
        return allowed

    rest = (rest_base_url or os.environ.get("BINANCE_REST_BASE_URL") or DEFAULT_REST_BASE_URL).strip().rstrip("/")
    quote = (quote_asset or quote_asset_from_env()).upper()
    lim = symbol_limit_from_env() if limit is None else max(1, int(limit))

    symbols = discover_top_futures_symbols(rest_base_url=rest, quote_asset=quote, limit=lim)
    logger.info(
        "Discovered top Binance futures symbols quote=%s limit=%s count=%s sample=%s",
        quote,
        lim,
        len(symbols),
        ",".join(symbols[:5]),
    )
    return resolve_production_symbols(symbols)
